import csv
import os
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["http_proxy"] = "http://hpc-proxy00.city.ac.uk:3128"
os.environ["https_proxy"] = "http://hpc-proxy00.city.ac.uk:3128"


LOG_FILE = "/users/addr777/archive/development/llm/data/output/classification_process_starter_packs.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

INPUT_CSV_FILEPATH = "/users/addr777/archive/development/llm/data/starterpacks.csv"
PROCESSED_URIS_FILE = "/users/addr777/archive/development/llm/data/output/processed_uris_starterpacks_classification.txt"
OUTPUT_FILE = "/users/addr777/archive/development/llm/data/output/classified_starter_packs.csv"

MODEL_PATH = "/users/addr777/archive/development/llm/Mistral-7B-Instruct-v0.3"  
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")

def load_processed_uris():
    """Loads a set of URIs that have already been processed to avoid duplication."""
    if os.path.exists(PROCESSED_URIS_FILE):
        with open(PROCESSED_URIS_FILE, 'r', encoding="utf-8") as f:
            return set(f.read().splitlines())
    return set()

def save_processed_uri(uri):
    """Appends a processed URI to the tracking file."""
    with open(PROCESSED_URIS_FILE, 'a', encoding="utf-8") as f:
        f.write(f"{uri}\n")

def classify_starter_pack(name, description):
    """Uses an LLM model to classify a starter pack and its participants."""
    # Role-based message structure
    messages = [
        {"role": "system", "content": "Classify the starter pack description according to the given rules. No extra text."},
        {"role": "user", "content": f"""
        Based solely on the provided name and description of the starter pack, classify it into a community/category. Do not provide any other text!

        Name: {name}
        Description: {description}
        
        Follow these instructions for the response:

        1. Provide two classifications:
            - The first classification should represent the starter pack itself. If there is insufficient information or if the description is unclear, classify as "unknown".
            - The second classification should represent the participants (e.g., "artists," "musicians," "politicians," "activists," "scientists," "unknown"), as you deem appropriate.
          If there is insufficient information or if the description is unclear, classify as "unknown".
        
        2. Each classification should be a category that represents the core idea of the community or its members. 
            - Do not use two or more words (e.g., "sports or water sports").
            - Avoid ambiguity or overlapping terms. Select only the most appropriate classification based on the description and do not add any other details, just the classification.
            - If the description is unclear or if there is insufficient information, classify as "unknown".
            - Do not provide anything other details alongside the classifications. Not even in brackets (). 
        
        Guidelines: 
        Provide your response in the following format and do not output any other text:
            - Starter Pack Classification: [only a suitable classification for the starter pack or "unknown" and nothing else (no explanations, no justifications, nothing)]
            - Participants Classification: [only a suitable classification for the participants (plural) or "unknown" and nothing else (no explanations, no justifications, nothing)]
        """}
    ]

    # Apply chat template to format the messages
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        # Generate output from the model
        output = model.generate(**inputs, max_new_tokens=100)  # Adjust max_new_tokens as needed

    # Decode the model's response
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # logging.info(f"Model raw response: {response}")

    # The prompt ends with the last line of the instructions, so we can split on the last occurrence of "Guidelines:"
    prompt_end = "Guidelines: \nProvide your response in the following format and do not output any other text:"
    response = response.split(prompt_end)[-1].strip()

    # Extract classifications by searching for the exact phrases
    try:
        starter_pack_classification = None
        participants_classification = None

        # Search the lines of the response for the classifications
        lines = response.split("\n")
        for line in lines:
            # Strip any leading/trailing whitespace to handle formatting inconsistencies
            line = line.strip()
            if line.startswith("Starter Pack Classification:"):
                starter_pack_classification = line.replace("Starter Pack Classification:", "").strip()
            elif line.startswith("Participants Classification:"):
                participants_classification = line.replace("Participants Classification:", "").strip()

        # If either classification is None, set it to "unknown"
        if not starter_pack_classification:
            starter_pack_classification = "unknown"
        if not participants_classification:
            participants_classification = "unknown"

        return starter_pack_classification, participants_classification

    except Exception as e:
        logging.error(f"Error extracting classifications from model response: {e}")
        return "unknown", "unknown"

def process_csv(input_file, output_file):
    """Processes the input CSV file, classifies starter packs, and saves results to the output CSV."""
    processed_uris = load_processed_uris()
    buffer = []  # Temporary buffer for checkpointing
    checkpoint_size = 10  # Save every 10 entries

    with open(input_file, "r", encoding="utf-8-sig") as infile, \
         open(output_file, "a", encoding="utf-8-sig", newline='') as outfile:

        reader = csv.DictReader(infile)
        fieldnames = ['uri', 'name', 'starter_pack_classification', 'participants_classification']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        # If output file is empty, write the header
        if os.stat(output_file).st_size == 0:
            writer.writeheader()

        for row in reader:
            uri = row["uri"].strip()
            name = row["name"].strip()
            description = row["description"].strip()

            # Skip if already processed or missing data
            if not description or uri in processed_uris:
                continue

            classification = classify_starter_pack(name, description)
            buffer.append({
                'uri': uri,
                'name': name,
                'starter_pack_classification': classification[0],
                'participants_classification': classification[1]
            })

            processed_uris.add(uri)  # Mark as processed
            save_processed_uri(uri)  # Save immediately

            # Checkpointing every 10 entries
            if len(buffer) >= checkpoint_size:
                writer.writerows(buffer)
                logging.info(f"Checkpoint: Saved {len(buffer)} classifications to file.")
                buffer.clear()  # Reset buffer

        # Write any remaining entries in the buffer
        if buffer:
            writer.writerows(buffer)
            logging.info(f"Final checkpoint: Saved {len(buffer)} remaining classifications.")

if __name__ == "__main__":
    if os.path.exists(INPUT_CSV_FILEPATH):
        logging.info("Starting classification process...")
        process_csv(INPUT_CSV_FILEPATH, OUTPUT_FILE)
        logging.info("Classification process completed.")
    else:
        logging.error(f"Input CSV file not found: {INPUT_CSV_FILEPATH}")