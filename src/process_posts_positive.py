import csv
import os
import logging
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["http_proxy"] = "http://hpc-proxy00.city.ac.uk:3128"
os.environ["https_proxy"] = "http://hpc-proxy00.city.ac.uk:3128"

LOG_FILE = "/users/addr777/archive/development/llm/data/output/classification_process_positive.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

NEGATIVE_POSTS_FILE = "/users/addr777/archive/development/llm/data/positive_posts.csv"
OUTPUT_FILE = "/users/addr777/archive/development/llm/data/output/classified_posts_posts_now.csv"
PROCESSED_FILE = "/users/addr777/archive/development/llm/data/processed_posts_positive.txt" 

model_path = "/users/addr777/archive/development/llm/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_processed_posts():
    """
    Load the set of processed posts from a file to avoid reprocessing.
    """
    if os.path.exists(PROCESSED_FILE):
        with open(PROCESSED_FILE, 'r') as f:
            processed_posts = set(f.read().splitlines())
    else:
        processed_posts = set()
    return processed_posts

def save_processed_post(post_id):
    """
    Save a processed post ID to prevent reprocessing.
    """
    with open(PROCESSED_FILE, 'a') as f:
        f.write(f"{post_id}\n")

def classify_post(post):
    """ Classifies a positive post using Mistral-7B-Instruct, ensuring only the category is returned. """
    messages = [
        {"role": "system", "content": "Classify the post into one of the categories. Respond in the format: 'Category: [Number. Category Name]'. No extra text."},
        {"role": "user", "content": f"Post: {post}\n\nCategories:\n"
            "1. Praising a Starter Pack or Starter Packs in General\n"
            "2. Explaining How the System Works or Reporting Starter Pack Experience\n"
            "3. Desire to Be Added to a Starter Pack\n"
            "4. Advertising a Starter Pack (including asking for members or inviting others to join)\n"
            "5. Expressing Frustration with the Current System (e.g., mass follow but zero engagement)\n"
            "6. Added Without Permission\n"
            "7. Suggesting Someone Create a New Starter Pack\n"
            "8. Asking for Help (e.g., understanding how the system works or looking for a specific Starter Pack)\n"
            "9. Asking for money to include someone in a starter pack\n"
            "10. Other\n\n"
            "Respond strictly as:\nCategory: [Number. Category Name]"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=20)

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    match = re.search(r"Category:\s*(\d+\.\s*.+)", response)
    return match.group(1).strip() if match else "9. Other"

def process_posts(input_file, output_file, max_posts=20):
    """
    Reads posts from the input CSV, classifies the first `max_posts`, writes results to a new CSV, 
    and tracks processed posts to avoid repetition.
    """
    processed_count = 0
    processed_posts = load_processed_posts()  

    with open(input_file, mode='r', encoding='utf-8') as infile, \
         open(output_file, mode='a', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        fieldnames = ['did', 'rkey', 'post_text', 'category']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        if os.stat(output_file).st_size == 0:
            writer.writeheader()

        for row in reader:
            did = row.get('did', '').strip()
            rkey = row.get('rkey', '').strip()
            post_text = row.get('post_text', '').strip()

            if not post_text:
                logging.warning(f"Skipping row with empty post_text: {row}")
                continue

            if did in processed_posts:
                logging.info(f"Post with DID {did} already processed, skipping.")
                continue

            classification = classify_post(post_text)
            print(f"Post classified as: {classification}")
            logging.info(f"Classified post for DID: {did}, RKEY: {rkey}, Category: {classification}")
            writer.writerow({'did': did, 'rkey': rkey, 'post_text': post_text, 'category': classification})

            save_processed_post(did)
            processed_count += 1

            if processed_count % 10 == 0:
                logging.info(f"Processed {processed_count} posts so far.")

if __name__ == "__main__":
    if os.path.exists(NEGATIVE_POSTS_FILE):
        logging.info("Starting classification process (Test run: first 20 posts).")
        process_posts(NEGATIVE_POSTS_FILE, OUTPUT_FILE, max_posts=20)
        logging.info("Test classification process completed successfully.")
    else:
        logging.error(f"Input file not found: {NEGATIVE_POSTS_FILE}")