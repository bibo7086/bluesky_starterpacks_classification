#!/bin/bash
#SBATCH -D /users/addr777/archive/development/llm      # Working directory
#SBATCH --job-name=starter_pack        # Job name
#SBATCH --partition=preemptgpu             # GPU partition
#SBATCH --nodes=1                      # Use 1 node
#SBATCH --ntasks-per-node=1            # Run 1 task
#SBATCH --cpus-per-task=4              # Request 4 CPU cores
#SBATCH --mem=32GB                     # Allocate 24GB CPU RAM
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --time=72:00:00                # Time limit (1 hour)
#SBATCH -o results/%x_%j.o             # Save stdout in results folder
#SBATCH -e results/%x_%j.e             # Save stderr in results folder

# Load HPC environment
source /opt/flight/etc/setup.sh
flight env activate gridware
module purge
module add gnu

# Ensure Pyenv is correctly set up
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Pyenv automatically picks up `llam3_env` due to `.python-version` in `myfolder`
which python  # Should output ~/.pyenv/shims/python
python --version  # Should output Python 3.10.14

# Run the LLaMA inference script
python classify_starter_packs.py