#!/bin/bash
set -euo pipefail

# --- 0. Environment Setup ---
echo "Starting provisioning script."

# --- 1. Install Git ---
echo "Installing git..."
apt-get update && apt-get install -y git

# --- 2. Install Poetry ---
echo "Installing poetry..."
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# Verify poetry installation
poetry --version

# --- 3. Set up SSH deploy key ---
echo "Setting up SSH key for Git access..."
mkdir -p ~/.ssh
echo "$SSH_PRIVATE_KEY" | base64 -d > ~/.ssh/id_rsa
chmod 600 ~/.ssh/id_rsa
# Add github to known hosts to avoid authenticity prompts
ssh-keyscan github.com >> ~/.ssh/known_hosts

# --- 4. Clone main project ---
echo "Cloning main repository..."
git clone "$GIT_REPO_URL"
REPO_NAME=$(basename "$GIT_REPO_URL" .git)

cd "$REPO_NAME"

# --- 5. Install Submodules ---
echo "Cloning submodules..."
git submodule update --init --recursive

# --- 6. Install main project and submodules with Poetry ---
echo "Installing main project in editable mode..."
poetry install

# Now install submodules as editable dependencies
for submodule_path in submodules/*; do
    if [ -d "$submodule_path" ]; then
        echo "Installing submodule $submodule_path in editable mode..."
        poetry add --editable "$submodule_path"
    fi
done

# --- 7. Ensure all dependencies are installed ---
echo "Ensuring all dependencies are installed..."
poetry install

# Weird bug where nnsight isn't present
echo "Attempting to add nnsight"
poetry add nnsight
echo "Successfully added nnsight"

# --- 8. Install WandB and Huggingface CLI ---
echo "Installing wandb and huggingface-cli..."
poetry run pip install wandb huggingface_hub

# --- 9. Authenticate WandB and Huggingface ---
echo "Logging into WandB and Huggingface..."

# For wandb: the environment variable WANDB_API_KEY must be set
poetry run wandb login --relogin "$WANDB_API_KEY"

# For Huggingface: the environment variable HUGGINGFACE_TOKEN must be set
poetry run huggingface-cli login --token "$HF_TOKEN"

#Automatically activate poetry environment
source "$(poetry env info --path)/bin/activate"


echo "Setup complete"
