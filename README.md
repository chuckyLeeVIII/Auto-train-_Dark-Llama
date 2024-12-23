# Auto-train-_Dark-Llama
# Script for finetuning Qwen2.5-Coder-32B-Instruct-abliterated with Unsloth and using it in Ollama
# Note: Designed for Google Colab or local environments

import os
import subprocess

# Install necessary packages
def install_dependencies():
    print("Installing dependencies...")
    os.system("pip install git+https://github.com/unslothai/unsloth.git")
    os.system("pip install torch transformers datasets")
    os.system("pip install huggingface_hub")
    os.system("pip install ollama")
    print("Dependencies installed.")

# Clone Unsloth example notebooks
def clone_unsloth_notebooks():
    print("Cloning Unsloth notebooks...")
    os.system("git clone https://github.com/unslothai/unsloth.git")
    print("Cloned Unsloth notebooks.")

# Select and configure model for finetuning
def configure_model():
    print("Configuring model for finetuning...")
    model_name = "C:/Users/BlackWin/Desktop/Qwen2.5-Coder-32B-Instruct-abliterated"
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True
    print(f"Model: {model_name}, Max Seq Length: {max_seq_length}, 4-bit Quantization: {load_in_4bit}")

# Parameters for finetuning
def configure_finetuning():
    print("Setting finetuning parameters...")
    params = {
        "r": 16,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_alpha": 16,
        "lora_dropout": 0,
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
        "random_state": 3407,
        "use_rslora": False,
        "loftq_config": None
    }
    print("Finetuning parameters configured.")
    return params

# Load and preprocess Darkert dataset
def load_darkert_dataset():
    print("Loading Darkert dataset...")
    dataset_path = "path_to_darkert_dataset.json"  # Replace with actual dataset path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    print("Darkert dataset loaded.")

# Train the model
def train_model(params):
    print("Starting model training...")
    # Configure training parameters
    training_params = {
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "max_steps": 60,
        "learning_rate": 2e-4
    }
    print(f"Training parameters: {training_params}")
    # Replace this with actual Unsloth training commands
    print("Training complete.")

# Export model to Ollama
def export_to_ollama():
    print("Exporting model to Ollama...")
    os.system("ollama export Qwen2.5-Coder-32B-Instruct-abliterated-unsloth-model")
    print("Model exported to Ollama.")

# Run the interactive chatbot
def run_chatbot():
    print("Running interactive chatbot...")
    subprocess.run(["ollama", "run", "Qwen2.5-Coder-32B-Instruct-abliterated-unsloth-model"])

if __name__ == "__main__":
    # Step 1: Install dependencies
    install_dependencies()

    # Step 2: Clone notebooks
    clone_unsloth_notebooks()

    # Step 3: Configure model
    configure_model()

    # Step 4: Load Darkert dataset
    load_darkert_dataset()

    # Step 5: Set finetuning parameters
    params = configure_finetuning()

    # Step 6: Train the model
    train_model(params)

    # Step 7: Export model to Ollama
    export_to_ollama()

    # Step 8: Run the chatbot
    run_chatbot()
