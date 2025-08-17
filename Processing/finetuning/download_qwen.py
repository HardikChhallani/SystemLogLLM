import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# tqdm will be automatically used by huggingface_hub for progress bars
# if it is installed in the environment. No extra code is needed for the download part.

model_id = "Qwen/Qwen2-1.5B-Instruct"
base_model_path = "model/qwen2-1.5b-base" # Central location for the base model

if not os.path.exists(base_model_path):
    print(f"Base model not found. Downloading '{model_id}'...")
    
    # When this line runs, you will see progress bars for each file being downloaded.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto"
    )
    
    # The tokenizer download is usually very fast, but a progress bar will appear if needed.
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print(f"\nSaving model and tokenizer to '{base_model_path}'...")
    model.save_pretrained(base_model_path)
    tokenizer.save_pretrained(base_model_path)
    print("✅ Base model downloaded and saved successfully.")
else:
    print(f"✅ Base model already exists at '{base_model_path}'.")