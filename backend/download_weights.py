import os
from huggingface_hub import hf_hub_download

MODEL_ID = "openai-community/gpt2"
FILES = ["model.safetensors", "config.json"]

def main():
    os.makedirs("models/gpt2-124m", exist_ok=True)
    for filename in FILES:
        path = hf_hub_download(repo_id=MODEL_ID, filename=filename, local_dir="models/gpt2-124m")
        print(f"downloaded {filename} -> {path}")

if __name__ == "__main__":
    main()
