import os
import subprocess

model_path = os.path.join(os.path.dirname(__file__), "..", "models", "gpt2-124m", "model.safetensors")

if not os.path.exists(model_path):
    print("weights not found, downloading...")
    subprocess.run(["python3", os.path.join(os.path.dirname(__file__), "download_weights.py")], check=True)

subprocess.run(["python3", os.path.join(os.path.dirname(__file__), "server.py")], check=True)
