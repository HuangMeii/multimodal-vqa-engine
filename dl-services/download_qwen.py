"""
Script tải Qwen model với đúng transformers version (4.44.0) 
để tương thích với Docker environment.

Cách chạy:
    pip install transformers==4.44.0
    python download_qwen.py
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
SAVE_PATH = "D:/MyData/Disk D/Thú vui lười biếng/vqa-models/qwen"

print(f"Downloading {MODEL_ID} ...")
print(f"Save path: {SAVE_PATH}")
print(f"Transformers version: {__import__('transformers').__version__}")

# Download tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Download model - dùng float32 để tương thích CPU
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
)

# Lưu model
os.makedirs(SAVE_PATH, exist_ok=True)
tokenizer.save_pretrained(SAVE_PATH)
model.save_pretrained(SAVE_PATH)

print("Qwen downloaded and saved successfully!")
