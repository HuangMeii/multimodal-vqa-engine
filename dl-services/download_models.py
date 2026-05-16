# download_models.py
import os
import sys

MODELS_DIR = sys.argv[1] if len(sys.argv) > 1 else "/app/models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Grounding DINO ──
print("Downloading Grounding DINO...")
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

model_id = "IDEA-Research/grounding-dino-tiny"
save_path = os.path.join(MODELS_DIR, "grounding-dino")
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
processor.save_pretrained(save_path)
model.save_pretrained(save_path)
print(f"  ✓ saved to {save_path}")

# ── BLIP-2 ──
print("Downloading BLIP-2...")
from transformers import Blip2Processor, Blip2Model

model_id = "Salesforce/blip2-opt-2.7b"
save_path = os.path.join(MODELS_DIR, "blip2")
processor = Blip2Processor.from_pretrained(model_id)
model = Blip2Model.from_pretrained(model_id)
processor.save_pretrained(save_path)
model.save_pretrained(save_path)
print(f"  ✓ saved to {save_path}")

# ── Qwen ──
print("Downloading Qwen...")
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
save_path = os.path.join(MODELS_DIR, "qwen")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
print(f"  ✓ saved to {save_path}")

print("\nAll models downloaded successfully!")
