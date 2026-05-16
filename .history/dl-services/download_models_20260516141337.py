# download_models.py
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

model_id = "IDEA-Research/grounding-dino-tiny"
save_path = "D:/vqa-models/grounding-dino"
save_path = "D:/MyData/Disk D\\Thú vui lười biếng\\vqa-models"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

processor.save_pretrained(save_path)
model.save_pretrained(save_path)
print("Grounding DINO saved.")