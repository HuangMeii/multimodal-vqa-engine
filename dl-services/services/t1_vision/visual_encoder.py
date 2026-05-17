# dl-services/services/t1_vision/visual_encoder.py
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image

class BLIP2Captioner:
    def __init__(self, model_path="/app/models/blip2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Blip2Processor.from_pretrained(model_path, local_files_only=True)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, device_map=self.device
        )
        self.model.eval()

    def generate_caption(self, image: Image.Image) -> str:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**inputs)
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return caption.strip()
