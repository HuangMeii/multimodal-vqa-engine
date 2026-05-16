# dl-services/services/t1_vision/visual_encoder.py
from transformers import Blip2Processor, Blip2Model
import torch
from PIL import Image

class BLIP2Encoder:
    def __init__(self, model_path="/app/models/blip2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Blip2Processor.from_pretrained(model_path, local_files_only=True)
        self.model = Blip2Model.from_pretrained(model_path, local_files_only=True, device_map=self.device)
        self.model.eval()

    def encode(self, image: Image.Image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model.get_qformer_features(**inputs)
        return outputs  # tensor (1, 32, 768)