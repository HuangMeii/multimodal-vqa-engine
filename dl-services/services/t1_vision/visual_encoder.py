# dl-services/services/t1_vision/visual_encoder.py
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from PIL import Image

class BLIP2Encoder:
    def __init__(self, model_path="/app/models/blip2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True
        ).to(self.device)
        self.model.eval()

    def encode(self, image: Image.Image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        # Lấy vision output rồi qua qformer để lấy features
        vision_outputs = self.model.vision_model(**inputs)
        qformer_outputs = self.model.qformer(
            inputs_embeds=vision_outputs.last_hidden_state,
            attention_mask=inputs.get("pixel_values"),
        )
        return qformer_outputs
