# dl-services/services/t1_vision/visual_encoder.py
from pathlib import Path

from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image


def _resolve_local_model_path(model_path: str) -> str:
    path = Path(model_path)
    if path.is_dir() and (path / "preprocessor_config.json").exists() and (path / "config.json").exists():
        return str(path)

    if path.is_dir():
        for candidate in path.glob("**/preprocessor_config.json"):
            candidate_dir = candidate.parent
            if (candidate_dir / "config.json").exists():
                return str(candidate_dir)

    return model_path

class BLIP2Captioner:
    def __init__(self, model_path="/app/models/blip2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_path = _resolve_local_model_path(model_path)
        self.processor = Blip2Processor.from_pretrained(model_path, local_files_only=True)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device
        )
        self.model.eval()

    def generate_caption(self, image: Image.Image) -> str:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**inputs)
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return caption.strip()
