# dl-services/services/t1_vision/object_detector.py
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
from PIL import Image

class GroundingDINODetector:
    def __init__(self, model_path="/app/models/grounding-dino", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_path, local_files_only=True, device_map=self.device
        )

    # ... phần detect giữ nguyên