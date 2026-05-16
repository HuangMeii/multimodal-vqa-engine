# dl-services/services/t1_vision/object_detector.py

from transformers import pipeline
from PIL import Image
import torch

class GroundingDINODetector:
    def __init__(self, model_id="IDEA-Research/grounding-dino-base", device=0):
        self.pipeline = pipeline(
            "zero-shot-object-detection",
            model=model_id,
            device=device
        )
    
    def detect(self, image: Image.Image, queries: list, threshold: float = 0.1):
        """
        Trả về danh sách detection, mỗi detection gồm:
        - label (str)
        - score (float)
        - box (dict: xmin, ymin, xmax, ymax)
        """
        results = self.pipeline(image, candidate_labels=queries, threshold=threshold)
        return results