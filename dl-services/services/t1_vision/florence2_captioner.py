# dl-services/services/t1_vision/florence2_captioner.py
"""Vision wrapper for realtime backend.

The class name stays for compatibility with existing imports, but the
implementation now uses YOLOv8 for detection and BLIP for captioning.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image


def _resolve_model_ref(model_ref: str | None, fallback: str) -> str:
    if not model_ref:
        return fallback

    path = Path(model_ref)
    if path.exists():
        return str(path)

    return model_ref


def _image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class Florence2Captioner:
    def __init__(
        self,
        yolo_model_path: str | None = None,
        caption_model_path: str | None = None,
        device: str | None = None,
        detector=None,
        processor=None,
        caption_model=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_model_path = _resolve_model_ref(
            yolo_model_path or os.getenv("YOLOV8_MODEL_PATH"),
            fallback="yolov8n.pt",
        )
        self.caption_model_path = _resolve_model_ref(
            caption_model_path or os.getenv("TINYBLIP_MODEL_PATH"),
            fallback="Salesforce/blip-image-captioning-base",
        )
        self.confidence = float(os.getenv("YOLOV8_CONFIDENCE", "0.25"))
        self.max_det = int(os.getenv("YOLOV8_MAX_DET", "10"))

        self._detector = detector
        self._processor = processor
        self._caption_model = caption_model

    def _get_detector(self):
        if self._detector is None:
            from ultralytics import YOLO

            self._detector = YOLO(self.yolo_model_path)
        return self._detector

    def _get_processor(self):
        if self._processor is None:
            from transformers import BlipProcessor

            self._processor = BlipProcessor.from_pretrained(self.caption_model_path)
        return self._processor

    def _get_caption_model(self):
        if self._caption_model is None:
            from transformers import BlipForConditionalGeneration

            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            self._caption_model = BlipForConditionalGeneration.from_pretrained(
                self.caption_model_path,
                local_files_only=False,
                torch_dtype=torch_dtype,
            )
            self._caption_model.to(self.device)
            self._caption_model.eval()
        return self._caption_model

    def _run_caption(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        processor = self._get_processor()
        model = self._get_caption_model()

        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
            )

        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return caption.replace(prompt, "").strip()

    def generate_caption(self, image: Image.Image) -> str:
        return self._run_caption(image, prompt="a photo of", max_new_tokens=40)

    def generate_detailed_caption(self, image: Image.Image) -> str:
        return self._run_caption(
            image,
            prompt="Describe the image in detail.",
            max_new_tokens=80,
        )

    def _normalize_requested_labels(self, queries: str | Iterable[str] | None) -> set[str]:
        if queries is None:
            return set()
        if isinstance(queries, str):
            parts = [part.strip().lower() for part in queries.split(",")]
            return {part for part in parts if part}
        return {str(part).strip().lower() for part in queries if str(part).strip()}

    def detect_objects(self, image: Image.Image, queries: str | Iterable[str] | None = None) -> list[dict]:
        detector = self._get_detector()
        requested_labels = self._normalize_requested_labels(queries)

        results = detector.predict(
            source=image,
            conf=self.confidence,
            device=self.device,
            verbose=False,
            max_det=self.max_det,
        )

        if not results:
            return []

        result = results[0]
        detections: list[dict] = []
        names = getattr(result, "names", None) or getattr(detector, "names", {})

        for box in getattr(result, "boxes", []):
            cls_id = int(box.cls.item()) if hasattr(box, "cls") else -1
            if isinstance(names, dict):
                label = names.get(cls_id, f"class_{cls_id}")
            elif isinstance(names, list) and 0 <= cls_id < len(names):
                label = names[cls_id]
            else:
                label = f"class_{cls_id}"

            if requested_labels and label.lower() not in requested_labels:
                continue

            confidence = float(box.conf.item()) if hasattr(box, "conf") else 0.0
            x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = max(x2, x1)
            y2 = max(y2, y1)

            crop_base64 = ""
            if x2 > x1 and y2 > y1:
                crop = image.crop((x1, y1, x2, y2))
                crop_base64 = _image_to_base64(crop)

            detections.append(
                {
                    "label": label,
                    "confidence": round(confidence, 4),
                    "bbox": {
                        "xmin": x1,
                        "ymin": y1,
                        "xmax": x2,
                        "ymax": y2,
                    },
                    "crop_base64": crop_base64,
                }
            )

        return detections

    def generate_od(self, image: Image.Image, queries: str | Iterable[str] | None = None) -> list[dict]:
        return self.detect_objects(image, queries=queries)
