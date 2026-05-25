# dl-services/services/t1_vision/object_detector.py

from pathlib import Path

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
import torch


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


class GroundingDINODetector:
    def __init__(self, model_path="/app/models/grounding-dino", device=None):
        """
        Khởi tạo Grounding DINO từ model đã download local.
        model_path: đường dẫn đến thư mục chứa model (grounding-dino)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_path = _resolve_local_model_path(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_path, local_files_only=True, device_map=self.device,
            ignore_mismatched_sizes=True
        )

    def detect(self, image: Image.Image, queries: list, threshold: float = 0.3, text_threshold: float = 0.2):
        """
        Phát hiện đối tượng trong ảnh dựa trên danh sách queries (mô tả văn bản).
        Mỗi query có dạng "a person", "a cup", v.v.
        Trả về danh sách detection.
        """
        # Grounding DINO yêu cầu text dạng string: "a cat. a dog."
        text = ". ".join(queries) + "."

        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Hậu xử lý – chuyển thành bounding box tuyệt đối theo kích thước ảnh gốc
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]]  # (height, width)
        )

        detections = []
        result = results[0]
        for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
            xmin, ymin, xmax, ymax = box.tolist()
            detections.append({
                "label": label,
                "score": round(score.item(), 4),
                "box": {
                    "xmin": round(xmin, 2),
                    "ymin": round(ymin, 2),
                    "xmax": round(xmax, 2),
                    "ymax": round(ymax, 2)
                }
            })

        return detections
