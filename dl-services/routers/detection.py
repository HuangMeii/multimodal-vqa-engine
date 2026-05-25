# dl-services/routers/detection.py

from fastapi import APIRouter, UploadFile, File, Form
from PIL import Image
import io
import base64
import torch

router = APIRouter()

# Lazy loading: Grounding DINO chỉ load khi cần
_detector = None


def get_detector():
    global _detector
    if _detector is None:
        from services.t1_vision.object_detector import GroundingDINODetector
        _detector = GroundingDINODetector()
    return _detector


@router.post("/api/v1/detect")
async def detect_objects(
    image: UploadFile = File(...),
    queries: str = Form("a person, a cup, a table, a chair")
):
    """
    Nhận ảnh và danh sách đối tượng (cách nhau bởi dấu phẩy).
    Trả về danh sách các đối tượng với bounding box và ảnh crop base64.
    """
    # Đọc ảnh
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    # Tách danh sách queries
    query_list = [q.strip() for q in queries.split(",") if q.strip()]
    
    # Chạy detection
    detector = get_detector()
    detections = detector.detect(img, query_list)
    
    # Giải phóng VRAM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Xây dựng response
    results = []
    for det in detections:
        box = det['box']
        # Crop ảnh
        from services.t1_vision.crop_utils import crop_image
        cropped = crop_image(img, box)
        # Chuyển crop sang base64
        buffered = io.BytesIO()
        cropped.save(buffered, format="JPEG")
        crop_b64 = base64.b64encode(buffered.getvalue()).decode()
        
        results.append({
            "label": det['label'],
            "confidence": round(det['score'], 4),
            "bbox": {
                "xmin": round(box['xmin'], 2),
                "ymin": round(box['ymin'], 2),
                "xmax": round(box['xmax'], 2),
                "ymax": round(box['ymax'], 2)
            },
            "crop_base64": crop_b64
        })
    
    return {"objects": results}
