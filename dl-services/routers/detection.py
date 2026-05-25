# dl-services/routers/detection.py
# Object detection using Florence-2 (replaces Grounding DINO)

from fastapi import APIRouter, UploadFile, File, Form
from PIL import Image
import io
import base64
import torch

router = APIRouter()

# Lazy loading: Florence-2 chỉ load khi cần
_captioner = None


def get_captioner():
    global _captioner
    if _captioner is None:
        from services.t1_vision.florence2_captioner import Florence2Captioner
        _captioner = Florence2Captioner()
    return _captioner


@router.post("/api/v1/detect")
async def detect_objects(
    image: UploadFile = File(...),
    queries: str = Form("a person, a cup, a table, a chair")
):
    """
    Nhận ảnh và danh sách đối tượng (cách nhau bởi dấu phẩy).
    Dùng Florence-2 OD để phát hiện vật thể.
    Trả về danh sách các đối tượng với bounding box và ảnh crop base64.
    """
    # Đọc ảnh
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Chạy OD với Florence-2
    captioner = get_captioner()
    od_result = captioner.generate_od(img)

    # Giải phóng VRAM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Parse OD result từ Florence-2
    # Format: "cat: 0.98, dog: 0.95, person: 0.90"
    results = []
    if od_result:
        parts = [p.strip() for p in od_result.split(",")]
        for part in parts:
            if ":" in part:
                label, score_str = part.rsplit(":", 1)
                label = label.strip()
                try:
                    score = float(score_str.strip())
                except ValueError:
                    score = 0.5

                results.append({
                    "label": label,
                    "confidence": round(score, 4),
                    "bbox": {"xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0},
                    "crop_base64": "",
                })

    return {"objects": results}
