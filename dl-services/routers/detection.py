# dl-services/routers/detection.py
# Object detection using YOLOv8.

from fastapi import APIRouter, UploadFile, File, Form
from PIL import Image
import io
import base64
import torch

router = APIRouter()

# Lazy loading: YOLOv8 chỉ load khi cần
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
    Dùng YOLOv8 để phát hiện vật thể.
    Trả về danh sách các đối tượng với bounding box và ảnh crop base64.
    """
    # Đọc ảnh
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Chạy OD với YOLOv8
    captioner = get_captioner()
    od_result = captioner.detect_objects(img, queries=queries)

    # Giải phóng VRAM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"objects": od_result or [], "model": "YOLOv8"}
