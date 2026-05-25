from fastapi import APIRouter, UploadFile, File, Form
from PIL import Image
import io
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


@router.post("/api/v1/florence2/caption")
async def florence2_caption(image: UploadFile = File(...)):
    """
    Nhận ảnh, trả về caption tổng quát bằng Florence-2 (<CAPTION>).
    """
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    captioner = get_captioner()
    caption = captioner.generate_caption(img)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"caption": caption, "model": "Florence-2-large"}


@router.post("/api/v1/florence2/detailed-caption")
async def florence2_detailed_caption(image: UploadFile = File(...)):
    """
    Nhận ảnh, trả về caption chi tiết bằng Florence-2 (<DETAILED_CAPTION>).
    """
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    captioner = get_captioner()
    caption = captioner.generate_detailed_caption(img)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"caption": caption, "model": "Florence-2-large"}


@router.post("/api/v1/florence2/od")
async def florence2_object_detection(image: UploadFile = File(...)):
    """
    Nhận ảnh, trả về kết quả object detection bằng Florence-2 (<OD>).
    """
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    captioner = get_captioner()
    result = captioner.generate_od(img)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"objects": result, "model": "Florence-2-large"}
