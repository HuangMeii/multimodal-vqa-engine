from fastapi import APIRouter, UploadFile, File
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


@router.post("/api/v1/caption")
async def caption_image(image: UploadFile = File(...)):
    """
    Nhận ảnh, trả về mô tả tự nhiên bằng Florence-2.
    """
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    captioner = get_captioner()
    caption = captioner.generate_caption(img)

    # Giải phóng VRAM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"caption": caption, "model": "Florence-2-large"}
