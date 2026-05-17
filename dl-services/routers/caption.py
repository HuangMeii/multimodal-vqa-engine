from fastapi import APIRouter, UploadFile, File
from PIL import Image
import io
from services.t1_vision.visual_encoder import BLIP2Captioner

router = APIRouter()

# Khởi tạo BLIP-2 captioner (load 1 lần)
captioner = BLIP2Captioner()


@router.post("/api/v1/caption")
async def caption_image(image: UploadFile = File(...)):
    """
    Nhận ảnh, trả về mô tả tự nhiên bằng BLIP-2.
    """
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    caption = captioner.generate_caption(img)

    return {"caption": caption}
