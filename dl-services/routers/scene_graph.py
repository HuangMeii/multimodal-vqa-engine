from fastapi import APIRouter, UploadFile, File
from PIL import Image
import io
from services.t1_vision.visual_encoder import BLIP2Captioner

router = APIRouter()

captioner = BLIP2Captioner()   # load model một lần

@router.post("/api/v1/scenegraph")
async def scene_graph(image: UploadFile = File(...)):
    """
    Nhận ảnh, trả về caption từ BLIP‑2 (tạm gọi là scene graph).
    """
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    caption = captioner.generate_caption(img)
    return {"caption": caption}