# dl-services/routers/detection.py

from fastapi import APIRouter, UploadFile, File, Form
from PIL import Image
import io
import base64
from services.t1_vision.object_detector import GroundingDINODetector
from services.t1_vision.crop_utils import crop_image

router = APIRouter()

# Khởi tạo detector toàn cục (load model 1 lần)
detector = GroundingDINODetector()

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
    detections = detector.detect(img, query_list)
    
    # Xây dựng response
    results = []
    for det in detections:
        box = det['box']
        # Crop ảnh
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
# dl-services/routers/detection.py

from fastapi import APIRouter, UploadFile, File
from PIL import Image
import io
from services.t1_vision.object_detector import GroundingDINODetector, Blip2Captioner

router = APIRouter()

# Khởi tạo model toàn cục
detector = GroundingDINODetector()
captioner = Blip2Captioner()

@router.post("/detect")
async def detect_objects(file: UploadFile = File(...), queries: list[str] = []):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    detections = detector.detect(image, queries)
    return {"detections": detections}

@router.post("/caption")
async def caption_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    caption = captioner.caption(image)
    return {"caption": caption}
from fastapi import APIRouter, UploadFile, File
from PIL import Image
import io
from services.t1_vision.object_detector import GroundingDINODetector, Blip2Captioner
from services.t2_reasoning.qwen import QwenLLM   # import Qwen

router = APIRouter()

# Khởi tạo model toàn cục
detector = GroundingDINODetector()
captioner = Blip2Captioner()
qwen = QwenLLM()

@router.post("/detect")
async def detect_objects(file: UploadFile = File(...), queries: list[str] = []):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    detections = detector.detect(image, queries)
    return {"detections": detections}

@router.post("/caption")
async def caption_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    caption = captioner.caption(image)
    return {"caption": caption}

@router.post("/chat")
async def chat_with_qwen(prompt: str):
    response = qwen.generate(prompt)
    return {"response": response}
