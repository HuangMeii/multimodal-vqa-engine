# dl-services/routers/learn.py
# API học tiếng Anh: 1 ảnh → nhiều caption + dịch nghĩa

from fastapi import APIRouter, UploadFile, File, Form
from PIL import Image
import io
import json
import re
import torch

router = APIRouter()

# Lazy loading: models chỉ được load khi cần, không load ở startup
_captioner = None
_detector = None
_llm = None


def get_captioner():
    global _captioner
    if _captioner is None:
        from services.t1_vision.florence2_captioner import Florence2Captioner
        _captioner = Florence2Captioner()
    return _captioner


def get_detector():
    global _detector
    if _detector is None:
        from services.t1_vision.object_detector import GroundingDINODetector
        _detector = GroundingDINODetector()
    return _detector


def get_llm():
    global _llm
    if _llm is None:
        from services.t2_reasoning.qwen import QwenLLM
        _llm = QwenLLM()
    return _llm


@router.post("/api/v1/detect-objects")
async def detect_objects(image: UploadFile = File(...)):
    """
    Bước 1: Detect objects trong ảnh bằng Grounding DINO.
    Trả về danh sách objects để user chọn.
    """
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Detect với danh sách queries phổ biến
    common_objects = [
        "a person", "a cat", "a dog", "a car", "a table", "a chair",
        "a book", "a phone", "a bottle", "a cup", "a laptop",
        "a bird", "a tree", "a flower", "a ball", "a bag"
    ]
    detector = get_detector()
    detections = detector.detect(img, common_objects, threshold=0.25)

    # Gom nhóm theo label, lấy box có score cao nhất
    seen = {}
    for det in detections:
        label = det["label"]
        if label not in seen or det["score"] > seen[label]["score"]:
            seen[label] = det

    objects = []
    for label, det in seen.items():
        objects.append({
            "label": label,
            "confidence": det["score"],
            "bbox": det["box"]
        })

    # Giải phóng VRAM sau Grounding DINO
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"objects": objects}


@router.post("/api/v1/learn")
async def learn_from_image(
    image: UploadFile = File(...),
    target_object: str = Form(""),
):
    """
    Bước 2: Học tiếng Anh từ ảnh + object đã chọn.
    - Florence-2: sinh caption
    - Qwen: rewrite + translate + format JSON
    """
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # === 1. Florence-2: sinh nhiều caption ===
    captioner = get_captioner()
    caption = captioner.generate_caption(img)
    detailed_caption = captioner.generate_detailed_caption(img)
    od_result = captioner.generate_od(img)

    # Giải phóng VRAM sau Florence-2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # === 2. Qwen: rewrite + translate + format ===

    prompt = f"""You are an English learning assistant. Given an image description and a target object, generate:

1. Multiple English sentences describing the image (different grammar structures)
2. Vietnamese translation for each sentence
3. Key vocabulary with meanings

Image captions:
- Short caption: "{caption}"
- Detailed caption: "{detailed_caption}"
- Objects detected: "{od_result}"

Target object: "{target_object if target_object else 'the whole scene'}"

Output ONLY valid JSON (no markdown, no explanation):
{{
  "sentences": [
    {{"en": "...", "vi": "..."}},
    {{"en": "...", "vi": "..."}},
    {{"en": "...", "vi": "..."}}
  ],
  "vocabulary": [
    {{"word": "...", "meaning": "..."}},
    {{"word": "...", "meaning": "..."}}
  ]
}}

Generate 3-5 sentences with different grammar (simple present, present continuous, there is/are, preposition, adjective). Translate naturally to Vietnamese."""

    llm = get_llm()
    raw_output = llm.generate(prompt, max_new_tokens=512)

    # === 3. Parse JSON từ output ===
    try:
        # Tìm JSON trong output (có thể bị wrap trong markdown)
        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(raw_output)
    except (json.JSONDecodeError, AttributeError):
        # Fallback nếu Qwen không ra JSON
        result = {
            "sentences": [
                {"en": caption, "vi": f"(Tự động dịch) {caption}"}
            ],
            "vocabulary": []
        }

    # Đảm bảo có đủ các field
    result.setdefault("sentences", [])
    result.setdefault("vocabulary", [])

    return result
