# dl-services/routers/learn.py
# API học tiếng Anh: 1 ảnh → Florence-2 caption + OD → TinyLlama dịch

from fastapi import APIRouter, UploadFile, File, Form
from PIL import Image
import io
import json
import re
import torch

router = APIRouter()

# Lazy loading
_captioner = None
_translator = None


def get_captioner():
    global _captioner
    if _captioner is None:
        from services.t1_vision.florence2_captioner import Florence2Captioner
        _captioner = Florence2Captioner()
    return _captioner


def get_translator():
    global _translator
    if _translator is None:
        from services.t2_reasoning.tinyllama_translator import TinyLlamaTranslator
        _translator = TinyLlamaTranslator()
    return _translator


# Template câu tiếng Anh mẫu cho các object phổ biến
ENGLISH_SENTENCE_TEMPLATES = {
    "person": [
        "There is a person in the image.",
        "The person is standing near the {object}.",
        "I can see a person in the picture.",
    ],
    "cat": [
        "There is a cat in the image.",
        "The cat is sitting on the {object}.",
        "I can see a cute cat.",
    ],
    "dog": [
        "There is a dog in the image.",
        "The dog is playing near the {object}.",
        "I can see a brown dog.",
    ],
    "car": [
        "There is a car in the image.",
        "The car is parked near the {object}.",
        "I can see a red car.",
    ],
    "table": [
        "There is a table in the image.",
        "The {object} is on the table.",
        "The table is made of wood.",
    ],
    "chair": [
        "There is a chair in the image.",
        "The chair is next to the {object}.",
        "I can see a comfortable chair.",
    ],
    "book": [
        "There is a book in the image.",
        "The book is on the {object}.",
        "I am reading an interesting book.",
    ],
    "phone": [
        "There is a phone in the image.",
        "The phone is on the {object}.",
        "I am using my phone.",
    ],
    "bottle": [
        "There is a bottle in the image.",
        "The bottle is on the {object}.",
        "I can see a water bottle.",
    ],
    "cup": [
        "There is a cup in the image.",
        "The cup is on the {object}.",
        "I am drinking from a cup.",
    ],
    "laptop": [
        "There is a laptop in the image.",
        "The laptop is on the {object}.",
        "I am working on my laptop.",
    ],
    "bird": [
        "There is a bird in the image.",
        "The bird is flying in the sky.",
        "I can see a beautiful bird.",
    ],
    "tree": [
        "There is a tree in the image.",
        "The tree is tall and green.",
        "The {object} is under the tree.",
    ],
    "flower": [
        "There is a flower in the image.",
        "The flower is beautiful and colorful.",
        "The {object} is near the flower.",
    ],
    "ball": [
        "There is a ball in the image.",
        "The ball is on the {object}.",
        "The children are playing with a ball.",
    ],
    "bag": [
        "There is a bag in the image.",
        "The bag is next to the {object}.",
        "I carry my books in a bag.",
    ],
}

# Template mặc định nếu object không có trong danh sách
DEFAULT_TEMPLATES = [
    "There is a {object} in the image.",
    "The {object} is near the {other_object}.",
    "I can see a {object} in the picture.",
]


def _get_sentence_templates(target_object: str, other_objects: list[str] = None) -> list[str]:
    """Lấy template câu cho object, fallback về default nếu không có."""
    obj_lower = target_object.lower()
    other = other_objects[0] if other_objects else "table"

    if obj_lower in ENGLISH_SENTENCE_TEMPLATES:
        templates = ENGLISH_SENTENCE_TEMPLATES[obj_lower]
    else:
        templates = DEFAULT_TEMPLATES

    # Format templates với object name
    result = []
    for t in templates:
        result.append(t.replace("{object}", target_object).replace("{other_object}", other))
    return result


@router.post("/api/v1/detect-objects")
async def detect_objects(image: UploadFile = File(...)):
    """
    Phát hiện vật thể trong ảnh bằng Florence-2 OD.
    Trả về danh sách objects để user chọn.
    """
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    captioner = get_captioner()
    od_result = captioner.generate_od(img)

    # Parse OD result từ Florence-2
    # Format: "cat: 0.98, dog: 0.95, person: 0.90"
    objects = []
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
                objects.append({
                    "label": label,
                    "confidence": round(score, 4),
                })

    # Giải phóng VRAM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"objects": objects}


@router.post("/api/v1/learn")
async def learn_from_image(
    image: UploadFile = File(...),
    target_object: str = Form(""),
):
    """
    Học tiếng Anh từ ảnh + object đã chọn.
    - Florence-2: sinh caption
    - TinyLlama: dịch câu
    """
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # === 1. Florence-2: sinh caption ===
    captioner = get_captioner()
    caption = captioner.generate_caption(img)
    detailed_caption = captioner.generate_detailed_caption(img)

    # Giải phóng VRAM sau Florence-2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # === 2. Sinh câu tiếng Anh từ template ===
    sentences_en = _get_sentence_templates(target_object if target_object else "scene")

    # === 3. TinyLlama: dịch câu ===
    translator = get_translator()
    sentences_vi = translator.translate_batch(sentences_en)

    # === 4. Xây dựng vocabulary từ caption ===
    vocabulary = _extract_vocabulary(caption, target_object)

    # === 5. Format kết quả ===
    sentences = []
    for en, vi in zip(sentences_en, sentences_vi):
        sentences.append({"en": en, "vi": vi})

    return {
        "sentences": sentences,
        "vocabulary": vocabulary,
        "caption": caption,
        "detailed_caption": detailed_caption,
    }


def _extract_vocabulary(caption: str, target_object: str) -> list[dict]:
    """Trích xuất từ vựng từ caption và target object."""
    vocab = []

    # Thêm target object
    if target_object:
        vocab.append({
            "word": target_object,
            "meaning": f"(đối tượng: {target_object})",
        })

    # Thêm một số từ phổ biến từ caption
    common_words = {
        "image": "hình ảnh",
        "picture": "bức tranh",
        "person": "người",
        "people": "mọi người",
        "cat": "con mèo",
        "dog": "con chó",
        "car": "xe hơi",
        "table": "cái bàn",
        "chair": "cái ghế",
        "book": "quyển sách",
        "phone": "điện thoại",
        "bottle": "cái chai",
        "cup": "cái cốc",
        "laptop": "máy tính xách tay",
        "bird": "con chim",
        "tree": "cái cây",
        "flower": "bông hoa",
        "ball": "quả bóng",
        "bag": "cái túi",
        "room": "căn phòng",
        "house": "ngôi nhà",
        "street": "đường phố",
        "beautiful": "đẹp",
        "small": "nhỏ",
        "large": "lớn",
        "colorful": "nhiều màu sắc",
        "white": "màu trắng",
        "black": "màu đen",
        "red": "màu đỏ",
        "blue": "màu xanh",
        "green": "màu xanh lá",
        "yellow": "màu vàng",
    }

    caption_lower = caption.lower()
    added_words = {target_object.lower()} if target_object else set()

    for word, meaning in common_words.items():
        if word in caption_lower and word not in added_words:
            vocab.append({"word": word, "meaning": meaning})
            added_words.add(word)
            if len(vocab) >= 8:  # Giới hạn 8 từ vựng
                break

    return vocab
