# dl-services/routers/learn.py
# API học tiếng Anh: 1 ảnh → Florence-2 caption + OD → TinyLlama rewrite → TinyLlama dịch

from fastapi import APIRouter, UploadFile, File, Form
from PIL import Image
import io
import json
import re
import torch

router = APIRouter()

# Lazy loading
_captioner = None
_rewriter = None


def get_captioner():
    global _captioner
    if _captioner is None:
        from services.t1_vision.florence2_captioner import Florence2Captioner
        _captioner = Florence2Captioner()
    return _captioner


def get_rewriter():
    global _rewriter
    if _rewriter is None:
        from services.t2_reasoning.tinyllama_translator import TinyLlamaTranslator
        _rewriter = TinyLlamaTranslator()
    return _rewriter


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

    print(f"[DEBUG] Florence-2 OD raw output: {repr(od_result)}")

    # Parse OD result từ Florence-2
    # Florence-2 OD format thực tế (với skip_special_tokens=False):
    #   </s><s><s><s>box<loc_0><loc_0><loc_999><loc_998>cat<loc_191><loc_226><loc_753><loc_725></s>
    # Format: label<loc_x1><loc_y1><loc_x2><loc_y2>label<loc_x1><loc_y1><loc_x2><loc_y2>...
    # Trong đó <loc_N> là tọa độ 0-999 scale
    # Mỗi label đi kèm với 4 <loc> ngay sau nó
    objects = []
    if od_result:
        # Dùng tokenize thủ công: duyệt từng token (label hoặc <loc_N>)
        # Pattern: ([a-zA-Z_]+|<loc_\d+>)
        tokens = re.findall(r'[a-zA-Z_]+|<loc_\d+>', od_result)
        current_label = None
        current_locs = []
        for token in tokens:
            if token.startswith('<loc_'):
                loc_val = int(token.replace('<loc_', '').replace('>', ''))
                current_locs.append(loc_val)
                if len(current_locs) == 4 and current_label:
                    # Đã đủ 4 tọa độ, thêm object
                    label = current_label
                    if label.lower() not in ('od', '/od', 'obj', '/obj', 'bbox', '/bbox', ''):
                        objects.append({
                            "label": label,
                            "confidence": 0.9,
                            "bbox": {
                                "xmin": current_locs[0],
                                "ymin": current_locs[1],
                                "xmax": current_locs[2],
                                "ymax": current_locs[3],
                            },
                        })
                    current_label = None
                    current_locs = []
            else:
                # Đây là label
                current_label = token
                # Nếu còn locs dang dở từ object trước, reset
                current_locs = []

        # Fallback: nếu không có bbox, tìm label đơn thuần
        if not objects:
            words = re.findall(r'[a-zA-Z_][a-zA-Z\s]*', od_result)
            for w in words:
                w = w.strip()
                if w and w.lower() not in ('od', 'obj', 'bbox', ''):
                    objects.append({
                        "label": w,
                        "confidence": 0.9,
                        "bbox": {"xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0},
                    })
                    break

    print(f"[DEBUG] Parsed {len(objects)} objects: {objects}")

    # Giải phóng VRAM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"objects": objects}


def _generate_sentences_from_caption(caption: str, detailed_caption: str, target_object: str) -> list[str]:
    """Sinh 1 câu tiếng Anh từ caption thực tế của ảnh bằng TinyLlama rewrite."""
    rewriter = get_rewriter()
    try:
        sentences = rewriter.rewrite_sentences(caption, detailed_caption, target_object)
        return sentences
    except Exception as e:
        print(f"[WARN] TinyLlama rewrite failed: {e}, falling back to raw captions")
        # Fallback: dùng caption gốc
        s = detailed_caption if detailed_caption else caption
        if not s or len(s) <= 5:
            s = f"I can see {target_object if target_object else 'something'} in the image."
        return [s]


@router.post("/api/v1/learn")
async def learn_from_image(
    image: UploadFile = File(...),
    target_object: str = Form(""),
):
    """
    Học tiếng Anh từ ảnh + object đã chọn.
    - Florence-2: sinh caption + object detection
    - TinyLlama: rewrite caption thành 1 câu tiếng Anh
    - KHÔNG dịch sang Việt ở đây (tách riêng endpoint /api/v1/translate)
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

    # === 2. TinyLlama: rewrite caption thành 1 câu tiếng Anh ===
    sentences_en = _generate_sentences_from_caption(
        caption, detailed_caption, target_object
    )

    # === 3. Xây dựng vocabulary từ caption ===
    vocabulary = _extract_vocabulary(caption, target_object)

    # === 4. Format kết quả (chỉ tiếng Anh, chưa dịch) ===
    sentences = [{"en": s, "vi": ""} for s in sentences_en]

    return {
        "sentences": sentences,
        "vocabulary": vocabulary,
        "caption": caption,
        "detailed_caption": detailed_caption,
    }


@router.post("/api/v1/translate")
async def translate_text(
    texts: str = Form(...),
):
    """
    Dịch câu tiếng Anh sang tiếng Việt bằng TinyLlama.
    Nhận vào JSON array các câu tiếng Anh, trả về JSON array các câu đã dịch.

    Args:
        texts: JSON string của list[string] các câu tiếng Anh
               Ví dụ: '["I can see a cat.", "The cat is sitting on the table."]'

    Returns:
        {"translations": ["Tôi có thể thấy một con mèo.", "Con mèo đang ngồi trên bàn."]}
    """
    # Parse input
    try:
        sentences_en = json.loads(texts)
        if not isinstance(sentences_en, list):
            sentences_en = [sentences_en]
    except (json.JSONDecodeError, TypeError):
        sentences_en = [texts]

    # === TinyLlama: dịch câu Anh → Việt ===
    rewriter = get_rewriter()
    sentences_vi = []
    try:
        sentences_vi = rewriter.translate_batch(sentences_en)
    except Exception as e:
        print(f"[WARN] TinyLlama translation failed: {e}")
        sentences_vi = sentences_en[:]  # fallback: giữ nguyên tiếng Anh

    return {"translations": sentences_vi}


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
