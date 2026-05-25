from fastapi import APIRouter
from pydantic import BaseModel
import torch
from services.t2_reasoning.qwen import QwenLLM

router = APIRouter()

# Lazy loading: Qwen chỉ load khi cần
_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = QwenLLM()
    return _llm


class QuestionsRequest(BaseModel):
    caption: str
    target_object: str | None = None


@router.post("/api/v1/questions")
async def generate_questions(req: QuestionsRequest):
    """
    Sinh câu hỏi dựa trên caption và đối tượng mục tiêu.
    """
    if req.target_object:
        prompt = f"""Based on this image description: "{req.caption}"
Create one English question about "{req.target_object}" in the image.
Question:"""
    else:
        prompt = f"""Based on this image description: "{req.caption}"
Create one English question about the image.
Question:"""

    llm = get_llm()
    question = llm.generate(prompt, max_new_tokens=48)

    # Giải phóng VRAM sau Qwen (nếu có)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"question": question}
