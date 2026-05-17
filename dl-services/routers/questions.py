from fastapi import APIRouter, Form
from services.t2_reasoning.qwen import QwenLLM

router = APIRouter()

llm = QwenLLM()   # load Qwen một lần

@router.post("/api/v1/questions")
async def generate_questions(
    caption: str = Form(...),
    target_object: str = Form(...)
):
    """
    Sinh câu hỏi dựa trên caption và đối tượng mục tiêu.
    """
    prompt = f"""Dựa vào mô tả ảnh sau: "{caption}"
Hãy tạo một câu hỏi về đối tượng "{target_object}" trong ảnh.
Câu hỏi:"""
    
    question = llm.generate(prompt)
    return {"question": question}