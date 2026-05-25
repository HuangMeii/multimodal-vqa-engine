# dl-services/services/t2_reasoning/qwen.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class QwenLLM:
    def __init__(self, model_path="/app/models/qwen", device=None):
        # Force CPU để tiết kiệm VRAM cho Florence-2 và Grounding DINO
        self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        self.model.eval()

    def generate(self, prompt, max_new_tokens=64):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Bỏ phần prompt nếu nó lặp lại
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        return response
