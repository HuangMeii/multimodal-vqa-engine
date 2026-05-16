# dl-services/services/t2_reasoning/qwen.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class QwenLLM:
    def __init__(self, model_path="/app/models/qwen", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, torch_dtype=torch.float16, device_map=self.device
        )

    def generate(self, prompt, max_length=128):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)