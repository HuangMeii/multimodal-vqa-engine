# dl-services/services/t2_reasoning/tinyllama_translator.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re


class TinyLlamaTranslator:
    """
    Dùng TinyLlama để dịch câu tiếng Anh sang tiếng Việt.
    TinyLlama là model 1.1B parameters, chạy được trên CPU.
    """

    def __init__(self, model_path="/app/models/tinyllama", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
        )
        self.model.eval()

        # TinyLlama không có pad_token mặc định
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def translate(self, text: str, target_lang: str = "Vietnamese") -> str:
        """
        Dịch câu tiếng Anh sang tiếng Việt.
        
        Args:
            text: Câu tiếng Anh cần dịch
            target_lang: Ngôn ngữ đích (mặc định: Vietnamese)
        
        Returns:
            Câu đã dịch
        """
        prompt = f"""Translate the following English sentence to {target_lang}. Only output the translation, nothing else.

English: {text}
{target_lang}:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract translation after the prompt
        if prompt in response:
            translation = response[len(prompt):].strip()
        else:
            # Try to find the last line as translation
            lines = response.strip().split("\n")
            translation = lines[-1].strip() if lines else response.strip()

        # Clean up common issues
        translation = translation.split("\n")[0].strip()
        translation = re.sub(r'^["\']|["\']$', '', translation)

        return translation if translation else text

    def translate_batch(self, sentences: list[str], target_lang: str = "Vietnamese") -> list[str]:
        """
        Dịch nhiều câu cùng lúc.
        """
        results = []
        for sentence in sentences:
            translated = self.translate(sentence, target_lang)
            results.append(translated)
        return results
