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

    def rewrite_sentences(self, caption: str, detailed_caption: str, target_object: str = "") -> list[str]:
        """
        Dùng TinyLlama để rewrite caption thành 1 câu tiếng Anh,
        phù hợp với ngữ cảnh thực tế của ảnh.

        Args:
            caption: Caption ngắn từ Florence-2
            detailed_caption: Caption chi tiết từ Florence-2
            target_object: Object người dùng chọn để học (có thể rỗng)

        Returns:
            list[str]: 1 câu tiếng Anh
        """
        object_hint = f' focusing on the object "{target_object}"' if target_object else ""

        prompt = f"""Given an image caption, rewrite it into 1 natural English sentence for learning English{object_hint}.

Caption: {detailed_caption if detailed_caption else caption}

Output exactly 1 sentence:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the generated part after the prompt
        if prompt in response:
            generated = response[len(prompt):].strip()
        else:
            generated = response.strip()

        # Parse 1 sentence
        sentence = generated.split("\n")[0].strip()
        # Remove numbering like "1." if present
        sentence = re.sub(r'^\d+\.\s*', '', sentence)

        # Fallback if empty
        if not sentence or len(sentence) <= 5:
            if target_object:
                sentence = f"I can see {target_object} in the image."
            else:
                sentence = f"I can see something in the image."

        return [sentence]
