# dl-services/services/t1_vision/florence2_captioner.py
from pathlib import Path

from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from PIL import Image


def _resolve_local_model_path(model_path: str) -> str:
    path = Path(model_path)
    if path.is_dir() and (path / "preprocessor_config.json").exists() and (path / "config.json").exists():
        return str(path)

    if path.is_dir():
        for candidate in path.glob("**/preprocessor_config.json"):
            candidate_dir = candidate.parent
            if (candidate_dir / "config.json").exists():
                return str(candidate_dir)

    return model_path


class Florence2Captioner:
    def __init__(self, model_path="/app/models/florence2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_path = _resolve_local_model_path(model_path)
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
        )
        self.model.eval()

    def _prepare_inputs(self, image: Image.Image, task_prompt: str):
        """Chuẩn bị inputs, ép kiểu pixel_values về cùng dtype với model."""
        inputs = self.processor(text=task_prompt, images=image, return_tensors="pt").to(self.device)
        # Ép kiểu pixel_values về cùng dtype với model để tránh lỗi
        # "Input type (float) and bias type (c10::Half) should be the same"
        model_dtype = next(self.model.parameters()).dtype
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)
        return inputs

    def generate_caption(self, image: Image.Image) -> str:
        task_prompt = "<CAPTION>"
        inputs = self._prepare_inputs(image, task_prompt)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=50,
                num_beams=1,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        caption = caption.replace(task_prompt, "").strip()
        return caption

    def generate_detailed_caption(self, image: Image.Image) -> str:
        task_prompt = "<DETAILED_CAPTION>"
        inputs = self._prepare_inputs(image, task_prompt)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=200,
                num_beams=1,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        caption = caption.replace(task_prompt, "").strip()
        return caption

    def generate_od(self, image: Image.Image) -> str:
        task_prompt = "<OD>"
        inputs = self._prepare_inputs(image, task_prompt)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=200,
                num_beams=1,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        result = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        result = result.replace(task_prompt, "").strip()
        return result
