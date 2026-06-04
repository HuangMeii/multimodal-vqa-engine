"""Lightweight smoke test for the vision wrapper.

This test uses injected stubs so it does not require downloading model weights.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from services.t1_vision.florence2_captioner import Florence2Captioner


class _FakeDetector:
	names = {0: "person"}

	def predict(self, source, conf, device, verbose, max_det):
		class _Box:
			def __init__(self):
				import torch

				self.cls = torch.tensor([0])
				self.conf = torch.tensor([0.99])
				self.xyxy = torch.tensor([[1.0, 2.0, 50.0, 60.0]])

		class _Result:
			names = {0: "person"}
			boxes = [_Box()]

		return [_Result()]


class _FakeProcessor:
	def __call__(self, images=None, text=None, return_tensors=None):
		import torch

		return {
			"pixel_values": torch.zeros((1, 3, 224, 224)),
			"input_ids": torch.zeros((1, 3), dtype=torch.long),
		}

	def batch_decode(self, generated_ids, skip_special_tokens=True):
		return ["A person is near a table."]


class _FakeCaptionModel:
	def to(self, device):
		return self

	def eval(self):
		return self

	def generate(self, **kwargs):
		import torch

		return torch.tensor([[1, 2, 3]])


def test_vision_wrapper_smoke():
	captioner = Florence2Captioner(
		detector=_FakeDetector(),
		processor=_FakeProcessor(),
		caption_model=_FakeCaptionModel(),
	)

	img = Image.new("RGB", (224, 224), color="red")

	assert captioner.generate_caption(img) == "A person is near a table."
	assert captioner.generate_detailed_caption(img) == "A person is near a table."
	objects = captioner.generate_od(img)
	assert objects and objects[0]["label"] == "person"
