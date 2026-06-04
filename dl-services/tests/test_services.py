# dl-services/tests/test_services.py
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from main import app
from PIL import Image


class _FakeCaptioner:
    def generate_caption(self, image):
        return "A person is standing near a table."

    def generate_detailed_caption(self, image):
        return "A person is standing near a table in a bright room."

    def detect_objects(self, image, queries=None):
        objects = [
            {
                "label": "person",
                "confidence": 0.98,
                "bbox": {"xmin": 1, "ymin": 2, "xmax": 30, "ymax": 40},
                "crop_base64": "",
            },
            {
                "label": "table",
                "confidence": 0.87,
                "bbox": {"xmin": 20, "ymin": 30, "xmax": 80, "ymax": 90},
                "crop_base64": "",
            },
        ]
        if queries:
            requested = {part.strip().lower() for part in str(queries).split(",") if part.strip()}
            return [item for item in objects if item["label"].lower() in requested]
        return objects


class _FakeRewriter:
    def rewrite_sentences(self, caption, detailed_caption, target_object, detected_objects=None):
        return ["A person is standing near the table."]

    def translate_batch(self, sentences, target_lang="Vietnamese"):
        return ["Một người đang đứng gần cái bàn."] * len(sentences)


def _install_fakes():
    import routers.detection as detection_router
    import routers.learn as learn_router
    import routers.florence2 as florence2_router

    detection_router.get_captioner = lambda: _FakeCaptioner()
    learn_router.get_captioner = lambda: _FakeCaptioner()
    learn_router.get_rewriter = lambda: _FakeRewriter()
    florence2_router.get_captioner = lambda: _FakeCaptioner()


client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_detect():
    _install_fakes()
    # Tạo một ảnh nhỏ màu trắng
    img = Image.new("RGB", (100, 100), color="white")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    
    response = client.post(
        "/api/v1/detect",
        files={"image": ("test.jpg", buf, "image/jpeg")},
        data={"queries": "a person, a cup"}
    )
    assert response.status_code == 200
    assert "objects" in response.json()

def test_detect_objects():
    _install_fakes()
    img = Image.new("RGB", (100, 100), color="white")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    
    response = client.post(
        "/api/v1/detect-objects",
        files={"image": ("test.jpg", buf, "image/jpeg")}
    )
    assert response.status_code == 200
    assert "objects" in response.json()

def test_learn():
    _install_fakes()
    img = Image.new("RGB", (100, 100), color="white")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    
    response = client.post(
        "/api/v1/learn",
        files={"image": ("test.jpg", buf, "image/jpeg")},
        data={"target_object": "cat"}
    )
    assert response.status_code == 200
    assert "sentences" in response.json()
    assert "objects" in response.json()
