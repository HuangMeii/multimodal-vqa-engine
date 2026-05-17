# dl-services/tests/test_services.py
import sys
sys.path.insert(0, "/app")

from fastapi.testclient import TestClient
from main import app
import io
from PIL import Image

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_detect():
    # Tạo một ảnh nhỏ màu trắng
    img = Image.new("RGB", (100, 100), color="white")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    
    response = client.post(
        "/api/v1/detect",
        files={"image": ("test.jpg", buf, "image/jpeg")},
        data={"queries": "a person. a cup"}
    )
    assert response.status_code == 200
    assert "objects" in response.json()

def test_scenegraph():
    img = Image.new("RGB", (100, 100), color="white")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    
    response = client.post(
        "/api/v1/scenegraph",
        files={"image": ("test.jpg", buf, "image/jpeg")}
    )
    assert response.status_code == 200
    assert "caption" in response.json()

def test_questions():
    response = client.post(
        "/api/v1/questions",
        data={"caption": "a man holding a cup", "target_object": "cup"}
    )
    assert response.status_code == 200
    assert "question" in response.json()