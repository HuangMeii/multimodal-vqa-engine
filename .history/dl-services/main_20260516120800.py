from fastapi import FastAPI

app = FastAPI(title="DL Service")

@app.get("/health")
async def health():
    return {"status": "ok", "service": "dl-service"}

@app.get("/api/v1/detect")
async def mock_detect():
    """Giả lập phát hiện đối tượng, sau này sẽ thay bằng Grounding DINO"""
    return {
        "objects": [
            {"label": "person", "bbox": [10, 20, 100, 200]},
            {"label": "cup", "bbox": [50, 60, 120, 150]}
        ]
    }