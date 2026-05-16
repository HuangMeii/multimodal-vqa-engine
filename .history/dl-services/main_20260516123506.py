# dl-services/main.py

from fastapi import FastAPI
from routers import detection

app = FastAPI(title="DL Service", version="0.2.0")

app.include_router(detection.router)

@app.get("/health")
async def health():
    return {"status": "ok", "service": "dl-service"}