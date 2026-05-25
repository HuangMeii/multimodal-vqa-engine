# dl-services/main.py

from fastapi import FastAPI
from routers import detection, learn, florence2

app = FastAPI(title="DL Service", version="0.7.0")

app.include_router(detection.router)
app.include_router(learn.router)
app.include_router(florence2.router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "dl-service"}
