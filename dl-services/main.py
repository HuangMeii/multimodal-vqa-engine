# dl-services/main.py

from fastapi import FastAPI
from routers import detection, scene_graph, questions, caption, florence2, learn

app = FastAPI(title="DL Service", version="0.6.0")

app.include_router(detection.router)
app.include_router(scene_graph.router)
app.include_router(caption.router)
app.include_router(questions.router)
app.include_router(florence2.router)
app.include_router(learn.router)

@app.get("/health")
async def health():
    return {"status": "ok", "service": "dl-service"}
