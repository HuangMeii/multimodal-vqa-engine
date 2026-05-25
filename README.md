# VQA System — Visual Question Answering

Hệ thống trả lời câu hỏi trực quan (VQA) gồm 3 tầng: **Flutter App → DL Service → Models AI**.

---

## 🏗️ Kiến trúc tổng quan

```
┌──────────────────────────────────────────────────────────────────┐
│  Flutter App (english_learning)                                   │
│  Port: 8000 (mặc định: http://10.0.2.2:8000)                    │
│  Gọi API: /health, /api/v1/caption, /api/v1/questions,...        │
└────────────────────┬─────────────────────────────────────────────┘
                     │ HTTP (từ Android emulator / thiết bị thật)
                     ▼
┌──────────────────────────────────────────────────────────────────┐
│  DL Service (FastAPI - Python) — port 8000                       │
│  Chứa các model AI: Grounding DINO, Florence-2, Qwen            │
│  Router endpoints:                                               │
│    • POST /api/v1/detect           → GroundingDINODetector       │
│    • POST /api/v1/caption          → Florence2Captioner          │
│    • POST /api/v1/scenegraph       → Florence2Captioner          │
│    • POST /api/v1/questions        → QwenLLM                     │
│    • POST /api/v1/florence2/caption                               │
│    • POST /api/v1/florence2/detailed-caption                      │
│    • POST /api/v1/florence2/od                                    │
└────────────────────┬─────────────────────────────────────────────┘
                     │ HTTP (nội bộ qua Docker network)
                     ▼
┌──────────────────────────────────────────────────────────────────┐
│  Server Backend (Spring Boot - Java) — port 8080                 │
│  Endpoints: /health, /call-dl                                     │
│  Vai trò: proxy/trung gian (hiện tại Flutter gọi thẳng DL)      │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📂 Cấu trúc thư mục

```
vqa-system/
├── dl-services/                    # Python FastAPI — AI Service
│   ├── Dockerfile
│   ├── main.py                     # App FastAPI, mount routers
│   ├── requirements.txt
│   ├── routers/
│   │   ├── detection.py            # POST /api/v1/detect
│   │   ├── caption.py              # POST /api/v1/caption
│   │   ├── scene_graph.py          # POST /api/v1/scenegraph
│   │   ├── questions.py            # POST /api/v1/questions
│   │   └── florence2.py            # POST /api/v1/florence2/*
│   ├── services/
│   │   ├── t1_vision/              # Tầng thị giác
│   │   │   ├── object_detector.py  # Grounding DINO
│   │   │   ├── florence2_captioner.py  # Florence-2
│   │   │   ├── visual_encoder.py   # BLIP-2 (dự phòng)
│   │   │   └── crop_utils.py       # Crop ảnh theo bbox
│   │   └── t2_reasoning/           # Tầng suy luận
│   │       └── qwen.py             # Qwen LLM
│   └── tests/
│       └── test_services.py
│
├── server-backend/                 # Java Spring Boot — Backend
│   ├── Dockerfile
│   ├── pom.xml
│   └── src/
│       └── main/java/com/vqa/server/
│           ├── VqaServerApplication.java
│           └── controller/
│               └── HealthController.java  # /health, /call-dl
│
├── infra/
│   └── docker-compose.yml          # Docker Compose (2 services)
│
└── README.md
```

---

## 🧠 Chi tiết Model AI

### 1. Grounding DINO — Object Detection
| Mục | Chi tiết |
|------|---------|
| File | `services/t1_vision/object_detector.py` |
| Model path | `/app/models/grounding-dino` |
| Load | `local_files_only=True` (từ volume mount) |
| Input | ảnh + danh sách object queries |
| Output | `{ "objects": [{"label","confidence","bbox","crop_base64"}] }` |
| Endpoint | `POST /api/v1/detect` |

### 2. Florence-2 — Captioning & Scene Graph
| Mục | Chi tiết |
|------|---------|
| File | `services/t1_vision/florence2_captioner.py` |
| Model path | `/app/models/florence2` |
| Tasks | `<CAPTION>`, `<DETAILED_CAPTION>`, `<OD>` |
| Endpoints | `POST /api/v1/caption`, `/scenegraph`, `/florence2/*` |

### 3. Qwen — Question Generation
| Mục | Chi tiết |
|------|---------|
| File | `services/t2_reasoning/qwen.py` |
| Model path | `/app/models/qwen` |
| Input | `{"caption": "...", "target_object": "..."}` |
| Output | `{"question": "..."}` |
| Endpoint | `POST /api/v1/questions` |

---

## 📱 Flutter App → Luồng xử lý ảnh

```
HomeScreen.initState()
  → VqaCubit.checkHealth()              // GET /health
  → User chọn ảnh (camera / gallery)
  → VqaCubit.analyze()
    → VqaRepositoryImpl.analyze()
      → VqaRemoteDatasource.caption()   // POST /api/v1/caption (multipart image)
      → VqaRemoteDatasource.questions() // POST /api/v1/questions (JSON)
    → Cubit emit: VqaStatus.loaded
    → HomeScreen hiển thị caption + danh sách câu hỏi
```

**Flutter App config** (`api_endpoints.dart`):
```dart
defaultValue: 'http://10.0.2.2:8000'   // Android emulator → host localhost
// USB reverse: http://127.0.0.1:8000  (cần chạy: adb reverse tcp:8000 tcp:8000)
```

> **Lưu ý:** Flutter App hiện tại gọi **trực tiếp DL Service** (port 8000), không đi qua Server Backend (port 8080). Server Backend chỉ có endpoint `/health` và `/call-dl`.

---

## 🐳 Docker Infrastructure

### Services

| Service | Image Base | Port | GPU | Mô tả |
|---------|-----------|------|-----|-------|
| `dl-service` | `python:3.10-slim` | `8000:8000` | ✅ NVIDIA | FastAPI + Models |
| `server-backend` | `eclipse-temurin:21-jre` | `8080:8080` | ❌ | Spring Boot proxy |

### Volume Mounts

```yaml
vqa-models/grounding-dino → /app/models/grounding-dino  (local_files_only)
vqa-models/qwen           → /app/models/qwen            (local_files_only)
vqa-models/florence2      → /app/models/florence2       (local_files_only)
```

### Network

Cả 2 service nằm trong `vqa-net` (bridge). Server Backend gọi DL Service qua `http://dl-service:8000`.

---

## 🚀 Hướng dẫn chạy

### Docker (production)

```bash
cd vqa-system/infra
docker-compose up --build -d
```

Kiểm tra:
```bash
curl http://localhost:8000/health          # DL Service
curl http://localhost:8080/health          # Server Backend
curl http://localhost:8080/call-dl         # Server → DL (test proxy)
```

### DL Service local (development)

```bash
cd vqa-system/dl-services
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Flutter App

```bash
cd english_learning
flutter run
```

Nếu chạy trên Android emulator, đảm bảo DL Service đang chạy ở port 8000.  
Nếu chạy trên thiết bị thật qua USB, chạy:
```bash
adb reverse tcp:8000 tcp:8000
```

---

## 🔌 Danh sách API Endpoints

### DL Service (port 8000)

| Method | Path | Mô tả |
|--------|------|-------|
| GET | `/health` | Health check |
| POST | `/api/v1/detect` | Object detection (Grounding DINO) |
| POST | `/api/v1/caption` | Caption ảnh (Florence-2) |
| POST | `/api/v1/scenegraph` | Scene graph (Florence-2 detailed caption) |
| POST | `/api/v1/questions` | Sinh câu hỏi (Qwen) |
| POST | `/api/v1/florence2/caption` | Florence-2 caption |
| POST | `/api/v1/florence2/detailed-caption` | Florence-2 detailed caption |
| POST | `/api/v1/florence2/od` | Florence-2 object detection |

### Server Backend (port 8080)

| Method | Path | Mô tả |
|--------|------|-------|
| GET | `/health` | Health check |
| GET | `/call-dl` | Proxy → DL Service health |

---

## ❌ Troubleshooting

| Vấn đề | Nguyên nhân | Fix |
|--------|------------|-----|
| App không kết nối được server | DL Service chưa chạy | `docker-compose up -d` |
| Connection timeout | Sai IP/cổng | Kiểm tra URL trong app, `adb reverse` nếu cần |
| 422 Unprocessable Entity | Sai format request | Kiểm tra body/multipart đúng format |
| GPU không hoạt động | Thiếu NVIDIA driver | Cài [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) |
