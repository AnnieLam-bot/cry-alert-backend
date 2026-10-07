from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from transformers import pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "Wiam/baby-cry-classification-finetuned-babycry-v4"

print(f"載入模型中：{MODEL_NAME}...")
classifier = pipeline("audio-classification", model=MODEL_NAME)
print("模型載入完成！")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            pcm = np.frombuffer(data, dtype=np.float32)
            if len(pcm) < 100:
                continue
            results = classifier(pcm, sampling_rate=16000)
            top = results[0]
            confidence = float(top['score'])
            label = top['label']
            await websocket.send_json({
                "label": "cry" if confidence > 0.5 else "noise",
                "confidence": confidence,
                "cryType": label if confidence > 0.5 else None,
            })
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Error: {e}")

@app.get("/")
def health():
    return {"status": "ok"}
