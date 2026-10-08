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

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

print(f"载入模型中：{MODEL_NAME}...")
classifier = pipeline("audio-classification", model=MODEL_NAME)
print("模型载入完成！")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("客户端已连接")

    try:
        while True:
            data = await websocket.receive_bytes()
            pcm = np.frombuffer(data, dtype=np.float32)

            if len(pcm) < 100:
                continue

            try:
                results = classifier(pcm, sampling_rate=16000)
                top = results[0]
                confidence = float(top['score'])
                label = top['label']

                if websocket.client_state.name != "CONNECTED":
                    print("客户端已断开，停止发送")
                    break

                await websocket.send_json({
                    "label": "cry" if "cry" in label.lower() else "noise",
                    "confidence": confidence,
                    "cryType": label,
                })

            except Exception as e:
                print(f"推演错误：{e}")
                continue

    except WebSocketDisconnect:
        print("客户端断开连接")

    except Exception as e:
        print(f"WebSocket 错误：{e}")

    finally:
        try:
            await websocket.close()
        except:
            pass


@app.get("/")
def health():
    return {"status": "ok"}
