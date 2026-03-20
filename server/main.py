from fastapi import FastAPI, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from face_engine import decode_base64_image
import services
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    await websocket.accept() # Mở "đường ống" kết nối
    try:
        while True:
            # Nhận frame dạng base64 từ Client liên tục
            image_base64 = await websocket.receive_text()
            
            # Giải mã ảnh và đưa qua YOLO + InsightFace xử lý
            img_orig = decode_base64_image(image_base64)
            matches = services.process_detection(img_orig)
            
            # Bơm trả kết quả (Tọa độ, Tên, Giới tính) về Client ngay lập tức
            await websocket.send_json({"status": "success", "matches": matches})
            
    except WebSocketDisconnect:
        print("Client đã ngắt kết nối WebSocket")

@app.post("/register_step")
async def register_step(name: str = Form(...), pose_target: str = Form(...), image_base64: str = Form(...)):
    img_orig = decode_base64_image(image_base64)
    return services.process_registration(img_orig, pose_target)

@app.post("/save_user")
async def save_user(name: str = Form(...), gender: str = Form(...), embeddings_json: str = Form(...)):
    return services.save_user_to_db(name, gender, embeddings_json)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)