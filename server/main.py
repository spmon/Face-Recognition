from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from face_engine import decode_base64_image
import services

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/detect")
async def detect_face(image_base64: str = Form(...)):
    img_orig = decode_base64_image(image_base64)
    matches = services.process_detection(img_orig)
    return {"status": "success", "matches": matches}

@app.post("/register_step")
async def register_step(name: str = Form(...), pose_target: str = Form(...), image_base64: str = Form(...)):
    img_orig = decode_base64_image(image_base64)
    return services.process_registration(img_orig, pose_target)

@app.post("/save_user")
async def save_user(name: str = Form(...), gender: str = Form(...), embeddings_json: str = Form(...)):
    return services.save_user_to_db(name, gender, embeddings_json)

if __name__ == "__main__":
    import uvicorn
    # Dùng string "main:app" để reload=True hoạt động chuẩn nhất
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)