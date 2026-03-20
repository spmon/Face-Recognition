import cv2
import numpy as np
import base64
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import config

print("🚀 Đang nạp model YOLO11 Tracking...")
yolo_model = YOLO(config.YOLO_MODEL_PATH)

print("🚀 Đang nạp model InsightFace...")
face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider']) 
face_app.prepare(ctx_id=0, det_size=(640, 640))

def decode_base64_image(image_base64: str):
    encoded_data = image_base64.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def crop_face(img, x1, y1, x2, y2, padding=10):
    x1_crop = max(0, x1 - padding)
    y1_crop = max(0, y1 - padding)
    x2_crop = min(img.shape[1], x2 + padding)
    y2_crop = min(img.shape[0], y2 + padding)
    return img[y1_crop:y2_crop, x1_crop:x2_crop]