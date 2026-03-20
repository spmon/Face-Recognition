import os
from dotenv import load_dotenv

# Tải các biến từ file .env vào môi trường hệ thống
load_dotenv()

DB_HOST = os.getenv('DB_HOST')
DB_PORT = int(os.getenv('DB_PORT'))
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASS = os.getenv('DB_PASS')

YOLO_MODEL_PATH = os.getenv('YOLO_MODEL_PATH')
THRESHOLD = float(os.getenv('THRESHOLD', 0.75))