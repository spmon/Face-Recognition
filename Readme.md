# 🚀 Real-time Face Recognition System (YOLO11 + InsightFace)

Hệ thống nhận diện khuôn mặt thời gian thực hiệu năng cao, được thiết kế theo kiến trúc **Client-Server** tách biệt. Dự án kết hợp sức mạnh tracking của **YOLO11** và khả năng trích xuất đặc trưng của **InsightFace**, kết hợp với cơ sở dữ liệu vector để đạt tốc độ xử lý **30+ FPS**.

---

## 🌟 Tính năng & Kỹ thuật cốt lõi

* **High Performance Tracking (ByteTrack):** Tối ưu hóa FPS bằng cách dùng YOLO11 bám đuổi khuôn mặt (Tracking). Chỉ gọi AI trích xuất Embedding cho các ID mới, giảm thiểu tối đa nút thắt cổ chai (bottleneck).
* **Deep Insight & L2 Normalization:** Sử dụng model `buffalo_l` để tạo Vector 512-D. Tất cả vector đều được chuẩn hóa L2 (`np.linalg.norm`) từ lúc lưu vào DB đến lúc inference, giúp hệ thống duy trì độ chính xác cao ngay cả khi người dùng đứng xa camera (Ngưỡng `THRESHOLD = 0.75`).
* **Coarse-to-Fine Search (Tối ưu tìm kiếm):** * **Lọc thô (Metadata Filtering):** Sử dụng nhãn giới tính (Gender) từ YOLO11 để thu hẹp 50% không gian tìm kiếm ngay từ câu lệnh SQL.
  * **Lọc tinh (Vector Similarity):** Áp dụng toán tử Euclidean (`<->`) của `pgvector` trên tập dữ liệu đã thu hẹp để tìm ra khuôn mặt khớp nhất.
* **Kiến trúc Decoupled:** Tách biệt hoàn toàn Frontend (HTML/Canvas) và Backend (FastAPI), dễ dàng scale-up và deploy lên các môi trường cloud.

---

## 📂 Cấu trúc thư mục

```text
Face-Recognition/
├── server/             # Backend (Python, FastAPI, AI Models)
│   ├── main.py         # API Gateway
│   ├── services.py     # Logic xử lý AI & Database
│   ├── face_engine.py  # Khởi tạo YOLO & InsightFace
│   ├── database.py     # Kết nối PostgreSQL
│   ├── config.py       # Quản lý cấu hình & Hyperparameters
│   ├── .env            # Biến môi trường (Bảo mật - Không push lên Git)
│   └── requirements.txt
└── client/             # Frontend (HTML5, JavaScript, Canvas)
    ├── index.html      # Trang Monitor Real-time
    └── register.html   # Trang Đăng ký người mới
🗄️ Database Schema & Truy vấn
Hệ thống sử dụng PostgreSQL kết hợp với extension pgvector.

Bảng face_embeddings:
| Cột | Kiểu dữ liệu | Mô tả |
| :--- | :--- | :--- |
| id | bigserial (PK) | Khóa chính. |
| name | varchar(100) | Họ tên người dùng. |
| gender | varchar(10) | Giới tính (Male/Female) để Metadata Filtering. |
| embedding | vector(512) | Vector đặc trưng 512-D đã chuẩn hóa L2. |
Truy vấn cốt lõi (Tích hợp Coarse-to-Fine):

SQL
SELECT name, gender, (embedding <-> %s::vector) AS distance 
FROM face_embeddings 
WHERE gender = %s  -- Lọc thô bằng nhãn YOLO
ORDER BY distance LIMIT 1;
🛠 Hướng dẫn cài đặt
1. Khởi chạy Database (Docker)
Cần có PostgreSQL hỗ trợ pgvector. Khởi chạy nhanh bằng Docker:

Bash
docker run --name face-db -e POSTGRES_PASSWORD=your_password -p 5433:5432 -d ankane/pgvector
(Lưu ý: Truy cập DB và tạo bảng/extension như mô tả trong code database.py)

2. Cài đặt Backend
Mở Terminal, di chuyển vào thư mục server/:

Bash
cd server
Cài đặt PyTorch (CUDA): Để hệ thống chạy max tốc độ, hãy cài đặt PyTorch phiên bản hỗ trợ GPU tương ứng với máy của bạn tại pytorch.org. Ví dụ (CUDA 12.1):

Bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
Cài đặt các thư viện lõi:

Bash
pip install -r requirements.txt
Cấu hình file .env (từ .env.example) và khởi chạy server:

Bash
python main.py
3. Khởi chạy Frontend
Server đã chạy ở localhost:8000. Bạn chỉ cần mở trực tiếp file client/index.html hoặc client/register.html trên trình duyệt (Chrome/Edge/Brave) để sử dụng hệ thống.