# 🚀 Real-time Face Recognition System (YOLO11 + InsightFace)

Hệ thống nhận diện khuôn mặt thời gian thực hiệu năng cao, được thiết kế theo kiến trúc **Client-Server** tách biệt. Dự án kết hợp sức mạnh tracking của **YOLO11** và khả năng trích xuất đặc trưng của **InsightFace**, kết hợp với cơ sở dữ liệu vector để đạt tốc độ xử lý **30+ FPS**.

---

## 🌟 Tính năng & Kỹ thuật cốt lõi

* **High Performance Tracking (ByteTrack):** Tối ưu hóa FPS bằng cách dùng YOLO11 bám đuổi khuôn mặt (Tracking). Chỉ gọi AI trích xuất Embedding cho các ID mới, giảm thiểu tối đa nút thắt cổ chai.
* **Deep Insight & L2 Normalization:** Sử dụng model `buffalo_l` để tạo Vector 512-D. Tất cả vector đều được chuẩn hóa L2 (`np.linalg.norm`) từ lúc lưu vào DB đến lúc inference, giúp hệ thống duy trì độ chính xác cao ngay cả khi người dùng đứng xa camera (Ngưỡng `THRESHOLD = 0.75`).
* **Coarse-to-Fine Search (Tối ưu tìm kiếm):** Sử dụng chiến lược hai lớp để tăng tốc truy vấn. 
  * **Lớp 1 (Lọc thô):** Dùng nhãn giới tính từ YOLO11 để thu hẹp 50% không gian tìm kiếm. 
  * **Lớp 2 (Lọc tinh):** Áp dụng toán tử Euclidean (`<->`) của `pgvector` trên tập dữ liệu đã thu hẹp để tìm khuôn mặt khớp nhất.
* **Kiến trúc Decoupled:** Tách biệt hoàn toàn Frontend (HTML/Canvas) và Backend (FastAPI), dễ dàng scale-up và deploy.

---

## 🗄️ Database Schema & Truy vấn

Hệ thống sử dụng **PostgreSQL** kết hợp với extension **pgvector**.

**Bảng `face_embeddings`:**

| Cột | Kiểu dữ liệu | Mô tả |
| :--- | :--- | :--- |
| **id** | `bigserial` (PK) | Khóa chính. |
| **name** | `varchar(100)` | Họ tên nhân viên. |
| **gender** | `varchar(10)` | Giới tính (Male/Female) để Metadata Filtering. |
| **embedding** | `vector(512)` | Vector đặc trưng 512-D đã chuẩn hóa L2. |

**Truy vấn cốt lõi (Tích hợp Coarse-to-Fine):**

```sql
SELECT name, gender, (embedding <-> %s::vector) AS distance 
FROM face_embeddings 
WHERE gender = %s  -- Lọc thô bằng nhãn YOLO
ORDER BY distance LIMIT 1;