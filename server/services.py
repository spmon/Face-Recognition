import json
import numpy as np
import config
from database import get_db_connection
from face_engine import yolo_model, face_app, crop_face

active_tracks = {}

def process_detection(img_orig):
    global active_tracks
    results = yolo_model.track(source=img_orig, persist=True, tracker="bytetrack.yaml", verbose=False)
    
    if len(results[0].boxes) == 0:
        active_tracks.clear()
        return []

    # Quét rác
    current_ids = [int(i) for i in results[0].boxes.id.cpu().numpy()] if results[0].boxes.id is not None else []
    for old_id in list(active_tracks.keys()):
        if old_id not in current_ids:
            del active_tracks[old_id]

    matches = []
    conn = get_db_connection()
    cur = conn.cursor() if conn else None

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cls = int(box.cls[0].item())
        gender_yolo = "Female" if cls == 0 else "Male"
        track_id = int(box.id[0].item()) if box.id is not None else -1

        name_identified, final_gender = "Unknown", gender_yolo

        if track_id != -1 and track_id in active_tracks and active_tracks[track_id]["name"] != "Unknown":
            name_identified = active_tracks[track_id]["name"]
            final_gender = active_tracks[track_id]["gender"]
        else:
            img_crop = crop_face(img_orig, x1, y1, x2, y2, padding=40)
            if img_crop.shape[0] > 40 and img_crop.shape[1] > 40:
                faces_cropped = face_app.get(img_crop)
                
                if len(faces_cropped) > 0 and cur:
                    embedding = faces_cropped[0].embedding
                    # CHUẨN HÓA VECTOR CAMERA
                    embedding = embedding / np.linalg.norm(embedding)
                    
                    cur.execute("""
                        SELECT name, gender, (embedding <-> %s::vector) AS distance 
                        FROM face_embeddings 
                        WHERE gender = %s 
                        ORDER BY distance LIMIT 1
                    """, (embedding.tolist(), gender_yolo))
                    
                    row = cur.fetchone()
                    if row:
                        db_name, db_gender, distance = row
                        print(f"🔍 [ID {track_id}] Giống '{db_name}' | Khoảng cách: {distance:.4f} | Ngưỡng: {config.THRESHOLD}")
                        if distance < config.THRESHOLD:
                            name_identified, final_gender = db_name, db_gender

            if track_id != -1:
                active_tracks[track_id] = {"name": name_identified, "gender": final_gender}

        matches.append({
            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), 
            "name": name_identified, "gender": final_gender
        })

    if cur: cur.close()
    if conn: conn.close()
    return matches

def process_registration(img_orig, pose_target):
    yolo_res = yolo_model.predict(source=img_orig, conf=0.5, verbose=False)
    if len(yolo_res[0].boxes) == 0:
        return {"status": "error", "message": "YOLO không thấy mặt!"}
    
    box = yolo_res[0].boxes[0]
    gender_label = "Female" if int(box.cls[0].item()) == 0 else "Male"
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
    
    img_crop = crop_face(img_orig, x1, y1, x2, y2, padding=20)
    faces = face_app.get(img_crop)
    
    if len(faces) == 0: return {"status": "error", "message": "InsightFace lỗi!"}
    
    face = faces[0]
    pitch, yaw, roll = face.pose

    if pose_target == "straight" and (abs(yaw) > 15 or abs(pitch) > 15): return {"status": "error", "message": "Nhìn thẳng!"}
    elif pose_target == "left" and yaw < 15: return {"status": "error", "message": "Quay TRÁI!"}
    elif pose_target == "right" and yaw > -15: return {"status": "error", "message": "Quay PHẢI!"}

    return {"status": "success", "gender": gender_label, "embedding": face.embedding.tolist()}

def save_user_to_db(name, gender, embeddings_json):
    try:
        avg_embedding = np.mean(json.loads(embeddings_json), axis=0)
        # CHUẨN HÓA VECTOR TRƯỚC KHI LƯU
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        
        conn = get_db_connection()
        if not conn: return {"status": "error", "message": "Lỗi DB!"}
        cur = conn.cursor()
        cur.execute('INSERT INTO face_embeddings (name, gender, embedding) VALUES (%s, %s, %s)', (name, gender, avg_embedding.tolist()))
        conn.commit()
        cur.close()
        conn.close()
        return {"status": "success", "message": f"Đã lưu {name} ({gender})!"}
    except Exception as e:
        return {"status": "error", "message": str(e)}