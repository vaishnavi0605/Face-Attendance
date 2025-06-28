import cv2
from ultralytics import YOLO
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from pymongo import MongoClient
from datetime import datetime

# Load YOLOv8 face detection model (using a pre-trained model)
# You may need to download a YOLOv8 face model or use a generic YOLOv8 model
model = YOLO('yolov8n.pt')  # Replace with a face-specific model if available

# Precompute embeddings for faces_db using a lighter model (Facenet)
embeddings_db = []
embeddings_names = []
for db_img in os.listdir("faces_db"):
    if db_img.lower().endswith((".jpg", ".jpeg", ".png")):
        db_img_path = os.path.join("faces_db", db_img)
        try:
            embedding = DeepFace.represent(img_path=db_img_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
            embeddings_db.append(embedding)
            embeddings_names.append(os.path.splitext(db_img)[0])
        except Exception as e:
            pass

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['attendance_system']
attendance_collection = db['attendance']

def log_attendance(name):
    if name != "Unknown":
        log_file = os.path.join("attendance_logs", f"{name}.txt")
        os.makedirs("attendance_logs", exist_ok=True)
        with open(log_file, "a") as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")

def erase_attendance_records(name):
    log_file = os.path.join("attendance_logs", f"{name}.txt")
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"Deleted attendance log for {name}.")
    else:
        print(f"No attendance log found for {name}.")

def detect_faces(image_path):
    image = cv2.imread(image_path)
    results = model(image)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('YOLO Face Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_faces_from_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    frame_count = 0
    recognition_interval = 5  # Run recognition every 5th frame
    last_names = []
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting...")
            break
        results = model(frame)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        names = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            face_img = frame[y1:y2, x1:x2]
            name = "Unknown"
            if frame_count % recognition_interval == 0:
                try:
                    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    face_embedding = DeepFace.represent(face_img_rgb, model_name="Facenet", enforce_detection=False)[0]["embedding"]
                    best_score = 0.3  # similarity threshold
                    best_name = "Unknown"
                    for db_emb, db_name in zip(embeddings_db, embeddings_names):
                        sim = cosine_similarity([face_embedding], [db_emb])[0][0]
                        if sim > best_score:
                            best_score = sim
                            best_name = db_name
                    name = best_name
                    log_attendance(name)
                except Exception as e:
                    pass
                names.append(name)
            else:
                # Use last recognized names if available
                if len(last_names) == len(boxes):
                    name = last_names[len(names)]
                names.append(name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        if frame_count % recognition_interval == 0:
            last_names = names
        frame_count += 1
        cv2.imshow('YOLO Face Detection - Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def register_face_from_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    print("Press 'c' to capture and register your face, or 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting...")
            break
        results = model(frame)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.imshow('Register Face - Press c to capture', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if len(boxes) == 1:
                x1, y1, x2, y2 = map(int, boxes[0][:4])
                face_img = frame[y1:y2, x1:x2]
                name = input("Enter name for this face: ").strip()
                if name:
                    save_path = os.path.join("faces_db", f"{name}.jpg")
                    cv2.imwrite(save_path, face_img)
                    print(f"Face saved as {save_path}. Please restart the script to update the database.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                else:
                    print("Name cannot be empty.")
            else:
                print("Please ensure only one face is visible for registration.")
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Select mode:")
    print("1. Face Recognition (Attendance)")
    print("2. Register New Face")
    print("3. Erase Attendance Records for a Student")
    mode = input("Enter 1, 2 or 3: ").strip()
    if mode == '1':
        detect_faces_from_camera()
    elif mode == '2':
        register_face_from_camera()
    elif mode == '3':
        name = input("Enter the name of the student to erase records: ").strip()
        erase_attendance_records(name)
    else:
        print("Invalid selection.")
