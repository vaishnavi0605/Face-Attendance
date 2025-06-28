# Automatic Attendance System Using Face Detection

## Overview
This project is an automatic attendance system that uses YOLO for face detection and DeepFace for face recognition. It features a Tkinter-based UI for face registration and uses local text files to log attendance for each recognized student. The system is implemented in Python and is optimized for speed and ease of use.

## Features
- Real-time face detection using YOLOv8
- Face recognition using DeepFace (Facenet model)
- Fast, local attendance logging (one text file per student)
- Face registration from camera
- Option to erase all attendance records for a student

## Directory Structure
```
Attendance System/
├── yolo_face_detect.py           # Main script
├── yolov8n.pt                    # YOLOv8 model weights
├── faces_db/                     # Registered face images (one per student)
├── attendance_logs/              # Attendance logs (one .txt per student)
```

## Requirements
- Python 3.8+
- opencv-python
- ultralytics
- deepface
- scikit-learn

## Setup
1. Clone or download this repository.
2. Create and activate a virtual environment:
   ```zsh
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```zsh
   pip install opencv-python ultralytics deepface scikit-learn
   ```
4. Ensure `yolov8n.pt` is present in the project directory.
5. Make sure `faces_db/` and `attendance_logs/` folders exist (they will be created automatically if not).

## Usage
Run the main script:
```zsh
python yolo_face_detect.py
```
You will be prompted to select a mode:
- **1. Face Recognition (Attendance):** Start attendance logging via camera.
- **2. Register New Face:** Register a new face for recognition.
- **3. Erase Attendance Records for a Student:** Delete all attendance logs for a student.

## Attendance Logs
- Each recognized student has a `.txt` file in `attendance_logs/`.
- Each line is a timestamp of a recognition event.

## Notes
- For best results, register faces in good lighting and with only one face visible.
- Restart the script after registering a new face to update the recognition database.

---
Developed by Ashutosh Rajput
