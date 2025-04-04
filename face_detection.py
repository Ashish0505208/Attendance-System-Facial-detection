import face_recognition
import cv2
import os
import csv
from datetime import datetime

# Folder of known students
KNOWN_FACES_DIR = "student_images"
VIDEO_PATH = "classroom_video.mp4"
OUTPUT_CSV = "attendance.csv"

known_face_encodings = []
known_face_names = []

# Load known faces
for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(os.path.splitext(filename)[0])

# Track attendance
attendance_set = set()

# Open the video
video = cv2.VideoCapture(VIDEO_PATH)

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        best_match_index = None
        if face_distances.size > 0:
            best_match_index = face_distances.argmin()

        if best_match_index is not None and matches[best_match_index]:
            name = known_face_names[best_match_index]
            attendance_set.add(name)
            print(f"[INFO] Marked: {name}")

video.release()

# Save to CSV
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "Status", "Timestamp"])
    for name in known_face_names:
        status = "Present" if name in attendance_set else "Absent"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([name, status, timestamp])

print("[✅] Attendance saved to attendance.csv")
