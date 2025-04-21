#imports
import cv2
import dlib
import os
import face_recognition
import csv
from datetime import datetime
from scipy.spatial import distance

# Eye aspect ratio
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Load known faces
known_face_encodings = []
known_face_names = []

for filename in os.listdir("student_images_rgb"):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join("student_images_rgb", filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])

# Init Dlib detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye landmark indices
LEFT_EYE_IDX = list(range(42, 48))
RIGHT_EYE_IDX = list(range(36, 42))

# Parameters
blink_threshold = 0.25 #change this to make the blinking more effective 0.25- is eye closed 0.25+is open eye
consec_frames = 2 #change this to reduce the number of blinks to mark the attendance
tolerance = 0.45 #tolerance for the image comparision in the video to the photos in the dataset

# Trackers
blink_counters = {}
frame_counters = {}
blink_confirmed = {}
attendance_marked = set()

# CSV Setup
date_str = datetime.now().strftime("%Y-%m-%d")
csv_filename = f"attendance_{date_str}.csv"
if not os.path.exists(csv_filename):
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Timestamp'])

# Start video
cap = cv2.VideoCapture(0)
print("[👁️] Multi-face Blink Detection with Attendance started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for encoding, loc in zip(face_encodings, face_locations):
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, encoding)

        if len(face_distances) > 0:
            best_match_index = face_distances.argmin()
            if face_distances[best_match_index] < tolerance:
                name = known_face_names[best_match_index]

        top, right, bottom, left = [v * 4 for v in loc]

        face_rect = dlib.rectangle(left, top, right, bottom)
        shape = predictor(gray, face_rect)
        landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        left_eye = [landmarks[i] for i in LEFT_EYE_IDX]
        right_eye = [landmarks[i] for i in RIGHT_EYE_IDX]
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

        if name not in blink_counters:
            blink_counters[name] = 0
            frame_counters[name] = 0
            blink_confirmed[name] = False

        if ear < blink_threshold:
            frame_counters[name] += 1
        else:
            if frame_counters[name] >= consec_frames:
                blink_counters[name] += 1
                blink_confirmed[name] = True

                if name != "Unknown" and name not in attendance_marked:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    with open(csv_filename, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([name, timestamp])
                    attendance_marked.add(name)
                    print(f"[✅] Attendance marked for {name} at {timestamp}")

            frame_counters[name] = 0

        # Draw box and name
        color = (0, 255, 0) if blink_confirmed[name] else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        label = f"{name} | ✅" if name in attendance_marked else f"{name}"
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for (x, y) in left_eye + right_eye:
            cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)

    cv2.imshow("Multi-Person Liveness + Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
