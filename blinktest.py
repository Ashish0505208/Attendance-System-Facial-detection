import cv2
import dlib
from scipy.spatial import distance
import time

# Eye aspect ratio calculation
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Load detector & predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Landmark indexes for eyes
LEFT_EYE_IDX = list(range(42, 48))
RIGHT_EYE_IDX = list(range(36, 42))

cap = cv2.VideoCapture(0)
blink_count = 0
blink_detected = False
blink_threshold = 0.25
consec_frames = 2
frame_counter = 0

print("[👁️] Blink detection started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        landmarks = [(shape.part(n).x, shape.part(n).y) for n in range(68)]

        left_eye = [landmarks[i] for i in LEFT_EYE_IDX]
        right_eye = [landmarks[i] for i in RIGHT_EYE_IDX]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        if ear < blink_threshold:
            frame_counter += 1
        else:
            if frame_counter >= consec_frames:
                blink_count += 1
                blink_detected = True
            frame_counter = 0

        # Draw eyes
        for (x, y) in left_eye + right_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # Display blink count
    cv2.putText(frame, f"Blinks: {blink_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Liveness Detection (Blink)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or blink_detected:
        break

cap.release()
cv2.destroyAllWindows()

if blink_detected:
    print("[✅] Liveness confirmed (blink detected).")
else:
    print("[⚠️] No blink detected. Possible spoofing attempt.")