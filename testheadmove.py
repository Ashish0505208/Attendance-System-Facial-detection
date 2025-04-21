import cv2
import dlib
import numpy as np

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Optical flow parameters
lk_params = dict(winSize=(15, 15), maxLevel=2, 
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialize tracking variables
prev_gray = None
prev_points = None

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    status_text = "No face detected"
    color = (0, 255, 255)

    if len(faces) > 0:
        for face in faces:
            landmarks = predictor(gray, face)

            # Select key facial landmarks
            points = np.array([
                (landmarks.part(30).x, landmarks.part(30).y),  # Nose
                (landmarks.part(36).x, landmarks.part(36).y),  # Left eye corner
                (landmarks.part(45).x, landmarks.part(45).y),  # Right eye corner
                (landmarks.part(48).x, landmarks.part(48).y),  # Left mouth
                (landmarks.part(54).x, landmarks.part(54).y)   # Right mouth
            ], dtype=np.float32)

            # Check for previous frame
            if prev_gray is not None and prev_points is not None:
                new_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, points, **lk_params)

                # Compute movement vectors
                motion_vectors = [np.linalg.norm(new_points[i] - prev_points[i]) for i in range(len(points))]

                # Compute standard deviation of motion vectors
                std_dev = np.std(motion_vectors)

                if std_dev > 1.5:  # High variation means real movement
                    status_text = "Real (3D Head Movement)"
                    color = (0, 255, 0)  # Green
                else:  # Low variation means fake movement (like phone shaking)
                    status_text = "Fake (Flat Image Motion)"
                    color = (0, 0, 255)  # Red

            # Update previous frame and points
            prev_gray = gray.copy()
            prev_points = points.reshape(-1, 1, 2)

    # Display status
    cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Head Movement Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
