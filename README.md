````markdown
# Attendance System with Facial and Liveness Detection

An intelligent attendance system utilizing facial recognition and liveness detection to ensure secure and accurate attendance logging. The system captures real-time facial data, analyzes texture and motion-based cues to distinguish between live individuals and spoofing attempts, and logs attendance accordingly.

## Features

- **Facial Recognition**: Identifies individuals based on facial features.
- **Liveness Detection**: Prevents spoofing by detecting live presence through blink detection and head movements.
- **Real-time Processing**: Captures and processes video streams in real-time.
- **Attendance Logging**: Records attendance with timestamps in CSV format.

## Requirements

- Python 3.11
- OpenCV
- Dlib
- NumPy
- Pandas

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Ashish0505208/Attendance-System-Facial-detection.git
   cd Attendance-System-Facial-detection
````

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Install Dlib (if not included via `requirements.txt`):

   ```bash
   pip install dlib-19.24.1-cp311-cp311-win_amd64.whl
   ```

4. Ensure that `shape_predictor_68_face_landmarks.dat` is present in the project directory.

## Usage

1. **Register Faces**: Capture and store images of individuals in the `student_images_rgb` directory.

2. **Train Model**: Use `imageconverter.py` to process and encode facial images.

3. **Run Attendance System**: Execute the main script to start the attendance system with liveness detection.

   ```bash
   python face_detection_with_blink.py
   ```

4. **View Attendance**: Attendance records are saved in CSV files such as:

   * `attendance.csv`
   * `attendance_YYYY-MM-DD.csv`

## File Structure

* `face_detection_with_blink.py` – Main script for facial recognition with blink detection
* `blinktest.py` – Script to test blink detection
* `testheadmove.py` – Script to test head movement detection
* `imageconverter.py` – Encodes and stores known faces
* `student_images_rgb/` – Directory of registered user images
* `attendance.csv` – Master attendance log
* `attendance_YYYY-MM-DD.csv` – Daily attendance logs
* `shape_predictor_68_face_landmarks.dat` – Required facial landmark model
* `dlib-19.24.1-cp311-cp311-win_amd64.whl` – Dlib installer (optional)

## Acknowledgements

* [Dlib](http://dlib.net/)
* [OpenCV](https://opencv.org/)

---

For any queries or contributions, contact [Ashish Ravipati](mailto:ashishravipati6@gmail.com).

```reach out to me at linkedin https://www.linkedin.com/in/ashish-ravipati-b4806b32b/
```
