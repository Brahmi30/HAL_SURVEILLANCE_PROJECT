import cv2
import os
import time
from ultralytics import YOLO
from deepface import DeepFace

# Load YOLO model
model = YOLO("yolov8n.pt")

# Face database path
DB_PATH = "database/faces"

# Open webcam
cap = cv2.VideoCapture(0)

# Haar Cascade Face Detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Identity persistence variables
last_recognized_name = "UNKNOWN"
last_recognition_time = 0

print("HAL AI Surveillance System Started")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Resize frame for speed
    frame = cv2.resize(frame, (640, 480))

    # YOLO Detection
    results = model(frame)

    # Loop through detections
    for result in results:

        boxes = result.boxes

        for box in boxes:

            cls = int(box.cls[0])

            # Class 0 = person
            if cls == 0:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Crop detected person
                person_roi = frame[y1:y2, x1:x2]

                # Convert to grayscale
                gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)

                # Detect face inside ROI
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(80, 80)
                )

                recognized_name = "UNKNOWN"

                # Loop through faces
                for (fx, fy, fw, fh) in faces:

                    # Crop face ROI
                    roi = person_roi[fy:fy+fh, fx:fx+fw]

                    try:

                        # Face recognition
                        face_results = DeepFace.find(
                            img_path=roi,
                            db_path=DB_PATH,
                            enforce_detection=False,
                            silent=True,
                            model_name="Facenet",
                            detector_backend="opencv"
                        )

                        # Match found
                        if len(face_results) > 0 and len(face_results[0]) > 0:

                            best_match = face_results[0].iloc[0]

                            distance = best_match['distance']

                            print("Distance:", distance)

                            # Threshold filtering
                            if distance < 0.55:

                                identity = best_match['identity']

                                recognized_name = identity.split(os.sep)[-2]

                                # Save recognized identity
                                last_recognized_name = recognized_name
                                last_recognition_time = time.time()

                    except Exception as e:
                        print("Recognition Error:", e)

                    # Draw face box
                    cv2.rectangle(
                        person_roi,
                        (fx, fy),
                        (fx + fw, fy + fh),
                        (255, 0, 0),
                        2
                    )

                # Identity Persistence Logic
                current_time = time.time()

                # Different colors for known and unknown
                if recognized_name == "UNKNOWN":
                     box_color = (0, 0, 255)  # Red
                     cv2.putText(frame,"WARNING: UNKNOWN TARGET DETECTED",(20, 40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 255),2    )
                else:
                     box_color = (0, 255, 0)  # Green

# Draw person box
                cv2.rectangle(frame,
    (x1, y1),
    (x2, y2),
    box_color,
    2
)
                # Display name
                cv2.putText(
                    frame,
                    recognized_name,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    box_color,
                    2
                )

    cv2.imshow("HAL AI Surveillance", frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()