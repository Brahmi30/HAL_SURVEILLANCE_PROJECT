import cv2
import os
from deepface import DeepFace
from ultralytics import YOLO

# 1. FIX THE CORE PATHS (FORCING IT TO LOOK IN THE MAIN DIRECTORY)
current_dir = os.path.dirname(os.path.abspath(__file__))

# If running inside testing_sandbox, go up one level to the main folder
if "testing_sandbox" in current_dir:
    BASE_DIR = os.path.dirname(current_dir) # Go up to HAL_AI_SURVEILLANCE
else:
    BASE_DIR = current_dir

# Set the absolute correct folder locations
AIRCRAFT_DB = os.path.join(BASE_DIR, "database", "aircrafts")
TEST_IMAGE_PATH = os.path.join(current_dir, "test_plane.jpg")

print(f"📂 Debug Path Check: Looking for aircraft database in: {AIRCRAFT_DB}")

# Load the base object tracking model
print("🛰️ Loading YOLOv8 aerospace target tracker...")
detector_model = YOLO("yolov8n.pt")

# 2. FILE VERIFICATION GUARDRAILS
if not os.path.exists(TEST_IMAGE_PATH):
    print(f"❌ Error: Cannot find your test image file 'test_plane.jpg' in: {BASE_DIR}")
    print("Please save an airplane image in that main folder and rename it to 'test_plane.jpg'")
    exit()

if not os.path.exists(AIRCRAFT_DB):
    print(f"❌ Error: Your database folder path is missing! Expected: {AIRCRAFT_DB}")
    exit()

# Read the image file from disk
frame = cv2.imread(TEST_IMAGE_PATH)
display_frame = frame.copy()

print("✈️ Scanning image frame for structural aircraft contours...")
predictions = detector_model(frame, verbose=False, stream=True)

aircraft_found = False

# 3. RUN THE DETECTOR LOOP
for result in predictions:
    boxes = result.boxes
    for box in boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        
        # Class ID 4 is strictly reserved for "Airplanes" inside YOLO
        if class_id == 4 and confidence > 0.40:
            aircraft_found = True
            x1, y1, x2, y2 = box.xyxy[0]
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            
            # Slice the exact coordinates of the airplane out of the frame
            cropped_matrix = frame[y:y+h, x:x+w]
            computed_name = "UNIDENTIFIED AIRCRAFT" # Default tag fallback
            
            if cropped_matrix.size > 0:
                try:
                    # 4. DEPLOY SECONDARY CLASS-MATCHING ENGINE
                    aircraft_results = DeepFace.find(
                        img_path=cropped_matrix, 
                        db_path=AIRCRAFT_DB,
                        model_name="GhostFaceNet", 
                        detector_backend="skip", # Frame is pre-cropped by YOLO
                        distance_metric="cosine", 
                        enforce_detection=False, 
                        silent=True
                    )
                    
                    if aircraft_results and len(aircraft_results) > 0 and not aircraft_results[0].empty:
                        air_df = aircraft_results[0]
                        
                        # Apply strict mathematical distance limits (0.62 threshold)
                        valid_matches = air_df[air_df['distance'] <= 0.62]
                        
                        if not valid_matches.empty:
                            # Pull top 3 matches to vote on the specific plane name
                            top_matches = valid_matches.head(3).copy()
                            top_matches['air_type'] = top_matches['identity'].apply(
                                lambda p: os.path.basename(os.path.dirname(str(p))).upper()
                            )
                            
                            # Read the winning folder name string from your dataset
                            vote_counts = top_matches['air_type'].value_counts()
                            computed_name = vote_counts.index[0]
                            
                except Exception as e:
                    print(f"📡 Fine classification check skipped or no matches found: {e}")
            
            # 5. DRAW VISUAL TELEMETRY BOX
            print(f"🎯 [TARGET DETECTED] Match Confidence: {confidence*100:.1f}% -> Model Name: {computed_name}")
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 3) # Target red border
            cv2.putText(display_frame, f"AIR-LOCK: {computed_name}", (x, y - 12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

if not aircraft_found:
    print("❌ Analysis Complete: YOLO did not see any aircraft patterns in this image.")
    print("Make sure your 'test_plane.jpg' has a clear silhouette of a plane against the sky.")

# 6. RENDER GRAPHICAL TESTING VIEW
cv2.imshow("HAL Tactical Aircraft Test Window", display_frame)
print("\n💡 Click into the popup picture window and press any key on your keyboard to close it cleanly.")
cv2.waitKey(0)
cv2.destroyAllWindows()