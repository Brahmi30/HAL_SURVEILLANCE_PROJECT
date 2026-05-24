from flask import Flask, render_template, Response, jsonify, send_file, request
import cv2
import os
import threading
import time
from datetime import datetime
import csv
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
# pip install face_recognition
# pip install cmake
# pip install dlib
app = Flask(__name__)

# ==========================================
# 1. INITIALIZE GLOBAL PATHS & AI OBJECTS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACES_DB = os.path.join(BASE_DIR, "database", "faces")
AIRCRAFT_DB = os.path.join(BASE_DIR, "database", "aircraft")
LOGS_FILE = os.path.join(BASE_DIR, "surveillance_logs.csv")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

# Guarantee target tracking directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FACES_DB, exist_ok=True)
os.makedirs(AIRCRAFT_DB, exist_ok=True)

# Pre-seed persistent CSV logger
if not os.path.exists(LOGS_FILE):
    with open(LOGS_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Target Name", "Classification Type"])

# GLOBAL STREAM MEMORY DATA STRUCTURES
global_frame = None          
camera_active = False
camera_pointer = None
detected_targets = []  
live_logs_cache = ["System initialized successfully..."] 

# Thread synchronization lock
frame_lock = threading.Lock()

# Thread-safe dictionary to maintain dedicated local YOLO instances per runtime thread
models = {}

def get_yolo_model():
    """Dynamically binds or retrieves a dedicated YOLO model for the current execution thread."""
    thread_id = threading.get_ident()
    if thread_id not in models:
        models[thread_id] = YOLO("yolov8n.pt")
    return models[thread_id]


# ==========================================
# 2. CORE UTILITY & BACKGROUND LOGGING LOGIC
# ==========================================
def append_to_surveillance_log(name, target_type):
    global live_logs_cache
    try:
        timestamp_str = datetime.now().strftime("%H:%M:%S")
        timestamp_full = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(LOGS_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp_full, name, target_type.upper()])
            
        if name in ["UNKNOWN VISITOR", "SUSPECT / UNKNOWN", "UNIDENTIFIED AIRCRAFT"]:
            log_entry = f"[{timestamp_str}] <span style='color: #ff4444; font-weight: bold;'>ALERT: {name} DETECTED</span>"
        else:
            log_entry = f"[{timestamp_str}] <span style='color: #00c851; font-weight: bold;'>{name} ({target_type.upper()}) IDENTIFIED</span>"
            
        with frame_lock:
            if log_entry not in live_logs_cache:
                live_logs_cache.append(log_entry)
                if len(live_logs_cache) > 8:
                    live_logs_cache.pop(1)
                    
    except Exception as e:
        print(f"[LOGGING ERROR] Could not write to CSV: {e}")


# ==========================================
# 3. BACKGROUND MULTI-THREADED WORKERS
# ==========================================
def camera_grabber_worker():
    global global_frame, camera_active, camera_pointer
    while camera_active and camera_pointer is not None:
        success, frame = camera_pointer.read()
        if not success:
            time.sleep(0.01)
            continue
        resized_frame = cv2.resize(frame, (750, 500))
        with frame_lock:
            global_frame = resized_frame
        time.sleep(0.015)

def ai_processing_worker():
    global global_frame, camera_active, detected_targets
    print("[AI ENGINE] Precision Target Recognition Core Active.")
    
    local_detector = get_yolo_model()
    frame_counter = 0
    cached_targets = []
    last_logged_targets = {} 

    while camera_active:
        local_frame = None
        with frame_lock:
            if global_frame is not None:
                local_frame = global_frame.copy()
                
        if local_frame is None:
            time.sleep(0.05)
            continue
            
        frame_counter += 1
        # Throttling inference window down to every 3rd frame to optimize system overhead
        if frame_counter % 3 != 0 and cached_targets:
            with frame_lock:
                detected_targets = cached_targets
            time.sleep(0.03)  # Balanced sleep prevents CPU starvation loop on skipped iterations
            continue
            
        temp_detected_targets = []
        current_time = time.time()
        
        try:
            predictions = local_detector(local_frame, verbose=False, stream=True)
            
            for result in predictions:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Target Filter: Person (0) or Airplane (4) only
                    if confidence > 0.30 and (class_id == 0 or class_id == 4):
                        x1, y1, x2, y2 = box.xyxy[0]
                        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                        
                        pad = int(h * 0.12)  
                        y_start = max(0, y - pad)
                        y_end = min(local_frame.shape[0], y + h + pad)
                        x_start = max(0, x - pad)
                        x_end = min(local_frame.shape[1], x + w + pad)
                        
                        cropped_matrix = local_frame[y_start:y_end, x_start:x_end]
                        if cropped_matrix.size == 0:
                            continue
                            
                        # --- BRANCH A: AEROSPACE ---
                        if class_id == 4:
                            computed_name = "UNIDENTIFIED AIRCRAFT"
                            try:
                                aircraft_results = DeepFace.find(
                                    img_path=cropped_matrix, db_path=AIRCRAFT_DB,
                                    model_name="Facenet", detector_backend="skip",
                                    distance_metric="cosine", enforce_detection=False, silent=True
                                )
                                if aircraft_results and len(aircraft_results) > 0 and not aircraft_results[0].empty:
                                    best_match = aircraft_results[0].iloc[0]
                                    if float(best_match['distance']) < 0.55: 
                                        computed_name = os.path.basename(os.path.dirname(str(best_match['identity']))).upper()
                            except Exception: 
                                pass
                            temp_detected_targets.append({"box": [x, y, w, h], "name": computed_name, "type": "air"})

                        # --- BRANCH B: PERSONNEL IDENTIFICATION (VOTING K-NN) ---
                        elif class_id == 0:
                            computed_name = "UNKNOWN VISITOR" 
                            try:
                                face_results = DeepFace.find(
                                    img_path=cropped_matrix, db_path=FACES_DB,
                                    model_name="GhostFaceNet", detector_backend="opencv", 
                                    distance_metric="cosine", enforce_detection=False, silent=True
                                )
                                
                                if face_results and len(face_results) > 0 and not face_results[0].empty:
                                    df = face_results[0]
                                    valid_matches = df[df['distance'] <= 0.40] 
                                    
                                    if not valid_matches.empty:
                                        top_k_matches = valid_matches.head(5).copy()
                                        top_k_matches['identity_name'] = top_k_matches['identity'].apply(
                                            lambda p: os.path.basename(os.path.dirname(str(p))).upper()
                                        )
                                        
                                        vote_counts = top_k_matches['identity_name'].value_counts()
                                        dominant_label = vote_counts.index[0]
                                        dominant_votes = vote_counts.iloc[0]
                                        total_votes_cast = len(top_k_matches)
                                        
                                        if (dominant_votes / total_votes_cast) >= 0.60:
                                            computed_name = dominant_label
                                            
                            except Exception as e:
                                print(f"[VOTING CORE ERROR] Matrix processing exception: {e}")
                                
                            temp_detected_targets.append({"box": [x, y, w, h], "name": computed_name, "type": "ground"})
                            
                            # Log to persistent CSV if target hasn't been logged within the last 12 seconds
                            if computed_name not in last_logged_targets or (current_time - last_logged_targets[computed_name]) > 12:
                                append_to_surveillance_log(computed_name, "Personnel Profile")
                                last_logged_targets[computed_name] = current_time
                                
            if not temp_detected_targets:
                temp_detected_targets = [{"box": None, "name": "PERIMETER SECURE - SCANNING...", "type": "status"}]
                
            cached_targets = temp_detected_targets
            with frame_lock:
                detected_targets = temp_detected_targets
                
        except Exception as error_context:
            print(f"[AI PIPELINE CORE MALFUNCTION] {error_context}")
            
        time.sleep(0.01)

def generate_frames():
    global global_frame, detected_targets, camera_active
    while True:
        if not camera_active:
            blank_frame = np.zeros((500, 750, 3), dtype=np.uint8)
            cv2.putText(blank_frame, "TACTICAL STANDBY", (240, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
            ret, buffer = cv2.imencode('.jpg', blank_frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
            continue
            
        local_display_frame = None
        local_targets = []
        with frame_lock:
            if global_frame is not None:
                local_display_frame = global_frame.copy()
                local_targets = list(detected_targets)
                
        if local_display_frame is None:
            time.sleep(0.01)
            continue
            
        for target in local_targets:
            box = target.get("box")
            name = target.get("name", "SCANNING...")
            target_type = target.get("type", "status")
            
            if box is not None:
                x, y, w, h = box
                box_color = (0, 0, 255) if target_type == "air" else (0, 255, 0)
                label_prefix = "AIR-LOCK" if target_type == "air" else "ID-LOCK"
                cv2.rectangle(local_display_frame, (x, y), (x + w, y + h), box_color, 2)
                cv2.putText(local_display_frame, f"{label_prefix}: {name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)
            else:
                alert_color = (0, 255, 0) if "SECURE" in name else (0, 165, 255)
                cv2.putText(local_display_frame, f"STATUS: {name}", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, alert_color, 2)
                        
        ret, buffer = cv2.imencode('.jpg', local_display_frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.02)


# ==========================================
# 4. FLASK ROUTING & WEB INTERFACES
# ==========================================
@app.route('/')
def home():
    return render_template('index.html', processed_img_path=None, detections=None)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/system_status')
def get_system_status():
    global camera_active, detected_targets, live_logs_cache
    
    with frame_lock:
        targets_list = list(detected_targets)
        current_logs = list(live_logs_cache)
    
    actual_targets = [t for t in targets_list if t.get("box") is not None]
    target_count = len(actual_targets)
    has_unknown = any(t.get("name") in ["UNKNOWN VISITOR", "SUSPECT / UNKNOWN", "UNIDENTIFIED AIRCRAFT"] for t in actual_targets)
    
    if not camera_active:
        camera_state = "INACTIVE"
        recognition_state = "OFFLINE"
        sys_status = "STANDBY"
    else:
        camera_state = "ACTIVE"
        recognition_state = "ON AIR"
        sys_status = "ALERT (TARGET DETECTED)" if has_unknown else "SECURE" if target_count > 0 else "SCANNING..."

    return jsonify({
        "camera": camera_state,
        "recognition": recognition_state,
        "targets_detected": target_count,
        "system_status": sys_status,
        "logs": current_logs 
    })

@app.route('/start')
def start_surveillance():
    global camera_pointer, camera_active, detected_targets, live_logs_cache
    if not camera_active:
        camera_pointer = cv2.VideoCapture(0)
        camera_active = True
        detected_targets = [{"box": None, "name": "SYNCHRONIZING CORE CHANNELS...", "type": "status"}]
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        live_logs_cache.append(f"[{timestamp}] TACTICAL MONITORING CHANNELS ENGAGED...")
        
        t1 = threading.Thread(target=camera_grabber_worker, daemon=True)
        t1.start()
        t2 = threading.Thread(target=ai_processing_worker, daemon=True)
        t2.start()
    return "Surveillance Core Active"

@app.route('/stop')
def stop_surveillance():
    global camera_pointer, camera_active, global_frame, detected_targets, live_logs_cache
    camera_active = False
    time.sleep(0.15)
    if camera_pointer is not None:
        camera_pointer.release()
        camera_pointer = None
    with frame_lock:
        global_frame = None
        detected_targets = [{"box": None, "name": "OFFLINE", "type": "status"}]
        timestamp = datetime.now().strftime("%H:%M:%S")
        live_logs_cache.append(f"[{timestamp}] PERIMETER DISENGAGED - SYSTEM OFFLINE.")
    return "Surveillance Core Terminated"

@app.route('/upload_verify', methods=['POST'])
def upload_verify():

    if 'query_img' not in request.files:
        return "No file uploaded", 400

    file = request.files['query_img']

    if file.filename == '':
        return "No file selected", 400

    try:

        input_path = os.path.join(UPLOAD_FOLDER, "temp_input.jpg")
        output_path = os.path.join(UPLOAD_FOLDER, "temp_output.jpg")

        file.save(input_path)

        frame = cv2.imread(input_path)

        if frame is None:
            return "Invalid image", 400

        # Resize for faster inference
        frame = cv2.resize(frame, (750, 500))

        # YOLO detection
        detector_model = get_yolo_model()

        predictions = detector_model(frame, verbose=False)

        detected_names = []

        for result in predictions:

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # Only person detection
                if confidence < 0.45 or class_id != 0:
                    continue

                x1, y1, x2, y2 = box.xyxy[0]

                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)

                # Face padding
                pad = 20

                x_start = max(0, x - pad)
                y_start = max(0, y - pad)

                x_end = min(frame.shape[1], x + w + pad)
                y_end = min(frame.shape[0], y + h + pad)

                cropped_face = frame[y_start:y_end, x_start:x_end]

                if cropped_face.size == 0:
                    continue

                computed_name = "UNKNOWN VISITOR"

                try:

                    # ==========================================
                    # TEST FACE EMBEDDING
                    # ==========================================
                    test_embedding = DeepFace.represent(
                        img_path=cropped_face,
                        model_name="ArcFace",
                        detector_backend="opencv",
                        enforce_detection=False
                    )[0]["embedding"]

                    best_match_name = "UNKNOWN VISITOR"
                    best_similarity = -1

                    # ==========================================
                    # DATABASE COMPARISON
                    # ==========================================
                    for person_name in os.listdir(FACES_DB):

                        person_folder = os.path.join(FACES_DB, person_name)

                        if not os.path.isdir(person_folder):
                            continue

                        for img_name in os.listdir(person_folder):

                            img_path = os.path.join(person_folder, img_name)

                            try:

                                db_embedding = DeepFace.represent(
                                    img_path=img_path,
                                    model_name="ArcFace",
                                    detector_backend="opencv",
                                    enforce_detection=False
                                )[0]["embedding"]

                                # Cosine similarity
                                similarity = np.dot(
                                    test_embedding,
                                    db_embedding
                                ) / (
                                    np.linalg.norm(test_embedding)
                                    * np.linalg.norm(db_embedding)
                                )

                                # Keep highest similarity
                                if similarity > best_similarity:

                                    best_similarity = similarity
                                    best_match_name = person_name.upper()

                            except Exception:
                                pass

                    # ==========================================
                    # FINAL DECISION
                    # ==========================================
                    if best_similarity > 0.35:
                        computed_name = best_match_name

                except Exception as e:
                    print(f"[FACE RECOGNITION ERROR] {e}")

                detected_names.append(computed_name)

                # Draw results
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    computed_name,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        # No detections
        if len(detected_names) == 0:
            detected_names.append("NO FACE DETECTED")

        # Save result image
        cv2.imwrite(output_path, frame)

        # Logs
        timestamp = datetime.now().strftime("%H:%M:%S")

        live_logs_cache.append(
            f"[{timestamp}] MANUAL ANALYSIS EXECUTED: {', '.join(detected_names)}"
        )

        return render_template(
            'index.html',
            processed_img_path='/static/uploads/temp_output.jpg?v=' + str(time.time()),
            detections=detected_names
        )

    except Exception as e:

        print(f"[UPLOAD VERIFY ERROR] {e}")

        return "Processing Failed", 500
@app.route('/view_raw_logs')
def view_raw_logs():
    if os.path.exists(LOGS_FILE):
        with open(LOGS_FILE, 'r') as f:
            content = f.read()
        return Response(content, mimetype='text/plain')
    return "No surveillance records generated yet."

@app.route('/download_report')
def download_report():
    if os.path.exists(LOGS_FILE):
        return send_file(
            LOGS_FILE, mimetype='text/csv', as_attachment=True,
            download_name=f"HAL_Surveillance_Report_{datetime.now().strftime('%Y%m%d')}.csv"
        )
    return "No report generation file found.", 404

if __name__ == "__main__":
    # COLD-START EMBEDDING COMPUTATION STRATEGY
    print("[SYSTEM STARTUP] Pre-computing image representations for target databases...")
    try:
        # Runs a baseline zero-matrix evaluation to safely instantiate/validate structural pickle formats
        if os.path.exists(FACES_DB) and os.listdir(FACES_DB):
            DeepFace.represent(img_path=np.zeros((100, 100, 3), dtype=np.uint8), model_name="GhostFaceNet", enforce_detection=False)
        if os.path.exists(AIRCRAFT_DB) and os.listdir(AIRCRAFT_DB):
            DeepFace.represent(img_path=np.zeros((100, 100, 3), dtype=np.uint8), model_name="Facenet", enforce_detection=False)
    except Exception as startup_err:
        print(f"[STARTUP WARNING] Initialization optimizer bypassed: {startup_err}")

    app.run(debug=True, use_reloader=False, port=5000, threaded=True)