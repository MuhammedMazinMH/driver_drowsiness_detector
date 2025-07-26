
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
from utils import LEFT_EYE, RIGHT_EYE, play_alert
import pathlib
import os

##### Load the yawn CNN model from file without download #####
MODEL_PATH = pathlib.Path(__file__).resolve().parent.parent / "models" / "yawn_detector.tflite"

if not MODEL_PATH.exists():
    print(f"[ERROR] Model file not found at {MODEL_PATH}!")
    print("Make sure you downloaded a model (e.g., yawn_model_quant_80.tflite), copied it here, and renamed it to yawn_detector.tflite.")
    exit(1)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model expects input shape:", input_details[0]['shape'])


##### MediaPipe Landmark Setup #####
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False)

##### EAR helper #####
def compute_ear(pts):
    # Compute Eye Aspect Ratio as in the project description
    A = np.linalg.norm(pts[1] - pts[5]) + np.linalg.norm(pts[2] - pts[4])
    B = 2.0 * np.linalg.norm(pts[0] - pts[3])
    return A / B if B > 0 else 0

##### Webcam Loop + Detection #####
cap = cv2.VideoCapture(0)
EAR_THRESHOLD = 0.18
CLOSE_FRAMES = 24   # ~2 seconds at 24fps
YAWN_FRAMES = 5    # consecutive frames for yawn

ear_history = []
eye_closed_counter = 0
yawn_counter = 0
last_alarm_time = 0

print("[INFO] Running... Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture image from camera.")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        cv2.imshow("Drowsiness Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue

    h, w, _ = frame.shape
    mesh = results.multi_face_landmarks[0]
    landmarks = np.array([(lm.x * w, lm.y * h) for lm in mesh.landmark])

    # --- EYE -- 
    left_eye_pts = landmarks[LEFT_EYE]
    right_eye_pts = landmarks[RIGHT_EYE]
    ear = (compute_ear(left_eye_pts) + compute_ear(right_eye_pts)) / 2.0
    ear_history.append(ear)
    if len(ear_history) > CLOSE_FRAMES:
        ear_history.pop(0)
    
    if ear < EAR_THRESHOLD:
        eye_closed_counter += 1
    else:
        eye_closed_counter = 0

    # --- MOUTH/YAWN CNN ---
    # Get mouth ROI: you can adjust these indices if required
    mouth_indices = [78, 308, 13, 14, 87, 317, 82, 312]
    mouth = landmarks[mouth_indices]
    x1, y1 = np.min(mouth, axis=0).astype(int)
    x2, y2 = np.max(mouth, axis=0).astype(int)
    # Slightly expand ROI for better generalization
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    w_box = (x2 - x1) * 1.2
    h_box = (y2 - y1) * 1.4
    x1 = max(int(cx - w_box // 2), 0)
    x2 = min(int(cx + w_box // 2), w-1)
    y1 = max(int(cy - h_box // 2), 0)
    y2 = min(int(cy + h_box // 2), h-1)
    mouth_crop = rgb[y1:y2, x1:x2]
    if mouth_crop.shape[0] == 0 or mouth_crop.shape[1] == 0:
        is_yawn = False
    else:
        mouth_crop_gray = cv2.cvtColor(mouth_crop, cv2.COLOR_RGB2GRAY)
        mouth_resized = cv2.resize(mouth_crop_gray, (100, 100))
        mouth_input = np.expand_dims(mouth_resized.astype(np.float32) / 255.0, axis=0)  # shape: (1, 100, 100)
        mouth_input = np.expand_dims(mouth_input, axis=-1)                              # shape: (1, 100, 100, 1)

        if tuple(input_details[0]['shape'][1:3]) == (32, 64):
             mouth_input = mouth_input
        elif tuple(input_details[0]['shape'][1:3]) == (64, 32):
            mouth_input = mouth_input.transpose(0,2,1,3)
        else:
            mouth_input = mouth_input


        interpreter.set_tensor(input_details[0]['index'], mouth_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        # If the model is binary classifier, threshold
        print("Yawn model output:", output) 
        is_yawn = bool(np.argmax(output) == 1 or output[0][0] > 0.3)
        cv2.imshow("Mouth Crop", mouth_resized)

    if is_yawn:
        yawn_counter += 1
    else:
        yawn_counter = 0

    # --- Alert logic: alarm if either event is observed ---
    alarm_trigger = False
    if eye_closed_counter > CLOSE_FRAMES or yawn_counter > YAWN_FRAMES:
        if time.time() - last_alarm_time > 2:  # Avoid spamming the alarm
            play_alert()
            last_alarm_time = time.time()
        alarm_trigger = True

    # -- Display info and HUD
    cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if ear > EAR_THRESHOLD else (0,0,255), 2)
    if is_yawn:
        cv2.putText(frame, "Yawn: YES", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    else:
        cv2.putText(frame, "Yawn: NO ", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    if alarm_trigger:
        cv2.putText(frame, "DROWSINESS ALERT!", (100,200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 4)

    cv2.imshow("Drowsiness Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Program ended.")

