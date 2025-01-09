from collections import deque
import numpy as np
import cv2
import winsound
from datetime import datetime
import os

class Config:
    def __init__(self):
        # Ensure model and labels file exist before loading
        if not os.path.exists("model/action_recognition_kinetics.txt"):
            raise FileNotFoundError("Action recognition labels file not found.")
        if not os.path.exists("model/resnet-34_kinetics.onnx"):
            raise FileNotFoundError("Model file not found.")
        
        self.CLASS_LABELS = open("model/action_recognition_kinetics.txt").read().strip().split("\n")
        self.MODEL_PATH = 'model/resnet-34_kinetics.onnx'
        self.FRAME_HISTORY = 16
        self.FRAME_DIM = 112
        self.ALERT_ACTIONS = ['smoking']

config = Config()
frame_buffer = deque(maxlen=config.FRAME_HISTORY)

# Load the pre-trained activity recognition model
print("[INFO] Loading activity recognition model...")
model = cv2.dnn.readNet(model=config.MODEL_PATH)

# Open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Could not open camera.")
    exit()

# Open a file to log the detected activities and timestamps
with open("activity_log.txt", "a") as log_file:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame from camera.")
            break

        frame_resized = cv2.resize(frame, (config.FRAME_DIM, config.FRAME_DIM))
        frame_buffer.append(frame_resized)

        if len(frame_buffer) < config.FRAME_HISTORY:
            continue

        blob = cv2.dnn.blobFromImages(list(frame_buffer), 1.0,
                                      (config.FRAME_DIM, config.FRAME_DIM),
                                      (114.7748, 107.7354, 99.4750),
                                      swapRB=True, crop=True)

        blob = np.transpose(blob, (1, 0, 2, 3))
        blob = np.expand_dims(blob, axis=0)

        model.setInput(blob)
        predictions = model.forward()

        # Check if model output shape matches expectations
        if predictions.shape[1] != len(config.CLASS_LABELS):
            print("[ERROR] The output shape of the model does not match the number of labels.")
            break

        action_label = config.CLASS_LABELS[np.argmax(predictions)]

        # Get current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if action_label in config.ALERT_ACTIONS:
            print(f"[ALERT] Suspicious activity detected: {action_label}")
            print(f"Time: {current_time}")
            winsound.Beep(1000, 500)  # if detect human smoking, its display sound

            # Log the detected activity and timestamp into the file
            log_file.write(f"Alert: {action_label}, Time: {current_time}\n")

        # Display the time on the frame
        cv2.putText(frame, f"Time: {current_time}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Show frame
        cv2.imshow("Detect_Human_Activity(Smoking)", frame)

        # Break if 'a' is pressed
        if cv2.waitKey(1) & 0xFF == ord("a"):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
