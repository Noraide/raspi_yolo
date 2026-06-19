import numpy as np
import cv2
import time
from ai_edge_litert import interpreter as tflite
from picamera2 import Picamera2
from libcamera import controls

# === Setup model ===
MODEL_PATH = "detect.lite"
LABEL_PATH = "labels.txt"
CONFIDENCE_THRESHOLD = 0.5

# Load label
with open(LABEL_PATH, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === Setup camera ===
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
# Continuous autofocus is only on Pi Camera Modules; USB/UVC cameras don't advertise it.
try:
    picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
except RuntimeError:
    pass
picam2.start()
time.sleep(2)

window_name = "Detection"
cv2.namedWindow(window_name)

prev_time = time.time()

# === Main loop ===
while True:
    frame = picam2.capture_array()

    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time

    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    
    image_resized = cv2.resize(frame, (96, 96))

    input_data = image_resized.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)      # (1, 96, 96, 3)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])  # shape: (1, 12, 12, 3)
    output_data = output_data[0]  # remove batch dim → (12,12,3)

    display_frame = cv2.resize(frame, (640, 480))
    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

    grid_h, grid_w, num_classes = output_data.shape
    cell_h = display_frame.shape[0] // grid_h
    cell_w = display_frame.shape[1] // grid_w

    label_scores = {}
    label_positions = {}

    for y in range(grid_h):
        for x in range(grid_w):
            scores = output_data[y, x]
            class_index = np.argmax(scores)
            confidence = scores[class_index]

            if class_index != 0 and confidence > CONFIDENCE_THRESHOLD:
                label = labels[class_index - 1]

                if (label not in label_scores) or (confidence > label_scores[label]):
                    label_scores[label] = confidence
                    label_positions[label] = (x, y)

    for label, (x, y) in label_positions.items():
        confidence = label_scores[label]
        cx = x * cell_w + cell_w // 2
        cy = y * cell_h + cell_h // 2

        # Dot hijau
        cv2.circle(display_frame, (cx, cy), 10, (0, 255, 0), -1)

        # Label (putih)
        cv2.putText(display_frame, label, (cx - 10, cy - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Confidence (merah)
        cv2.putText(display_frame, f"{confidence:.2f}", (cx - 10, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.setWindowTitle(window_name, f"Detection - {fps:.2f} FPS")
    cv2.imshow(window_name, display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()
