"""
4K USB webcam object detection.

Fixes the "glittery"/torn frames you get when a 4K USB camera is opened with
OpenCV's default uncompressed YUYV format. At 4K, YUYV exceeds the USB
bandwidth, so frames arrive corrupted. The fix is to:
  1. Request MJPG (compressed) BEFORE setting the resolution.
  2. Capture at full sensor resolution but downscale for inference, since the
     Pi cannot run YOLO at 3840x2160.

Usage:
  venv/bin/python yolo_detect_4k.py --model yolo26n.pt --source usb0
  venv/bin/python yolo_detect_4k.py --model yolo26n.pt --source usb0 \
      --cap-resolution 3840x2160 --infer-resolution 640x480
"""
import os
import sys
import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True,
                    help='Path to YOLO model file (example: "yolo26n.pt")')
parser.add_argument('--source', required=True,
                    help='USB camera index ("usb0"). Use the device number.')
parser.add_argument('--thresh', default=0.5, type=float,
                    help='Minimum confidence threshold (example: "0.4")')
parser.add_argument('--cap-resolution', default='3840x2160',
                    help='Resolution to request from the camera (WxH). Default 4K.')
parser.add_argument('--infer-resolution', default='640x480',
                    help='Resolution to downscale to for inference/display (WxH). '
                         'Keep this small on a Pi for usable FPS.')
parser.add_argument('--record', action='store_true',
                    help='Record annotated output to "demo1.avi" at infer resolution.')
args = parser.parse_args()

model_path = args.model
img_source = args.source
min_thresh = args.thresh

if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or model was not found.')
    sys.exit(0)

if 'usb' not in img_source:
    print('ERROR: This script only supports USB cameras (example: "usb0").')
    sys.exit(0)
usb_idx = int(img_source[3:])

cap_w, cap_h = (int(v) for v in args.cap_resolution.split('x'))
infer_w, infer_h = (int(v) for v in args.infer_resolution.split('x'))

model = YOLO(model_path, task='detect')
labels = model.names

# Open the camera with MJPG so the high-res stream fits over USB bandwidth.
# FOURCC MUST be set before the resolution, otherwise OpenCV keeps YUYV and the
# 4K frames come back torn/glittery.
cap = cv2.VideoCapture(usb_idx)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)        # drop stale frames to reduce lag
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_h)

# Report what the camera actually gave us (it may not honor the exact request).
actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
fourcc_str = ''.join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4))
print(f'Camera opened at {actual_w}x{actual_h}, format={fourcc_str!r}')

if record := args.record:
    recorder = cv2.VideoWriter('demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'),
                               30, (infer_w, infer_h))

bbox_colors = [(164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133),
               (88, 159, 106), (96, 202, 231), (159, 124, 168), (169, 162, 241),
               (98, 118, 150), (172, 176, 184)]

avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200

while True:
    t_start = time.perf_counter()

    ret, frame = cap.read()
    if (frame is None) or (not ret):
        print('Unable to read frames from the camera. Exiting program.')
        break

    # Downscale the 4K frame for inference and display.
    frame = cv2.resize(frame, (infer_w, infer_h))

    results = model(frame, verbose=False)
    detections = results[0].boxes

    object_count = 0
    for i in range(len(detections)):
        xyxy = detections[i].xyxy.cpu().numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        if conf > min_thresh:
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            label = f'{classname}: {int(conf * 100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            object_count += 1

    cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
    cv2.putText(frame, f'Number of objects: {object_count}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
    cv2.imshow('YOLO detection results (4K cam)', frame)
    if record:
        recorder.write(frame)

    key = cv2.waitKey(5)
    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord('s') or key == ord('S'):
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'):
        cv2.imwrite('capture.png', frame)

    t_stop = time.perf_counter()
    frame_rate_calc = float(1 / (t_stop - t_start))
    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(frame_rate_calc)
    avg_frame_rate = np.mean(frame_rate_buffer)

print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
cap.release()
if record:
    recorder.release()
cv2.destroyAllWindows()
