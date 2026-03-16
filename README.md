# Yolo on Raspberry PI
How to run yolo detection model on raspberry Pi

## Step 1: Update Your System

Update and upgrade your system packages:

```bash
sudo apt update && sudo apt upgrade -y
```
## Step 2: Make yolo directory and virtual enviroment

Create a yolo file and enter the project directory:

```bash
mkdir YOLO && cd YOLO
```
Create a virtual environment named 'venv'
```bash
python3 -m venv --system-site-packages venv
```
Activate the virtual environment
```bash
source venv/bin/activate
```

## Step 3: Install the Ultralytics (YOLO) and NCNN
```bash
pip install ultralytics ncnn
```

## Step 4: Prepare the Model
Download the lightweight "Nano" version of YOLO26:
```bash
yolo detect predict model=yolo26n.pt
```
convert the PyTorch model to NCNN format:
```bash
yolo export model=yolo11n.pt format=ncnn
```
Connect webcam and check the camera, confirm the result show video 0 or 1:
```bash
ls dev/video*
```
Get the template python file:
```bash
wget https://edje.nyc/repo/yolo_detect.py
```
## Step 5: Running Inference
Run on Live Camera Feed
Ensure your USB webcam or Pi Camera is connected.
```bash
# Replace 'USB 0' with your camera index if necessary
python yolo_detect.py --model yolo11n_ncnn_model --source 0 --resolution 1280x720
```
#Reference
1. [How to Run YOLO Object Detection Models on the Raspberry Pi](https://www.youtube.com/watch?v=z70ZrSZNi-8)
2. [YOLO Object Detection on the Raspberry Pi AI HAT | Writing Python Scripts](https://www.youtube.com/watch?v=Zht2G1htFHA)
3. [YOLO Raspberry Pi AI HAT | Writing Python Scripts](https://github.com/hailo-ai/hailo-rpi5-examples?tab=readme-ov-file)
