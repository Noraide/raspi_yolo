# Deploy Model on Raspberry PI
How to deploy a model

## Step 1: Create Folder
3 files

1. inference.py
2. tensorflow file ( .tflite) download from Edge Impulse 
3. labels.tx

## Step 2: Setup Dependencies
```bash
python3 -m venv env --system-site-packages
```
```bash
env/bin/pip install ai-edge-litert opencv-python
```

## Step 3: Run inference
```bash
source env/bin/activate
```
```bash
python inference.py
```



