# Setup Raspberry PI
How to setup Raspberry OS

## Step 1: Install OS
 [Install using Imager](https://www.raspberrypi.com/software/) 

 [Setup the Raspberry Pi](https://www.raspberrypi.com/documentation/computers/getting-started.html)

 setup clock : 
 
```bash
sudo date -s "$(wget -qSO- --max-redirect=0 google.com 2>&1 | grep Date: | cut -d' ' -f5-8)Z"
```
 ## Step 2: Install VS Code
Open VS Code
Install Python ext

[Create tem.py file](https://github.com/Noraide/raspi_yolo/blob/main/tem.py)

 ## Step 3: Install ultralytic to run Yolo Model 
 [Yolo on Raspberry PI](https://github.com/Noraide/raspi_yolo/b;ob/main/YOLO.md)



