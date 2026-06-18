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
Install Python ext
