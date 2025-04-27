
# MCUNet for Image Classification (MacOS)

This project provides a real-time camera classification system based on a pretrained **MCUNet-in4** model.  
It features:
- Real-time webcam inference
- Real-time system performance monitoring (CPU memory, CPU usage, FPS, inference time)
- Live plotting of system metrics
- Specifically optimized for MacOS environments

---

## 📦 Project Structure

```bash
application/
├── mcunet/                  # MCUNet model implementation
├── models/                  # Pretrained MCUNet model (mcunet_cifar10_v1.pth)
├── scripts/                 # (Optional) Additional utility scripts
├── utils/                   # (Optional) Supporting functions
├── application_cv_cifar10.py # Main script for real-time webcam inference
├── README.md                 # Project setup and usage guide
├── requirements.txt          # Required Python packages
```

---

## 🚀 Quick Start (MacOS)

### 1. Move into the application directory

```bash
cd Efficient_ML_mcunet/application
```

---

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

> **Tip:** After activation, you should see `(venv)` at the start of your terminal prompt.

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Required packages include:
- `torch`
- `torchvision`
- `opencv-python`
- `psutil`
- `matplotlib`
- others

---

### 4. Verify camera permissions on MacOS

Before running, ensure that your Terminal/VSCode has permission to access your camera:

- Go to **System Settings → Privacy & Security → Camera**
- Allow access for your Terminal or VSCode

---

### 5. Run the application

Activate your virtual environment if it's not already active:

```bash
source venv/bin/activate
```

Then run:

```bash
python application_cv_cifar10.py
```

This will:
- Open your Mac’s webcam
- Classify frames in real-time based on CIFAR-10 classes
- Overlay prediction labels, FPS, and CPU usage on the video feed
- Plot system performance metrics live using Matplotlib

---

## 🎥 What You Will See

- A live webcam window showing:
  - Predicted class and confidence
  - Real-time FPS
  - Real-time CPU memory usage
- A separate Matplotlib window with graphs:
  - CPU memory usage
  - CPU usage percentage
  - Inference time per frame
  - FPS over time

Example overlay:

```
Predicted: bird (95.2%)
FPS: 29.87
CPU: 421.4MB 12.3%
```

---

## 🛠️ Troubleshooting

| Problem | Solution |
|:--------|:---------|
| Cannot open webcam | Check camera permissions in System Settings |
| Import errors | Make sure your virtual environment is activated |
| Webcam is black or slow | Ensure no other application (Zoom, Teams) is occupying the camera |

---
