import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import sys
import os
import psutil
import time
import datetime
from torchvision import transforms
import matplotlib.pyplot as plt
from mcunet.model_zoo import build_model

# ----------------------------- #
#    Device setup
# ----------------------------- #
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ----------------------------- #
#    Load the MCUNet model
# ----------------------------- #
model, resolution, desc = build_model('mcunet-in4')

# Adjust the final classifier layer to match CIFAR-10 (10 classes)
if hasattr(model, 'classifier'):
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 10)
else:
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 10)

# Move model to the specified device
model = model.to(device)

# Load pretrained weights
model.load_state_dict(torch.load(
    '/Users/daniel_feng/Desktop/MCUNet_efficient/models/mcunet_cifar10_v1.pth',
    map_location=device
), strict=False)
model.eval()
print("✅ Model load success!")

# ----------------------------- #
#    Image preprocessing
# ----------------------------- #
transform = transforms.Compose([
    transforms.ToPILImage(),                      # Convert numpy array to PIL Image
    transforms.Resize((resolution, resolution)),  # Resize to model's input resolution
    transforms.ToTensor(),                        # Convert PIL Image to Tensor
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3) # Normalize pixel values
])

# CIFAR-10 class names
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# ----------------------------- #
#    Open the webcam
# ----------------------------- #
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Can not open camera!")
    exit()

# Get the current process for resource monitoring
process = psutil.Process(os.getpid())

# ----------------------------- #
#    Prepare real-time plotting
# ----------------------------- #
cpu_mem_list = []         # List to record CPU memory usage
cpu_usage_list = []       # List to record CPU usage percentage
inference_time_list = []  # List to record inference time per frame
fps_list = []             # List to record FPS per frame
frames = []               # Frame index list

# Set up real-time matplotlib figure
plt.ion()
fig, axs = plt.subplots(4, 1, figsize=(10, 10))

def update_plot():
    """Refresh the matplotlib graphs every few frames."""
    axs[0].clear()
    axs[0].plot(frames, cpu_mem_list, label="CPU Memory (MB)")
    axs[0].legend()
    axs[0].set_ylabel("Memory MB")

    axs[1].clear()
    axs[1].plot(frames, cpu_usage_list, label="CPU Usage (%)", color='orange')
    axs[1].legend()
    axs[1].set_ylabel("CPU Usage %")

    axs[2].clear()
    axs[2].plot(frames, inference_time_list, label="Inference Time (ms)", color='green')
    axs[2].legend()
    axs[2].set_ylabel("Inference Time ms")

    axs[3].clear()
    axs[3].plot(frames, fps_list, label="FPS", color='red')
    axs[3].legend()
    axs[3].set_ylabel("FPS")
    axs[3].set_xlabel("Frame")

    plt.pause(0.001)  # Short pause to refresh plot

# ----------------------------- #
#    Start real-time inference loop
# ----------------------------- #
frame_idx = 0
fps_time = time.time()

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("❌ Can not load frame!")
        break

    # Preprocess frame
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Measure inference time
    start_inference = time.time()

    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()

    end_inference = time.time()
    inference_time_ms = (end_inference - start_inference) * 1000  # ms

    # Calculate FPS based on frame read time
    now_time = time.time()
    fps = 1.0 / (now_time - fps_time)
    fps_time = now_time

    # Collect CPU resource usage
    cpu_memory_MB = process.memory_info().rss / (1024 * 1024)
    cpu_usage_percent = process.cpu_percent(interval=0)

    # Record monitoring data
    frames.append(frame_idx)
    cpu_mem_list.append(cpu_memory_MB)
    cpu_usage_list.append(cpu_usage_percent)
    inference_time_list.append(inference_time_ms)
    fps_list.append(fps)
    frame_idx += 1

    # Update plot every 5 frames
    if frame_idx % 5 == 0:
        update_plot()

    # Overlay prediction info on frame
    label_text = f"{class_names[pred_idx]} ({confidence*100:.1f}%)"
    fps_text = f"FPS: {fps:.2f}"
    memory_text = f"CPU: {cpu_memory_MB:.1f}MB {cpu_usage_percent:.1f}%"

    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, memory_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("MCUNet Real-Time Classification", frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------- #
#    Release resources
# ----------------------------- #
cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
