import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from mcunet.model_zoo import build_model

# ─── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME   = 'mcunet-in4'
MODEL_PATH   = 'content/pruned.pth'
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
INPUT_SIZE   = (160, 160)
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base = os.path.splitext(os.path.basename(MODEL_PATH))[0]
SAVE_PATH = f"{base}_camera.txt"

# ─── Load pretrained model ────────────────────────────────────────────────────
def load_pretrained_model(model_name: str, weight_path: str, num_classes=10, device=DEVICE):

    model, resolution, desc = build_model(model_name)


    if hasattr(model, 'classifier'):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError("Cannot find final classification layer in model")


    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model

if MODEL_NAME == 'hard':
    model = torch.load("hardpruned_.pth")
    model.eval()
    SAVE_PATH = 'hard_camera.txt'
else:
    model = load_pretrained_model(MODEL_NAME, MODEL_PATH)
print("✅ Model load successful")


# ─── Video capture ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

infer_times = []
fps_list    = []
frame_count = 0
prev_t = time.time()
start_time = time.time()

while True:
    if time.time() - start_time >= 30.0:
        print("30s")
        break

    ret, frame = cap.read()
    if not ret: 
        print("camera failed")
        break
    img = cv2.resize(frame, INPUT_SIZE)
    tensor = (torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
              .permute(2,0,1).unsqueeze(0).float().to(DEVICE) / 255.0)

    t0 = time.time()
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
    t1 = time.time()
    infer_times.append((t1 - t0) * 1000)

    fps = 1.0 / (t1 - prev_t)
    infer_ms   = (t1 - t0) * 1000
    prev_t = t1
    infer_times.append(infer_ms)
    fps_list.append(fps)
    frame_count += 1

    label = f"{CLASS_NAMES[pred.item()]} {conf.item():.2f}"
    cv2.putText(img, label, (5,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(img, f"FPS:{fps:.1f}", (5,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    cv2.imshow('Infer', img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

# ─── Compute & dump summary ────────────────────────────────────────────────────
if len(infer_times) > 0:
    avg_inf = sum(infer_times) / len(infer_times)
min_inf = min(infer_times)
max_inf = max(infer_times)

if len(fps_list) > 0:
    avg_fps = sum(fps_list) / len(fps_list)
min_fps = min(fps_list)
max_fps = max(fps_list)

with open(SAVE_PATH, 'w') as f:
    f.write(f"Model file: {MODEL_PATH}\n")
    f.write(f"Frames processed: {frame_count}\n")
    f.write("-"*40 + "\n")
    f.write(f"Inference time (ms): avg {avg_inf:.2f}, min {min_inf:.2f}, max {max_inf:.2f}\n")
    f.write(f"FPS             : avg {avg_fps:.2f}, min {min_fps:.2f}, max {max_fps:.2f}\n")