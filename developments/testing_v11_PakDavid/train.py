from ultralytics import YOLO

# Load pretrained YOLOv11 nano (downloads auto)
model = YOLO("yolo11n.pt")
from ultralytics import YOLO

# Load pretrained YOLOv11 nano (downloads auto)
model = YOLO("yolo11n.pt")

# Train (MPS auto; specify device='mps')
results = model.train(
    data="datasets/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,  # Adjust for M3 RAM
    device="mps",  # Or 0/'cpu'
    project="runs/train",
    name="exp1",
    workers=8
)

# Auto-saves runs/train/exp1/weights/best.pt, results.csv (Excel viz)[web:20][page:2]