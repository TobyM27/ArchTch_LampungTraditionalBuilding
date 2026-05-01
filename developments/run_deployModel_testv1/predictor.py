import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

def get_device():
    if torch.backends.mps.is_available():
        return "mps"       # Apple Silicon
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

def load_model(model_path: str = "saved_model/model.pt"):
    device = get_device()
    model = YOLO(model_path)
    model.to(device)
    return model

def run_inference(model: YOLO, image: Image.Image, conf_threshold: float = 0.25):
    """
    Run YOLOv11 detection + segmentation on a PIL image.
    Returns the result object from Ultralytics.
    """
    results = model.predict(
        source=image,
        conf=conf_threshold,    # confidence threshold
        iou=0.45,               # NMS IoU threshold
        imgsz=640,              # inference size
        verbose=False
    )
    return results[0]           # single image → single result

def draw_results(result, class_colors: dict = None) -> np.ndarray:
    """
    Draw bounding boxes + segmentation masks on image.
    Returns annotated image as numpy array (RGB).
    """
    # Get annotated frame from Ultralytics built-in renderer
    annotated = result.plot(
        masks=True,         # draw segmentation masks
        boxes=True,         # draw bounding boxes
        labels=True,        # draw class labels
        conf=True,          # show confidence scores
        line_width=2
    )
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)  # convert to RGB for Streamlit

def parse_detections(result) -> list[dict]:
    """
    Parse raw YOLO result into a clean list of detections.
    """
    detections = []
    names = result.names  # class index → class name mapping

    boxes = result.boxes
    has_masks = result.masks is not None

    for i, box in enumerate(boxes):
        detection = {
            "class_id":    int(box.cls),
            "class_name":  names[int(box.cls)],
            "confidence":  float(box.conf),
            "bbox_xyxy":   box.xyxy[0].tolist(),   # [x1, y1, x2, y2]
            "has_mask":    has_masks,
        }
        detections.append(detection)

    return detections