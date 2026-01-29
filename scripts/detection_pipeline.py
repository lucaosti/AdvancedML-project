import cv2
import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

class YOLODetector:
    def __init__(self, model_path='yolov8n.pt', verbose=True):
        """Initialize YOLO detector with a pretrained model."""
        if verbose:
            print(f"Loading YOLO model from {model_path}...")
        try:
            self.model = YOLO(model_path)
            if verbose:
                print("YOLO model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def eval(self):
        """Set the YOLO model to evaluation mode."""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'eval'):
            self.model.model.eval()

    def __call__(self, x):
        """Forward pass through the YOLO model."""
        return self.model(x, verbose=False)

    def detect_and_crop(self, image_path, depth_path=None, target_class_id=None, crop_size=(224, 224), padding_pct=0.1):
        """Perform inference, extract bounding box, and return resized crops."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image at {image_path}")
        
        h_img, w_img = img.shape[:2]
        
        depth_img = None
        if depth_path is not None:
            depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            if depth_img is None:
                print(f"Warning: Could not load depth map at {depth_path}")

        results = self.model(img, verbose=False)
        boxes = results[0].boxes
        
        if len(boxes) == 0:
            print("No objects detected.")
            return None, None, None, None, None
        
        selected_box = None
        best_conf = -1.0

        for box in boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                selected_box = box

        if selected_box is None:
            print("No valid object found.")
            return None, None, None, None, None

        cls_id = int(selected_box.cls[0])
        class_name = self.model.names[cls_id]

        x1_orig, y1_orig, x2_orig, y2_orig = selected_box.xyxy[0].cpu().numpy()
        
        bw = x2_orig - x1_orig
        bh = y2_orig - y1_orig
        pad_x = int(bw * padding_pct)
        pad_y = int(bh * padding_pct)

        x1 = max(0, int(x1_orig - pad_x))
        y1 = max(0, int(y1_orig - pad_y))
        x2 = min(w_img, int(x2_orig + pad_x))
        y2 = min(h_img, int(y2_orig + pad_y))
        
        bbox_absolute = (x1, y1, x2, y2)
        
        bbox_normalized = (
            x1 / w_img,
            y1 / h_img,
            x2 / w_img,
            y2 / h_img
        )

        crop_rgb = img[y1:y2, x1:x2]
        crop_resized_rgb = cv2.resize(crop_rgb, crop_size, interpolation=cv2.INTER_LINEAR)
        crop_resized_rgb = cv2.cvtColor(crop_resized_rgb, cv2.COLOR_BGR2RGB)
        
        crop_resized_depth = None
        if depth_img is not None:
            crop_depth = depth_img[y1:y2, x1:x2]
            crop_resized_depth = cv2.resize(crop_depth, crop_size, interpolation=cv2.INTER_NEAREST)
        
        return crop_resized_rgb, crop_resized_depth, bbox_normalized, bbox_absolute, class_name