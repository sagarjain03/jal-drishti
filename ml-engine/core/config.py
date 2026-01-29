import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "weights")

# Model Paths
FUNIE_GAN_WEIGHTS = os.path.join(WEIGHTS_DIR, "funie_generator.pth")
YOLO_WEIGHTS = os.path.join(WEIGHTS_DIR, "yolov8n.pt")

# Thresholds
CONFIDENCE_THRESHOLD = 0.40
HIGH_CONFIDENCE_THRESHOLD = 0.75

# Output States
STATE_CONFIRMED_THREAT = "CONFIRMED_THREAT"
STATE_POTENTIAL_ANOMALY = "POTENTIAL_ANOMALY"
STATE_SAFE_MODE = "SAFE_MODE"

# Class Mappings (YOLOv8 COCO)
# We map everything to "anomaly" for this phase unless specific classes are needed
# For now, we keep the raw label or a generic one. Plan says "generic: anomaly"
generic_label = "anomaly"
