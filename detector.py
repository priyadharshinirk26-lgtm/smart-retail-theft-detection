from ultralytics import YOLO
import cv2
import numpy as np

class TheftDetector:
    def __init__(self, model_path="yolov8n.pt"):
        # This line creates the attribute you are missing
        print("Loading YOLO model...")
        self.model = YOLO(model_path)
        print("Model loaded.")

    def detect_theft(self, frame):
        h, w = frame.shape[:2]

        # Run YOLO on the frame
        results = self.model(frame, verbose=False)

        # Read detections safely
        if results[0].boxes is None:
            boxes = np.empty((0, 4))
            classes = np.empty((0,))
        else:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

        persons = 0
        items = 0
        for i in range(len(boxes)):
            cls = int(classes[i])
            if cls == 0:                  # person
                persons += 1
            elif cls in [39, 41, 42]:     # some product / bag classes
                items += 1

        theft_score = 0.8 if persons > 0 and items > 1 else 0.0

        annotated = results[0].plot()
        annotated = cv2.resize(annotated, (w, h))

        return theft_score, annotated
