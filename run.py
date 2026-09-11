import cv2
import numpy as np
import pygame
import os
import threading

# ---------- AUDIO ----------
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

def play_beep():
    def play():
        sample_rate = 22050
        duration = 0.3
        freq = 800
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(freq * t * 2 * np.pi)
        sound = (tone * 32767 / np.max(np.abs(tone))).astype(np.int16)
        stereo = np.repeat(sound[:, np.newaxis], 2, axis=1)
        pygame.mixer.Sound(buffer=stereo.tobytes()).play()
    threading.Thread(target=play, daemon=True).start()

# ---------- DETECTOR ----------
from ultralytics import YOLO

class TheftDetector:
    def __init__(self, model_path="yolov8n.pt"):
        print("Loading YOLO model...")
        self.model = YOLO(model_path)
        print("Model loaded.")
        self.roi = None  # no special region needed at first

    def detect_theft(self, frame):
        h, w = frame.shape[:2]
        results = self.model(frame, verbose=False)

        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else np.array([])
        classes = results[0].boxes.cls.cpu().numpy() if results[0].boxes is not None else np.array([])

        persons = 0
        items = 0
        for i in range(len(boxes)):
            cls = int(classes[i])
            if cls == 0:                # person
                persons += 1
            elif cls in [39, 41, 42]:   # some product / bag classes
                items += 1

        theft_score = 0.8 if persons > 0 and items > 1 else 0.0
        annotated = results[0].plot()
        annotated = cv2.resize(annotated, (w, h))

        return theft_score, annotated

# ---------- MAIN ----------
def main():
    video_path = r"C:\Users\priya\OneDrive\Desktop\mart_theft_detector\theft_video.mp4"
    if not os.path.exists(video_path):
        print("File not found. Check the path and try again.")
        return

    detector = TheftDetector()
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width  = int(cap.get(3))
    height = int(cap.get(4))
    output_path = video_path.replace(".", "_detected.")
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

    frame_count = 0
    theft_detected = False

    print(f"Processing: {video_path}")
    print(f"Output will be saved as: {output_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        theft_score, annotated = detector.detect_theft(frame)

        if theft_score > 0.7 and not theft_detected:
            play_beep()
            cv2.putText(annotated, "THEFT ALERT", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            theft_detected = True
        elif theft_score <= 0.3:
            theft_detected = False

        cv2.putText(annotated, f"Frame: {frame_count} Score: {theft_score:.2f}",
                    (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(annotated)
        cv2.imshow("Theft Detection - press q to quit", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Done!")

if __name__ == "__main__":
    main()
