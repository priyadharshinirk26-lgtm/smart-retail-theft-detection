import os
import cv2
from detector import TheftDetector
from audio_alert import AudioAlert


def select_video():
    """Get video path from user."""
    path = input("Enter full path to your video (or drag it here): ").strip().strip('"')
    if not os.path.exists(path):
        print("File not found, exiting.")
        return None
    return path

def main():
    video_path = select_video()
    if not video_path:
        return

    output_path = video_path.replace(".", "_detected.")
    print(f"Processing: {video_path}")
    print(f"Output will be saved as: {output_path}")

    detector = TheftDetector()
    alert_system = AudioAlert()

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    theft_detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        theft_score, annotated = detector.detect_theft(frame)

        # Graded alert (simple)
        if theft_score > 0.7 and not theft_detected:
            alert_system.play_beep("warning")
            cv2.putText(
                annotated,
                "THEFT ALERT!",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
            )
            theft_detected = True
        elif theft_score <= 0.3:
            theft_detected = False

        # Progress text on frame
        cv2.putText(
            annotated,
            f"Frame: {frame_count}  Score: {theft_score:.2f}",
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        out.write(annotated)

        # No cv2.imshow here – environment has no GUI
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames, score={theft_score:.2f}")

        out.write(annotated)

        cv2.imshow("Theft Detection - press Q to quit", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
          break
 

    cap.release()
    out.release()
    print("Done! Check:", output_path)
    os.startfile(output_path)

if __name__ == "__main__":
    main()

cv2.destroyAllWindows()
