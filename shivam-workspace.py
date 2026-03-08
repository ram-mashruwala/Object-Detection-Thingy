from ultralytics import YOLO
import cv2
import time

MODEL_PATH = "yolov8n.pt"
CAMERA_INDEX = 0


def main():
    # Load YOLO model
    model = YOLO(MODEL_PATH)

    # Open webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    print("Webcam started. Press 'q' to quit.")

    prev_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Warning: Could not read frame.")
            break

        # Run YOLO and measure detection time
        start_time = time.time()
        results = model(frame, conf=0.5, verbose=False)
        detection_time = time.time() - start_time

        # Draw detection boxes
        annotated_frame = results[0].plot()

        # Count detections
        boxes = results[0].boxes
        det_count = 0 if boxes is None else len(boxes)

        # Best detection
        best_conf = 0.0
        best_label = "none"
        if det_count > 0:
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            best_i = int(confs.argmax())
            best_conf = float(confs[best_i])
            best_label = model.names[clss[best_i]]

        # FPS calculation
        now = time.time()
        fps = 1 / (now - prev_time) if prev_time else 0.0
        prev_time = now

        # Overlay text
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            f"Detection Time: {detection_time * 1000:.1f} ms",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            f"Detections: {det_count}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            annotated_frame,
            f"Best: {best_label} {best_conf:.2f}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (200, 200, 255),
            2
        )

        # Show output
        cv2.imshow("YOLOv8 Object Detection", annotated_frame)

        # Quit on q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()