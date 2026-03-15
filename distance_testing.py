import sys
import time
from cv2.typing import MatLike
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# %%

model = YOLO("yolo11l.pt")

# %%


def draw_boxes(frame: MatLike, results):
    for result in results:
        for box in result.boxes:
            # Get coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Get class name and confidence
            cls_id = int(box.cls[0])
            label = f"{model.names[cls_id]} {box.conf[0]:.2f}"

            # Draw rectangle and label using cv2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )


# %%


# results = model.track(source=0, show=True)

# %%

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

delay = 1
ret, frame = cap.read()
if not ret:
    print("Failed to grab frame")
    sys.exit(1)

print(frame)
results = model(frame)

draw_boxes(frame, results)

cv2.waitKey(1)
cv2.imshow("", frame)

input("")

ret, frame = cap.read()
if not ret:
    print("Failed to grab frame")
    sys.exit(1)

print(frame)
results = model(frame)

draw_boxes(frame, results)

cv2.waitKey(1)
cv2.imshow("", frame)
input("")
cap.release()
cv2.destroyAllWindows()
