import time
import depth_pro
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# %%

model = YOLO("yolo11l.pt")
model_depth, transform = depth_pro.create_model_and_transforms()
model_depth.eval()

# %%

# results = model.track(source=0, show=True)

# %%

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

delay = 1
try:
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        print(frame)
        results = model(frame)
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

        image = transform(frame)

        cv2.imshow("", frame)
        cv2.waitKey(1)

        # prediction = model_depth.infer(image, f_px=0.1)
        # depth = prediction["depth"]  # Depth in [m].
        # depth_np = depth.sqeeze().cpu().numpy()
        # print(depth_np)
        end_time = time.time()
        # if end_time - start_time < delay:
        #     time.sleep(delay - (end_time - start_time))

except KeyboardInterrupt:
    cap.release()
    cv2.destroyAllWindows()


# %%
