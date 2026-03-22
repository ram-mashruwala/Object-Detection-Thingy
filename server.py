from ultralytics import YOLO
import cv2
import numpy as np
import torch
import time
import io
from PIL import Image
from flask import Flask, request, send_file
from depth_pro import create_model_and_transforms, load_rgb


def get_torch_device() -> torch.device:
    """Get the Torch device."""
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    return device


print("Loading model...")

device = get_torch_device()

model, transform = create_model_and_transforms(device=device, precision=torch.half)
model.eval()
model_yolo = YOLO("yolo11l.pt")

app = Flask(__name__)


@app.route("/process_image", methods=["POST"])
def process_image():
    with torch.no_grad():
        # Load, resize and transform image
        file = request.files["image"]
        image = Image.open(file.stream).convert("RGB")

        image_un_transformed = np.array(image)
        image = transform(image_un_transformed)

        # Inference
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        prediction = model.infer(image)
        torch.cuda.synchronize()
        elapsed_time = time.perf_counter() - start_time
        print(f"Time for model.infer(image): {elapsed_time:.3f} sec")

        # Transform depth to inverse depth
        depth = prediction["depth"].detach().cpu().numpy().squeeze()
        print(len(depth), len(depth[0]), depth.max())
        results = model_yolo(image_un_transformed)
        for r in results:
            img = r.orig_img.copy()  # Get the original numpy array
            for box in r.boxes:
                # Get coordinates as integers (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # center_x = (x1 + x2) // 2
                # center_y = (y1 + y2) // 2
                #
                # depth_value = depth[center_y, center_x]

                min_value = float("inf")

                for y in range(y1, y2):
                    for x in range(x1, x2):
                        if depth[y, x] < min_value:
                            min_value = depth[y, x]

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Optional: Add label
                label = f"{model_yolo.names[int(box.cls[0])]} {float(box.conf):.2f} Depth: {min_value:.2f}"
                cv2.putText(
                    img,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
        # Convert to JPEG
        # output_image = Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))
        output_image = Image.fromarray(img)
        img_io = io.BytesIO()
        output_image.save(img_io, "JPEG")
        img_io.seek(0)

        # Send it
        return send_file(img_io, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
