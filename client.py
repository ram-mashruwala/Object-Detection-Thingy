import numpy as np
import requests
import cv2
import threading

# Choose webcam here
webcam_id = 0


# This thread will request a new depth image from the server all the time.
def update_depth(state):
    while True:
        frame = state.frame

        data = cv2.imencode(".jpg", frame)[1].tobytes()

        url = "http://localhost:5000/process_image"

        with requests.post(url, files={"image": data}) as r:
            r.raise_for_status()
            data = r.content

        state.depth = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


class ThreadState:
    def __init__(self, frame):
        self.frame = frame
        self.depth = None


try:
    cam = cv2.VideoCapture(webcam_id)

    thread = None

    while True:
        ok, frame = cam.read()

        cv2.imshow("frame", frame)

        if thread is None:
            state = ThreadState(frame)
            thread = threading.Thread(target=update_depth, args=(state,))
            thread.daemon = True
            thread.start()

        state.frame = frame

        if not ok:
            print("Cannot read frame from webcam")
            break

        if state.depth is not None:
            cv2.imshow("depth", state.depth)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception as e:
    import traceback

    traceback.print_exc()
finally:
    cam.release()
    cv2.destroyAllWindows()
