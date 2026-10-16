import cv2
import mediapipe as mp
import threading

from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

app = FastAPI(title="Human Pose Tracker")

HTML_FILE = Path(__file__).parent / "templates" / "index.html"

camera_lock = threading.Lock()
stop_event = threading.Event()


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_FILE.read_text(encoding="utf-8")


def generate_frames():
    # Prevent multiple streams from opening the webcam
    if not camera_lock.acquire(blocking=False):
        return

    camera = None

    try:
        stop_event.clear()

        mp_pose = mp.solutions.pose
        mp_draw = mp.solutions.drawing_utils

        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not camera.isOpened():
            camera.release()
            camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            print("ERROR: Could not open webcam.")
            return

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:

            while not stop_event.is_set():
                success, frame = camera.read()

                if not success:
                    break

                frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False

                results = pose.process(rgb)

                if results.pose_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                    )
                    status = "POSE: DETECTED"
                    color = (0, 255, 0)
                else:
                    status = "POSE: NOT DETECTED"
                    color = (0, 0, 255)

                cv2.putText(
                    frame,
                    status,
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )

                success, buffer = cv2.imencode(".jpg", frame)

                if not success:
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buffer.tobytes()
                    + b"\r\n"
                )

    except Exception as error:
        print(f"Tracking error: {error}")

    finally:
        if camera is not None:
            camera.release()

        camera_lock.release()
        print("Camera released.")


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/stop_tracking")
def stop_tracking():
    stop_event.set()
    return JSONResponse({"status": "stop_requested"})