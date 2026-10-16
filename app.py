import cv2
import mediapipe as mp
import threading
import time
import csv
import math

from pathlib import Path
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import (
    HTMLResponse,
    StreamingResponse,
    JSONResponse,
)

app = FastAPI(title="Human Pose Tracker")

BASE_DIR = Path(__file__).parent
HTML_FILE = BASE_DIR / "templates" / "index.html"
CSV_FILE = BASE_DIR / "movement_history.csv"

camera_lock = threading.Lock()
stop_event = threading.Event()

metrics_lock = threading.Lock()
latest_metrics = {
    "pose_status": "NOT DETECTED",
    "activity": "WAITING",
    "movement_score": 0.0,
    "fps": 0.0,
}


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_FILE.read_text(encoding="utf-8")


@app.get("/metrics")
def get_metrics():
    with metrics_lock:
        return dict(latest_metrics)


def save_history(pose_status, activity, score, fps):
    file_exists = CSV_FILE.exists()

    try:
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "pose_status",
                    "activity",
                    "movement_score",
                    "fps",
                ])

            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                pose_status,
                activity,
                round(score, 2),
                round(fps, 2),
            ])

    except OSError as error:
        print(f"CSV logging error: {error}")


def generate_frames():
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

        previous_points = None
        movement_score = 0.0
        fps = 0.0

        frame_count = 0
        fps_start = time.perf_counter()
        last_log_time = time.monotonic()

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

                pose_status = "NOT DETECTED"
                activity = "NO POSE"

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    mp_draw.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                    )

                    # Use visible landmarks for movement measurement.
                    points = [
                        (i, lm.x, lm.y)
                        for i, lm in enumerate(landmarks)
                        if lm.visibility >= 0.5
                    ]

                    current_points = {
                        i: (x, y) for i, x, y in points
                    }

                    if previous_points:
                        common = (
                            current_points.keys()
                            & previous_points.keys()
                        )

                        distances = [
                            math.dist(
                                current_points[i],
                                previous_points[i],
                            )
                            for i in common
                        ]

                        if distances:
                            movement_score = (
                                sum(distances) / len(distances)
                            ) * 1000
                        else:
                            movement_score = 0.0
                    else:
                        movement_score = 0.0

                    previous_points = current_points
                    pose_status = "DETECTED"

                    # A simple threshold; we can tune it after testing.
                    activity = (
                        "MOVING"
                        if movement_score >= 2.5
                        else "STATIONARY"
                    )

                    status_color = (0, 255, 0)

                else:
                    previous_points = None
                    movement_score = 0.0
                    status_color = (0, 0, 255)

                frame_count += 1
                now = time.perf_counter()
                elapsed = now - fps_start

                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    frame_count = 0
                    fps_start = now

                # Display analytics on the live camera feed.
                lines = [
                    (f"POSE: {pose_status}", status_color),
                    (f"ACTIVITY: {activity}", (255, 255, 0)),
                    (
                        f"MOVEMENT SCORE: {movement_score:.2f}",
                        (255, 255, 255),
                    ),
                    (f"FPS: {fps:.1f}", (255, 255, 255)),
                ]

                for index, (label, color) in enumerate(lines):
                    cv2.putText(
                        frame,
                        label,
                        (15, 30 + index * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                with metrics_lock:
                    latest_metrics.update({
                        "pose_status": pose_status,
                        "activity": activity,
                        "movement_score": round(
                            movement_score, 2
                        ),
                        "fps": round(fps, 2),
                    })

                # Save one history record approximately each second.
                log_now = time.monotonic()

                if log_now - last_log_time >= 1.0:
                    save_history(
                        pose_status,
                        activity,
                        movement_score,
                        fps,
                    )
                    last_log_time = log_now

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