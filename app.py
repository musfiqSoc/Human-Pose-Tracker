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

# Shared reference to the active camera.
camera_state_lock = threading.Lock()
active_camera = None

metrics_lock = threading.Lock()
latest_metrics = {
    "pose_status": "NOT DETECTED",
    "activity": "WAITING",
    "movement_score": 0.0,
    "fps": 0.0,
}

# Activity Summary
summary_lock = threading.Lock()

activity_summary = {
    "total_tracking_seconds": 0.0,
    "moving_seconds": 0.0,
    "stationary_seconds": 0.0,
    "movement_events": 0,
}


def reset_summary():
    with summary_lock:
        activity_summary.update({
            "total_tracking_seconds": 0.0,
            "moving_seconds": 0.0,
            "stationary_seconds": 0.0,
            "movement_events": 0,
        })


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_FILE.read_text(encoding="utf-8")


@app.get("/metrics")
def get_metrics():
    with metrics_lock:
        return dict(latest_metrics)


@app.get("/summary")
def get_summary():
    with summary_lock:
        result = dict(activity_summary)

    for key in (
        "total_tracking_seconds",
        "moving_seconds",
        "stationary_seconds",
    ):
        result[key] = round(result[key], 2)

    return result


def save_history(pose_status, activity, score, fps):
    file_exists = CSV_FILE.exists()

    try:
        with open(
            CSV_FILE,
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)

            if not file_exists or CSV_FILE.stat().st_size == 0:
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
    global active_camera

    if not camera_lock.acquire(blocking=False):
        print("Tracking is already running.")
        return

    camera = None

    try:
        # Stop may have been requested before the stream started.
        if stop_event.is_set():
            print("STOP EVENT DETECTED before camera startup.")
            return

        reset_summary()

        mp_pose = mp.solutions.pose
        mp_draw = mp.solutions.drawing_utils

        # Open webcam on Windows, with a fallback backend.
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not camera.isOpened():
            camera.release()
            camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            print("ERROR: Could not open webcam.")
            return

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Publish the active camera so Stop can release it.
        with camera_state_lock:
            if stop_event.is_set():
                camera.release()
                camera = None
                print("STOP EVENT DETECTED during camera startup.")
                return

            active_camera = camera

        previous_points = None
        movement_score = 0.0
        fps = 0.0

        last_summary_time = time.monotonic()
        last_log_time = last_summary_time

        was_moving = False

        frame_count = 0
        fps_start = time.perf_counter()

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:

            while True:
                if stop_event.is_set():
                    print("STOP EVENT DETECTED")
                    break

                success, frame = camera.read()

                # Stop may have been requested during camera.read().
                if stop_event.is_set():
                    print("STOP EVENT DETECTED after camera read.")
                    break

                if not success:
                    if not stop_event.is_set():
                        print("WARNING: Could not read webcam frame.")
                    break

                frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = pose.process(rgb)

                pose_status = "NOT DETECTED"
                activity = "NO POSE"
                status_color = (0, 0, 255)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    mp_draw.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                    )

                    current_points = {
                        i: (lm.x, lm.y)
                        for i, lm in enumerate(landmarks)
                        if lm.visibility >= 0.5
                    }

                    if previous_points is not None:
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
                            ) * 1000.0
                        else:
                            movement_score = 0.0
                    else:
                        movement_score = 0.0

                    previous_points = current_points
                    pose_status = "DETECTED"
                    status_color = (0, 255, 0)

                    activity = (
                        "MOVING"
                        if movement_score >= 2.5
                        else "STATIONARY"
                    )

                else:
                    previous_points = None
                    movement_score = 0.0

                # Update elapsed tracking time.
                now = time.monotonic()
                delta_seconds = max(
                    0.0,
                    now - last_summary_time,
                )
                last_summary_time = now

                with summary_lock:
                    activity_summary[
                        "total_tracking_seconds"
                    ] += delta_seconds

                    if pose_status == "DETECTED":
                        if activity == "MOVING":
                            activity_summary[
                                "moving_seconds"
                            ] += delta_seconds

                            if not was_moving:
                                activity_summary[
                                    "movement_events"
                                ] += 1

                            was_moving = True

                        else:
                            activity_summary[
                                "stationary_seconds"
                            ] += delta_seconds

                            was_moving = False
                    else:
                        was_moving = False

                # Calculate FPS.
                frame_count += 1
                perf_now = time.perf_counter()
                elapsed = perf_now - fps_start

                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    frame_count = 0
                    fps_start = perf_now

                # Draw live information on the camera frame.
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

                # Save history approximately once per second.
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
        if stop_event.is_set():
            print(f"Tracking stopped during camera processing: {error}")
        else:
            print(f"Tracking error: {error}")

    finally:
        # Remove the shared camera reference safely.
        with camera_state_lock:
            if active_camera is camera:
                active_camera = None

        if camera is not None:
            try:
                camera.release()
            except Exception as error:
                print(f"Camera release error: {error}")

        camera_lock.release()
        print("Camera released.")


@app.get("/video_feed")
def video_feed():
    # Reset the stop signal before starting a new stream.
    stop_event.clear()

    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/stop_tracking")
def stop_tracking():
    global active_camera

    print("STOP REQUEST RECEIVED")
    stop_event.set()

    # Release the webcam directly when Stop is requested.
    with camera_state_lock:
        camera_to_release = active_camera
        active_camera = None

    if camera_to_release is not None:
        try:
            camera_to_release.release()
            print("Camera release requested by Stop button.")
        except Exception as error:
            print(f"Camera stop error: {error}")

    return JSONResponse({
        "status": "stop_requested"
    })


@app.get("/history")
def get_movement_history():
    if not CSV_FILE.exists():
        return {
            "records": [],
            "message": "No movement history available yet.",
        }

    records = []

    try:
        with open(
            CSV_FILE,
            "r",
            newline="",
            encoding="utf-8",
        ) as file:
            reader = csv.DictReader(file)

            for row in reader:
                try:
                    records.append({
                        "timestamp": row["timestamp"],
                        "pose_status": row["pose_status"],
                        "activity": row["activity"],
                        "movement_score": float(
                            row["movement_score"]
                        ),
                        "fps": float(row["fps"]),
                    })
                except (KeyError, TypeError, ValueError):
                    continue

    except OSError as error:
        return JSONResponse(
            {"error": str(error)},
            status_code=500,
        )

    return {
        "records": records,
        "total_records": len(records),
    }