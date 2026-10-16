import cv2
import mediapipe as mp
import time

def main():
    print("Starting Human Pose Tracker...")

    # MediaPipe Pose setup
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,  # Lightweight for an 8 GB RAM PC
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Open default webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Fallback if DirectShow cannot open the camera
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Webcam open hoyni.")
        print("Check camera connection and Windows camera permission.")
        pose.close()
        return

    # Reduce resolution to help performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    previous_time = time.time()

    print("Webcam connected!")
    print("Stand where your body is visible.")
    print("Press Q to quit.")

    try:
        while True:
            success, frame = cap.read()

            if not success:
                print("ERROR: Webcam frame read kora jacche na.")
                break

            # Mirror the image
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )
                status = "POSE: DETECTED"
                status_color = (0, 255, 0)
            else:
                status = "POSE: NOT DETECTED"
                status_color = (0, 0, 255)

            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / max(current_time - previous_time, 1e-6)
            previous_time = current_time

            cv2.putText(
                frame, status, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                status_color, 2, cv2.LINE_AA
            )

            cv2.putText(
                frame, f"FPS: {fps:.1f}", (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2, cv2.LINE_AA
            )

            cv2.imshow("Human Pose Tracker | Press Q to quit", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nStopping tracker...")

    finally:
        cap.release()
        pose.close()
        cv2.destroyAllWindows()
        print("Human Pose Tracker stopped.")

if __name__ == "__main__":
    main()