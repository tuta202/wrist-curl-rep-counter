import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

MOTION_THRESHOLD = 5
REP_UP_THRESHOLD = -5
REP_DOWN_THRESHOLD = 5

# Load models
model_det = YOLO("./runs/detect/train/weights/best.pt")  # custom model to detect weight
model_pose = YOLO("yolo11n-pose.pt")  # YOLOv8 pose model (you can use s/m/l)

# Open video
video_path = "./test_images/wrist_curl_test_2.mp4"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS:", fps, "COOLDOWN:", int(fps / 2))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("output_tracking_2.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

track_history = defaultdict(lambda: [])

# Loop video
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_height, frame_width = frame.shape[:2]

    # Run pose estimation (YOLO pose)
    pose_results = model_pose.predict(frame, verbose=False, stream=False)[0]
    nose_list = []

    for kp in pose_results.keypoints:
        kps = kp.xy[0].cpu().numpy()  # shape (17, 2)
        nose = kps[0]  # Nose = keypoint 0
        nose_list.append((int(nose[0]), int(nose[1])))
        # Optional: draw nose
        cv2.circle(frame, (int(nose[0]), int(nose[1])), 5, (255, 255, 0), -1)

    # Run object tracking (YOLO)
    result = model_det.track(frame, persist=True, verbose=False, stream=False)[0]

    if result.boxes and result.boxes.is_track:
        boxes = result.boxes.xywh.cpu()
        track_ids = result.boxes.id.int().cpu().tolist()
        # frame = result.plot()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            center = (float(x), float(y))
            track.append(center)
            
            # Initialize rep tracking
            if "rep_state" not in track_history:
                track_history["rep_state"] = {}

            if track_id not in track_history["rep_state"]:
                track_history["rep_state"][track_id] = {
                    "count": 0,
                    "state": "idle",     # can be: idle → up → down : 1rep → up → down : 2reps → up → down : 3reps
                    "last_y": y,
                    "cooldown": 0
                }

            rep_info = track_history["rep_state"][track_id]
            delta_y = y - rep_info["last_y"]
            rep_info["last_y"] = y

            # Apply cooldown to avoid rapid duplicate counts
            if rep_info["cooldown"] > 0:
                rep_info["cooldown"] -= 1

            # State machine for rep detection
            if rep_info["state"] == "idle":
                if delta_y < REP_UP_THRESHOLD:  # moving upward
                    rep_info["state"] = "up"

            elif rep_info["state"] == "down":
                if delta_y < REP_UP_THRESHOLD:  # upward
                    rep_info["state"] = "up"

            elif rep_info["state"] == "up":
                if delta_y > REP_DOWN_THRESHOLD:  # down again = full rep
                    if rep_info["cooldown"] == 0:
                        rep_info["count"] += 1
                        rep_info["cooldown"] = int(fps / 2)  # skip next few frames, 30 FPS → 10 frame = ~0.33 s
                    rep_info["state"] = "down"
            
            if len(track) > 30:
                track.pop(0)

            # 1. Motion filter
            y_positions = [pt[1] for pt in track]
            if max(y_positions) - min(y_positions) < MOTION_THRESHOLD:
                continue

            # 2. Only accept box if its center is LOWER than nose
            accept = False
            for nx, ny in nose_list:
                if y > ny:  # y : center.y box
                    accept = True
                    break
            if not accept:
                continue

            # 3. Draw track
            bx1, by1 = int(x - w / 2), int(y - h / 2)
            bx2, by2 = int(x + w / 2), int(y + h / 2)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (bx2 - 60, by1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Reps: {rep_info['count']}", (bx1, by1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 0), thickness=6)

    # Show frame
    cv2.imshow("YOLO + Pose Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    out.write(frame)

cap.release()
cv2.destroyAllWindows()
out.release()