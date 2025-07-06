import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import lap
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

# Load models
model_det = YOLO("./runs/detect/train/weights/best.pt")  # custom model to detect weight
model_pose = YOLO("yolo11n-pose.pt")  # YOLO11n pose model

# Open video
# video_path = "./test_images/wrist_curl_test.mp4"
# cap = cv2.VideoCapture(video_path)
# Open webcam
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("output_tracking.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# Constants
MOTION_THRESHOLD = 10 # ko thỏa mãn cái này thì chỉ bị tắt draw bounding box đi thôi
REP_UP_THRESHOLD = -8
REP_DOWN_THRESHOLD = 8
MAX_TRACK_LENGTH = 20
TRACK_EVERY_N_FRAMES = 2 

frame_index = 0
nose_position = None
track_history = defaultdict(lambda: [])
# Loop video
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    if frame_index % int(fps / 2) == 0:  # skip running pose estimation. ~0.5 s
        print('Run pose estimation. Frame: ', frame_index)
        pose_results = model_pose.predict(frame, verbose=False, stream=False)[0]
        if len(pose_results.keypoints) > 0:
            kps = pose_results.keypoints[0].xy[0].cpu().numpy() # pose_results.keypoints[0] -> Only 1 person
            nose_keypoint = kps[0]  # Nose = keypoint 0
            nose_position = (int(nose_keypoint[0]), int(nose_keypoint[1])) 
    if nose_position:
        cv2.circle(frame, nose_position, 5, (255, 255, 0), -1)
        
    if frame_index % TRACK_EVERY_N_FRAMES == 0:
        print('Run object tracking. Frame: ', frame_index)
        result = model_det.track(frame, persist=True, verbose=False, stream=False)[0]

    # Run through each frame with prev result
    if result and result.boxes and result.boxes.is_track:
        boxes = result.boxes.xywh.cpu()
        track_ids = result.boxes.id.int().cpu().tolist()
        # frame = result.plot()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            center = (float(x), float(y))
            track.append(center)
            
            # ----------Start rep tracking----------
            if "rep_state" not in track_history:
                track_history["rep_state"] = {}

            if track_id not in track_history["rep_state"]:
                track_history["rep_state"][track_id] = {
                    "count": 0,
                    "state": "idle",    # can be: idle → up → down : 1rep → up → down : 2reps → up → down : 3reps
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
                        rep_info["cooldown"] = int(fps / 2)  # skip next few frames. ~0.5 s
                    rep_info["state"] = "down"
            # ----------End rep tracking----------
            
            if len(track) > MAX_TRACK_LENGTH:
                track.pop(0)

            # 1. Motion filter
            y_positions = [pt[1] for pt in track]
            if max(y_positions) - min(y_positions) < MOTION_THRESHOLD:
                continue

            # 2. Only accept box if its center is LOWER than nose
            accept = False
            nx, ny = nose_position   
            if y <= ny: # y : ycenter box
                continue 

            # 3. Draw track
            bx1, by1 = int(x - w / 2), int(y - h / 2)
            bx2, by2 = int(x + w / 2), int(y + h / 2)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            text = f"ID:{track_id} | Rep:{rep_info['count']}"
            cv2.putText(frame, text, (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 0), thickness=6)

    # Show frame
    cv2.imshow("YOLO Reps Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    out.write(frame)
    frame_index += 1

cap.release()
cv2.destroyAllWindows()
out.release()