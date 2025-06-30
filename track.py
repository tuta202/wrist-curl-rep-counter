from ultralytics import YOLO

model = YOLO("./runs/detect/train/weights/best.pt")

results = model.track("./wrist_curl_test.mp4", show=True, tracker="bytetrack.yaml")  # with ByteTrack