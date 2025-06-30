import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.train(data="data.yaml", epochs=100, imgsz=640)