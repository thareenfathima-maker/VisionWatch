from ultralytics import YOLO


class VisionDetector:
    def __init__(self):
        print("Loading YOLO model...")

        # Downloads a small pretrained model on first use
        self.model = YOLO("yolo11n.pt")

        print("YOLO model loaded successfully!")

    def detect(self, frame):
        results = self.model.predict(
            source=frame,
            imgsz=416,
            conf=0.4,
            classes=[0],  # Person class only
            verbose=False
        )

        return results[0].plot()