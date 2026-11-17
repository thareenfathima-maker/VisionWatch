import cv2
import csv
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
EVIDENCE_DIR = PROJECT_ROOT / "evidence"
EVENT_FILE = PROJECT_ROOT / "events.csv"

EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD HELMET DETECTION MODEL
# ==========================================

print("Loading VisionWatch helmet detection model...")

model = YOLO(str(MODEL_PATH))

print("Helmet model loaded successfully!")
print("Classes:", model.names)


# ==========================================
# 3. EVENT LOGGER
# ==========================================

def log_event(event_type, severity, message, confidence=None):
    """Append a safety event to the shared CSV file."""

    file_exists = EVENT_FILE.exists()

    event = {
        "timestamp": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "event_type": event_type,
        "severity": severity,
        "message": message,
        "confidence": round(confidence, 3)
        if confidence is not None else "",
    }

    with EVENT_FILE.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "timestamp",
                "event_type",
                "severity",
                "message",
                "confidence",
            ],
        )

        if not file_exists or EVENT_FILE.stat().st_size == 0:
            writer.writeheader()

        writer.writerow(event)

    print("EVENT RECORDED:", event)


# ==========================================
# 4. SAVE EVIDENCE IMAGE
# ==========================================

def save_evidence(frame):
    """Save a timestamped image of a detected violation."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    image_path = EVIDENCE_DIR / (
        f"helmet_violation_{timestamp}.jpg"
    )

    success = cv2.imwrite(str(image_path), frame)

    if success:
        print("Evidence saved:", image_path)
        return image_path

    print("WARNING: Could not save evidence image.")
    return None


# ==========================================
# 5. MAIN HELMET DETECTION
# ==========================================

def main():

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\nVisionWatch Factory Safety Monitoring Started")
    print("Press Q to exit.")

    # Prevent an event from being logged on every frame
    violation_active = False

    # Require several consecutive positive frames
    violation_frames = 0
    clear_frames = 0

    REQUIRED_VIOLATION_FRAMES = 3
    REQUIRED_CLEAR_FRAMES = 10

    try:
        while True:

            success, frame = camera.read()

            if not success:
                print("ERROR: Could not read webcam frame.")
                break

            frame = cv2.resize(frame, (640, 480))

            results = model.predict(
                source=frame,
                imgsz=416,
                conf=0.35,
                verbose=False,
            )

            annotated_frame = results[0].plot()

            no_hardhat_detected = False
            highest_confidence = 0.0

            # Examine the detected classes
            for box in results[0].boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                class_name = str(
                    model.names[class_id]
                ).lower().replace("-", "_")

                if class_name in (
                    "no_hardhat",
                    "no_hard_hat",
                    "no_helmet",
                    "nohelmet",
                ):
                    no_hardhat_detected = True

                    highest_confidence = max(
                        highest_confidence,
                        confidence,
                    )

            # ----------------------------------
            # HANDLE DETECTED VIOLATION
            # ----------------------------------

            if no_hardhat_detected:

                violation_frames += 1
                clear_frames = 0

                cv2.putText(
                    annotated_frame,
                    "WARNING: NO HARD HAT DETECTED!",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                )

                # Record only after consecutive detections
                if (
                    not violation_active
                    and violation_frames
                    >= REQUIRED_VIOLATION_FRAMES
                ):

                    violation_active = True

                    # Save evidence of the violation
                    evidence_path = save_evidence(
                        annotated_frame
                    )

                    # Record event for the dashboard
                    log_event(
                        event_type="helmet_violation",
                        severity="high",
                        message=(
                            "Worker detected without "
                            "an industrial safety hard hat"
                        ),
                        confidence=highest_confidence,
                    )

            else:

                violation_frames = 0
                clear_frames += 1

                # Allow a new event after the violation clears
                if clear_frames >= REQUIRED_CLEAR_FRAMES:
                    violation_active = False

            # ----------------------------------
            # DISPLAY STATUS
            # ----------------------------------

            status = (
                "VIOLATION DETECTED"
                if violation_active
                else "Monitoring Factory Safety"
            )

            cv2.putText(
                annotated_frame,
                "VisionWatch - Factory Safety",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            cv2.imshow(
                "VisionWatch - Helmet Detection",
                annotated_frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("VisionWatch monitoring stopped.")


if __name__ == "__main__":
    main()