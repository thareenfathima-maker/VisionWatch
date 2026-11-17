from ultralytics import YOLO
from pathlib import Path

# Locate the downloaded helmet model
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"

print("Loading helmet detection model...")

try:
    model = YOLO(str(MODEL_PATH))

    print("\nModel loaded successfully!")
    print("\nClasses recognized by this model:")

    for class_id, class_name in model.names.items():
        print(f"{class_id}: {class_name}")

except Exception as error:
    print("\nCould not load the model.")
    print("Error:", error)