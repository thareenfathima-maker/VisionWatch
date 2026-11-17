import csv
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = PROJECT_ROOT / "evidence"
EVENT_FILE = PROJECT_ROOT / "events.csv"


def log_event(event_type, severity, message, confidence=None):
    """
    Save a VisionWatch event to events.csv.
    """

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    file_exists = EVENT_FILE.exists()

    event = {
        "timestamp": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "event_type": event_type,
        "severity": severity,
        "message": message,
        "confidence": confidence,
    }

    with EVENT_FILE.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=event.keys()
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(event)

    print("Event recorded:", event)