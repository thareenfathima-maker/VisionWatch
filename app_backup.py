import cv2
import csv
import time
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from ultralytics import YOLO


# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
EVIDENCE_DIR = PROJECT_ROOT / "evidence"
EVENT_FILE = PROJECT_ROOT / "events.csv"

EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

CSV_COLUMNS = [
    "timestamp",
    "event_type",
    "severity",
    "message",
    "confidence",
]


# =====================================================
# 2. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="VisionWatch | Factory Safety",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ VisionWatch")
st.caption("AI-Powered Factory Safety Monitoring System")


# =====================================================
# 3. LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Helmet model not found: {MODEL_PATH}"
        )

    return YOLO(str(MODEL_PATH))


try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the helmet model: {error}")
    st.stop()


# =====================================================
# 4. EVENT LOGGER
# =====================================================

def log_event(event_type, severity, message, confidence=None):

    event = {
        "timestamp": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "event_type": event_type,
        "severity": severity,
        "message": message,
        "confidence": (
            round(float(confidence), 3)
            if confidence is not None
            else ""
        ),
    }

    file_exists = EVENT_FILE.exists()

    with EVENT_FILE.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=CSV_COLUMNS,
        )

        if not file_exists or EVENT_FILE.stat().st_size == 0:
            writer.writeheader()

        writer.writerow(event)


# =====================================================
# 5. EVIDENCE SAVING
# =====================================================

def save_evidence(frame, event_type):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    image_path = EVIDENCE_DIR / (
        f"{event_type}_{timestamp}.jpg"
    )

    saved = cv2.imwrite(str(image_path), frame)

    return image_path if saved else None


# =====================================================
# 6. DASHBOARD CONTROLS
# =====================================================

st.sidebar.header("⚙️ Monitoring Controls")

camera_index = st.sidebar.selectbox(
    "Camera",
    [0, 1, 2],
    index=0,
)

enable_helmet = st.sidebar.checkbox(
    "Enable helmet detection",
    value=True,
)

enable_zone = st.sidebar.checkbox(
    "Enable restricted-zone detection",
    value=True,
)

st.sidebar.subheader("Restricted Zone")

zone_x1 = st.sidebar.slider(
    "Zone left",
    0,
    639,
    200,
)

zone_y1 = st.sidebar.slider(
    "Zone top",
    0,
    479,
    100,
)

zone_x2 = st.sidebar.slider(
    "Zone right",
    1,
    640,
    500,
)

zone_y2 = st.sidebar.slider(
    "Zone bottom",
    1,
    480,
    400,
)

helmet_confidence = st.sidebar.slider(
    "Helmet detection confidence",
    0.10,
    0.90,
    0.35,
    0.05,
)

start_monitoring = st.sidebar.toggle(
    "Start live monitoring",
    value=False,
)

st.sidebar.info(
    "Press Stop monitoring before changing camera settings."
)


# =====================================================
# 7. EVENT DATA HELPERS
# =====================================================

def read_events():

    if not EVENT_FILE.exists():
        return pd.DataFrame(columns=CSV_COLUMNS)

    try:
        df = pd.read_csv(EVENT_FILE)

        for column in CSV_COLUMNS:
            if column not in df.columns:
                df[column] = ""

        return df[CSV_COLUMNS]

    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=CSV_COLUMNS)


def show_dashboard_metrics(df):

    total = len(df)

    helmet_count = (
        df["event_type"] == "helmet_violation"
    ).sum()

    zone_count = (
        df["event_type"] == "restricted_zone_entry"
    ).sum()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Safety Events", int(total))
    col2.metric("Helmet Violations", int(helmet_count))
    col3.metric("Restricted-Zone Entries", int(zone_count))


# =====================================================
# 8. LIVE MONITORING
# =====================================================

st.subheader("📹 Live Factory Camera")

camera_placeholder = st.empty()
status_placeholder = st.empty()

if start_monitoring:

    if zone_x1 >= zone_x2 or zone_y1 >= zone_y2:
        st.error(
            "Invalid restricted zone. "
            "Right must be greater than left, and "
            "bottom must be greater than top."
        )
        st.stop()

    camera = cv2.VideoCapture(camera_index)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not camera.isOpened():

        st.error(
            "Could not open the selected camera. "
            "Close other programs using the webcam and try again."
        )

    else:

        status_placeholder.success("Monitoring is running.")

        # Incident state prevents one event per frame.
        helmet_violation_active = False
        zone_violation_active = False

        helmet_positive_frames = 0
        helmet_clear_frames = 0

        zone_positive_frames = 0
        zone_clear_frames = 0

        REQUIRED_POSITIVE_FRAMES = 3
        REQUIRED_CLEAR_FRAMES = 10

        frame_counter = 0
        latest_annotated = None

        try:

            while True:

                success, frame = camera.read()

                if not success:
                    status_placeholder.error(
                        "Unable to read from the camera."
                    )
                    break

                frame = cv2.resize(frame, (640, 480))

                # -----------------------------------------
                # A. HELMET DETECTION
                # -----------------------------------------

                results = None

                if enable_helmet:
                    results = model.predict(
                        source=frame,
                        imgsz=416,
                        conf=helmet_confidence,
                        verbose=False,
                    )

                    annotated = results[0].plot()
                else:
                    annotated = frame.copy()

                no_hardhat_detected = False
                highest_helmet_confidence = 0.0

                if results is not None:

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

                            highest_helmet_confidence = max(
                                highest_helmet_confidence,
                                confidence,
                            )

                if enable_helmet and no_hardhat_detected:

                    helmet_positive_frames += 1
                    helmet_clear_frames = 0

                    if (
                        not helmet_violation_active
                        and helmet_positive_frames
                        >= REQUIRED_POSITIVE_FRAMES
                    ):

                        helmet_violation_active = True

                        evidence_frame = annotated.copy()

                        evidence_path = save_evidence(
                            evidence_frame,
                            "helmet_violation",
                        )

                        log_event(
                            event_type="helmet_violation",
                            severity="high",
                            message=(
                                "Worker detected without "
                                "an industrial safety hard hat"
                            ),
                            confidence=highest_helmet_confidence,
                        )

                else:

                    helmet_positive_frames = 0
                    helmet_clear_frames += 1

                    if (
                        helmet_clear_frames
                        >= REQUIRED_CLEAR_FRAMES
                    ):
                        helmet_violation_active = False

                # -----------------------------------------
                # B. RESTRICTED-ZONE DETECTION
                # -----------------------------------------

                person_inside_zone = False

                if enable_zone:

                    cv2.rectangle(
                        annotated,
                        (zone_x1, zone_y1),
                        (zone_x2, zone_y2),
                        (0, 0, 255),
                        2,
                    )

                    cv2.putText(
                        annotated,
                        "RESTRICTED ZONE",
                        (
                            zone_x1,
                            max(zone_y1 - 10, 20),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        2,
                    )

                    # The existing helmet model detects hardhats,
                    # not full people. Use the existing person
                    # detector for restricted-zone monitoring.
                    person_results = None

                    if not hasattr(st.session_state, "person_model"):
                        try:
                            person_model_path = (
                                PROJECT_ROOT / "yolo11n.pt"
                            )

                            if person_model_path.exists():
                                st.session_state.person_model = YOLO(
                                    str(person_model_path)
                                )
                            else:
                                st.session_state.person_model = None

                        except Exception:
                            st.session_state.person_model = None

                    person_model = st.session_state.person_model

                    if person_model is not None:

                        person_results = person_model.predict(
                            source=frame,
                            imgsz=416,
                            conf=0.4,
                            classes=[0],
                            verbose=False,
                        )

                    if person_results is not None:

                        for box in person_results[0].boxes:

                            px1, py1, px2, py2 = map(
                                int,
                                box.xyxy[0].tolist(),
                            )

                            foot_x = (px1 + px2) // 2
                            foot_y = py2

                            inside = (
                                zone_x1 <= foot_x <= zone_x2
                                and zone_y1 <= foot_y <= zone_y2
                            )

                            if inside:

                                person_inside_zone = True

                                cv2.rectangle(
                                    annotated,
                                    (px1, py1),
                                    (px2, py2),
                                    (0, 0, 255),
                                    2,
                                )

                                cv2.putText(
                                    annotated,
                                    "ZONE VIOLATION",
                                    (
                                        px1,
                                        max(py1 - 10, 20),
                                    ),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55,
                                    (0, 0, 255),
                                    2,
                                )

                    if person_inside_zone:

                        zone_positive_frames += 1
                        zone_clear_frames = 0

                        if (
                            not zone_violation_active
                            and zone_positive_frames
                            >= REQUIRED_POSITIVE_FRAMES
                        ):

                            zone_violation_active = True

                            save_evidence(
                                annotated,
                                "restricted_zone_entry",
                            )

                            log_event(
                                event_type="restricted_zone_entry",
                                severity="high",
                                message=(
                                    "Person entered the "
                                    "restricted zone"
                                ),
                            )

                    else:

                        zone_positive_frames = 0
                        zone_clear_frames += 1

                        if (
                            zone_clear_frames
                            >= REQUIRED_CLEAR_FRAMES
                        ):
                            zone_violation_active = False

                else:

                    zone_positive_frames = 0
                    zone_clear_frames = 0
                    zone_violation_active = False

                # -----------------------------------------
                # C. OVERALL STATUS
                # -----------------------------------------

                if helmet_violation_active or zone_violation_active:

                    cv2.putText(
                        annotated,
                        "SAFETY VIOLATION DETECTED",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 255),
                        2,
                    )

                else:

                    cv2.putText(
                        annotated,
                        "FACTORY SAFETY MONITORING",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                    )

                # Convert OpenCV BGR to RGB for Streamlit.
                rgb_frame = cv2.cvtColor(
                    annotated,
                    cv2.COLOR_BGR2RGB,
                )

                camera_placeholder.image(
                    rgb_frame,
                    channels="RGB",
                    use_container_width=True,
                )

                # Avoid excessive CPU usage and UI updates.
                time.sleep(0.03)

        finally:

            camera.release()

            status_placeholder.info(
                "Camera released. Rerun the app to start again."
            )

else:

    st.info(
        "Use the sidebar to adjust the restricted zone, "
        "then enable Start live monitoring."
    )


# =====================================================
# 9. EVENT DASHBOARD
# =====================================================

st.divider()

st.subheader("📊 Safety Statistics")

events_df = read_events()

show_dashboard_metrics(events_df)

st.subheader("📝 Recent Safety Events")

if events_df.empty:

    st.info("No safety events recorded yet.")

else:

    st.dataframe(
        events_df.tail(100).iloc[::-1],
        use_container_width=True,
        hide_index=True,
    )


# =====================================================
# 10. EVIDENCE GALLERY
# =====================================================

st.divider()

st.subheader("📸 Evidence Gallery")

evidence_images = sorted(
    EVIDENCE_DIR.glob("*.jpg"),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)

if not evidence_images:

    st.info("No evidence images have been saved yet.")

else:

    columns = st.columns(3)

    for index, image_path in enumerate(
        evidence_images[:9]
    ):

        with columns[index % 3]:

            st.image(
                str(image_path),
                caption=image_path.name,
                use_container_width=True,
            )