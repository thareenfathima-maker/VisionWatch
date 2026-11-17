import cv2
from detector import VisionDetector
from restricted_zone import is_inside_zone
from event_logger import log_event


def main():
    # Load the YOLO person detection model
    detector = VisionDetector()

    # Open webcam
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Could not open webcam.")
        return

    # Resize camera frames
    frame_width = 640
    frame_height = 480

    # Restricted zone coordinates: x1, y1, x2, y2
    zone = (200, 100, 500, 400)

    # Track whether someone was already inside the zone
    person_was_inside = False

    print("VisionWatch restricted-zone monitoring started.")
    print("Nearby and distant people can trigger alerts.")
    print("Press Q to exit.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read camera frame.")
                break

            # Resize the frame
            frame = cv2.resize(
                frame,
                (frame_width, frame_height)
            )

            # Detect people using YOLO
            results = detector.model.predict(
                source=frame,
                imgsz=416,
                conf=0.4,
                classes=[0],
                verbose=False
            )

            # Create the displayed frame
            annotated = frame.copy()

            x1, y1, x2, y2 = zone

            # Draw restricted zone
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

            cv2.putText(
                annotated,
                "RESTRICTED ZONE",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

            person_inside = False

            # Check every detected person
            for box in results[0].boxes:

                # Get bounding box coordinates
                px1, py1, px2, py2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                # Approximate the person's foot position
                point = (
                    (px1 + px2) // 2,
                    py2
                )

                # Draw bounding box around the person
                cv2.rectangle(
                    annotated,
                    (px1, py1),
                    (px2, py2),
                    (0, 255, 0),
                    2
                )

                # Check whether the person is inside the zone
                if is_inside_zone(point, zone):

                    person_inside = True

                    # Highlight the person violating the zone
                    cv2.rectangle(
                        annotated,
                        (px1, py1),
                        (px2, py2),
                        (0, 0, 255),
                        3
                    )

                    cv2.putText(
                        annotated,
                        "ZONE VIOLATION!",
                        (px1, max(py1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

            # Log an event when someone enters the zone
            if person_inside and not person_was_inside:

                log_event(
                    event_type="restricted_zone_entry",
                    severity="high",
                    message="Person entered the restricted zone"
                )

                print("ALERT: Restricted zone entry detected!")

            # Update the previous state
            person_was_inside = person_inside

            # Display project title
            cv2.putText(
                annotated,
                "VisionWatch - Safety Monitoring",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            # Display the camera feed
            cv2.imshow(
                "VisionWatch - Restricted Zone Detection",
                annotated
            )

            # Press Q to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()