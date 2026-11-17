import cv2


def main():
    # Open the default webcam
    camera = cv2.VideoCapture(0)

    # Check whether the camera opened successfully
    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        print("Check your camera permissions.")
        return

    print("VisionWatch camera started.")
    print("Press Q to quit.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("ERROR: Could not read camera frame.")
                break

            # Resize the frame for smoother processing
            frame = cv2.resize(frame, (640, 480))

            # Display project name
            cv2.putText(
                frame,
                "VisionWatch - Live Camera",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # Display the webcam feed
            cv2.imshow("VisionWatch", frame)

            # Press Q to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()