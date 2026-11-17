import cv2
from detector import VisionDetector


def main():
    detector = VisionDetector()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Could not open webcam.")
        return

    print("VisionWatch person detection started.")
    print("Press Q to exit.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read webcam frame.")
                break

            frame = cv2.resize(frame, (640, 480))

            # Run AI detection
            output = detector.detect(frame)

            cv2.imshow("VisionWatch - Person Detection", output)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()