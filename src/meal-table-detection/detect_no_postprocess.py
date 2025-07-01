import cv2
from ultralytics import YOLO
import config

def run_demo_no_postprocess(model_path, video_path, scale: float = 1):
    model = YOLO(model_path)

    for res in model.track(source=str(video_path), stream=True):
        frame = res.plot()
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
        cv2.imshow("Result", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()

def main() -> None:
    run_demo_no_postprocess(
        model_path=config.MODEL_PATH,
        video_path=config.VIDEO_PATH,
        scale=0.2,
    )

if __name__ == "__main__":
    main()