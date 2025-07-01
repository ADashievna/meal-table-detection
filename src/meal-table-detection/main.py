from pathlib import Path
import cv2
from ultralytics.utils.plotting import Colors
from ultralytics import YOLO
import plate_state_logic
import config
from structures import *

TEXT_FONT = cv2.FONT_HERSHEY_PLAIN
TEXT_FONT_SCALE = 5
TEXT_FONT_THICKNESS = 8
LABEL_OFFSET = 8
BBOX_THICKNESS = 13
WINDOW_NAME = "Result"
WHITE_COLOR = (255, 255, 255)

colors = Colors()
colors_cache: Dict[str, int] = {}

def color_for(label: str) -> tuple:
    if label not in colors_cache:
        colors_cache[label] = len(colors_cache)
    return colors(colors_cache[label], bgr=True)

def draw_overlay(frame, items_to_draw: list[TableItem]) -> None:
    for item in items_to_draw:
        x1, y1, x2, y2 = map(int, item.bbox.as_xyxy())
        cv2.rectangle(frame, (x1, y1), (x2, y2),
                      color_for(item.label), BBOX_THICKNESS)

    for item in items_to_draw:
        x1, y1, _, _ = map(int, item.bbox.as_xyxy())
        txt = f"{item.label} ({item.score:.2f})"
        cv2.putText(frame, txt, (x1, y1 - LABEL_OFFSET),
                    TEXT_FONT, TEXT_FONT_SCALE,
                    WHITE_COLOR, TEXT_FONT_THICKNESS)


def run(model_path: Path, video_path: Path, min_confidence: float = 0.0) -> None:
    model = YOLO(model_path)

    for model_result in model.track(source=str(video_path), stream=True, conf=min_confidence):
        current_frame = model_result.orig_img.copy()
        refined_items = plate_state_logic.refine_detections(model_result)

        draw_overlay(current_frame, refined_items)
        current_frame = cv2.resize(current_frame, (config.DESIRED_WINDOWS_HEIGHT, config.DESIRED_WINDOWS_WIDTH),
                                   interpolation=cv2.INTER_AREA)
        cv2.imshow(WINDOW_NAME, current_frame)

        cv2.waitKey(1)
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()

def main() -> None:
    run(
        Path(config.MODEL_PATH),
        Path(config.VIDEO_PATH),
        config.MIN_CONFIDENCE_LEVEL, # Confidence threshold for detection
    )

if __name__ == "__main__":
    main()