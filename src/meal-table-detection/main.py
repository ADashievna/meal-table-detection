from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import plate_logic
import config

def _color_palette(n: int = 10) -> List[Tuple[int, int, int]]:
    cmap = plt.get_cmap("tab10")
    return  [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in cmap.colors]

_COLORS = _color_palette()
_COLOR_CACHE: Dict[str, Tuple[int, int, int]] = {}

def color_for(label: str) -> Tuple[int, int, int]:
    if label not in _COLOR_CACHE:
        _COLOR_CACHE[label] = _COLORS[len(_COLOR_CACHE) % len(_COLORS)]
    return _COLOR_CACHE[label]

def draw_overlay(frame, objs) -> None:
    for o in objs:                                 # bboxes
        x1, y1, x2, y2 = map(int, o["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_for(o["label"]), 15)

    for o in objs:                                 # labels
        x1, y1, *_ = map(int, o["bbox"])
        txt = f'{o["label"]} ({o["score"]:.2f})'
        cv2.putText(frame, txt, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 6)


def run(model_path: Path, video_path: Path, conf: float = 0.0, window_scale: float = 0.25) -> None:
    model = YOLO(model_path)

    for res in model.track(source=str(video_path), stream=True, conf=conf):
        frame = res.orig_img.copy()
        objs = list(plate_logic.post_process_result(res).values())

        draw_overlay(frame, objs)

        if window_scale != 1.0:
            frame = cv2.resize(frame, None, fx=window_scale, fy=window_scale)

        cv2.imshow("Result", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

def main() -> None:
    run(
        Path(config.MODEL_PATH),
        Path(config.VIDEO_PATH),
        0.1,
        0.2,
    )

if __name__ == "__main__":
    main()