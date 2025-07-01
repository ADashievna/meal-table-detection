import utils.bbox_utils as bbox_utils
import config
from typing import List, Tuple, Optional

BBoxValues = Tuple[float, float, float, float]
PlateEntry = Tuple[int, BBoxValues, str, float]

BBOX = "bbox"
LABEL = "label"
SCORE = "score"

def find_food_plate(food_bbox, plate_bboxes, img_h, img_w) -> Optional[PlateEntry]:
    if not plate_bboxes:
        return None
    candidates: List[PlateEntry] = [
        entry
        for entry in plate_bboxes
        if bbox_utils.is_inside(food_bbox, entry[1], img_h, img_w)
    ]
    if not candidates:
        return None
    # return the plate with the smallest area
    return min(candidates, key=lambda pair: pair[1][2] * pair[1][3])

def post_process_result(result):
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    scores = result.boxes.conf.cpu().numpy()
    names = result.names

    plates = []
    foods = []
    state_objects = []
    result_boxes = {}

    for i in range(len(boxes)):
        name = names[classes[i]]
        bbox = boxes[i]
        score = float(scores[i])

        if name in config.PLATE_CLASSES:
            plates.append((i, bbox, name, score))
        elif name in config.FOOD_CLASSES:
            foods.append((i, bbox, name, score))
        elif name in config.PLATE_STATE_CLASSES:
            state_objects.append((i, bbox, name, score))
        else:
            result_boxes[i] = {
                BBOX: bbox.tolist(),
                LABEL: name,
                SCORE: score
            }
    img_h, img_w = result.orig_shape

    for food_idx, food_box, food_name, food_score in foods:
        food_plate = find_food_plate(food_box, plates, img_h, img_w)
        if food_plate is not None:
            plate_idx, plate_bbox, _, _ = food_plate
            idx = plate_idx
            bbox = plate_bbox
        else:
            idx = food_idx
            bbox = food_box
        result_boxes[idx] = {
            BBOX: bbox.tolist(),
            LABEL: food_name,
            SCORE: food_score
        }

    for plate_idx, plate_box, plate_name, plate_score in plates:
        if plate_idx not in result_boxes:
            for state_idx, state_box, state_name, state_score in state_objects:
                if bbox_utils.is_inside(state_box, plate_box, img_h, img_w, config.PLATE_STATE_IOA_THRESHOLD):
                    plate_name = "empty " + plate_name
                    break
            result_boxes[plate_idx] = {
                BBOX: plate_box.tolist(),
                LABEL: plate_name,
                SCORE: plate_score
            }

    return result_boxes