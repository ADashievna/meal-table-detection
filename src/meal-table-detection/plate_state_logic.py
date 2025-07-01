import utils.bbox_utils as bbox_utils
import config
from typing import Optional
from structures import *
from collections import defaultdict

def find_food_plate(food_bbox: BBox, plates: List[TableItem]) -> Optional[TableItem]:
    candidates = [
        plate
        for plate in plates
        if bbox_utils.is_inside(
            food_bbox.as_xyxy(),
            plate.bbox.as_xyxy()
        )
    ]
    if not candidates:
        return None
    # return the plate with the smallest area
    return min(candidates, key=lambda p: p.bbox.area)


def classify_detections(result) -> Dict[Category, List[TableItem]]:
    bboxes       = result.boxes.xywh.cpu().numpy()
    classes   = result.boxes.cls.cpu().numpy().astype(int)
    confidences       = result.boxes.conf.cpu().numpy()
    names       = result.names

    buckets: Dict[Category, List[TableItem]] = defaultdict(list)

    for idx, (bbox, class_id, conf) in enumerate(zip(bboxes, classes, confidences)):
        label = names[class_id]
        score = float(conf)

        if label in config.PLATE_CLASSES:
            category = Category.PLATE
        elif label in config.FOOD_CLASSES:
            category = Category.FOOD
        elif label in config.PLATE_STATE_CLASSES:
            category = Category.STATE_OBJECT
        else:
            category = Category.OTHER

        cx, cy, w, h = bbox
        bbox_obj = BBox(cx, cy, w, h)
        buckets[category].append(TableItem(idx, bbox_obj, label, score))

    return buckets


def form_updated_items(buckets: Dict[Category, List[TableItem]]) -> List[TableItem]:
    plates = buckets[Category.PLATE]
    foods = buckets[Category.FOOD]
    state_objects = buckets[Category.STATE_OBJECT]
    others = buckets.get(Category.OTHER, [])

    result_map: Dict[int, TableItem] = {it.idx: it for it in others}

    for food in foods:
        matched_plate = find_food_plate(food.bbox, plates)
        if matched_plate is not None:
            idx = matched_plate.idx
            bbox = matched_plate.bbox
        else:
            idx = food.idx
            bbox = food.bbox
        result_map[idx] = TableItem(idx, bbox, food.label, food.score)

    for plate in plates:
        if plate.idx in result_map:
            continue
        is_empty = any(
            bbox_utils.is_inside(
                state_object.bbox.as_xyxy(),
                plate.bbox.as_xyxy(),
                config.PLATE_STATE_IOA_THRESHOLD,
            )
            for state_object in state_objects
        )

        label = f"empty {plate.label}" if is_empty else plate.label
        result_map[plate.idx] = TableItem(plate.idx, plate.bbox, label, plate.score)

    return [result_map[k] for k in sorted(result_map)]

def refine_detections(result):
    buckets = classify_detections(result)
    return form_updated_items(buckets)