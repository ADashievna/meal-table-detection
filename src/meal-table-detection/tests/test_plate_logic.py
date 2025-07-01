import sys
import types
import numpy as np
from mockito import when, verify, unstub, ANY
from structures import BBox, TableItem, Category
import utils.bbox_utils as bbox_utils

config = types.ModuleType("config")
config.PLATE_CLASSES            = {"plate"}
config.FOOD_CLASSES             = {"food"}
config.PLATE_STATE_CLASSES      = {"state"}
config.PLATE_STATE_IOA_THRESHOLD = 0.7
sys.modules["config"] = config

import plate_state_logic

class _Arr:
    def __init__(self, data): self._a = np.asarray(data, dtype=float)
    def cpu(self): return self
    def numpy(self): return self._a

class _Boxes:
    def __init__(self, xywh, cls, conf):
        self.xywh = _Arr(xywh); self.cls = _Arr(cls); self.conf = _Arr(conf)

class DummyResult:
    def __init__(self, xywh, cls, conf, names):
        self.boxes = _Boxes(xywh, cls, conf); self.names = names

def bb(cx, cy, w, h):
    return BBox(cx, cy, w, h)

def test_find_food_plate_returns_smallest_plate():
    plate_big   = TableItem(0, bb(50, 50, 100, 100), "plate", 0.95)
    plate_small = TableItem(1, bb(55, 55, 40,  40),  "plate", 0.93)
    food_bbox   = bb(55, 55, 10, 10)

    when(bbox_utils).is_inside(food_bbox.as_xyxy(), plate_big.bbox.as_xyxy()).thenReturn(True)
    when(bbox_utils).is_inside(food_bbox.as_xyxy(), plate_small.bbox.as_xyxy()).thenReturn(True)

    try:
        chosen = plate_state_logic.find_food_plate(food_bbox, [plate_big, plate_small])
        assert chosen is plate_small
        verify(bbox_utils, times=2).is_inside(ANY, ANY)
    finally:
        unstub()


def test_find_food_plate_returns_none_if_not_inside():
    plate = TableItem(0, bb(50, 50, 40, 40), "plate", 0.9)
    food_bbox = bb(150, 150, 10, 10)

    when(bbox_utils).is_inside(...).thenReturn(False)

    try:
        assert plate_state_logic.find_food_plate(food_bbox, [plate]) is None
        verify(bbox_utils).is_inside(food_bbox.as_xyxy(), plate.bbox.as_xyxy())
    finally:
        unstub()


def test_classify_detections_places_boxes_in_buckets():
    xywh  = [[50, 50, 100, 100],  [55, 55, 10, 10],  [150, 150, 20, 20]]
    cls   = [0, 1, 2]
    conf  = [0.9, 0.8, 0.75]
    names = {0: "plate", 1: "food", 2: "state"}

    buckets = plate_state_logic.classify_detections(DummyResult(xywh, cls, conf, names))

    assert len(buckets[Category.PLATE])        == 1
    assert len(buckets[Category.FOOD])         == 1
    assert len(buckets[Category.STATE_OBJECT]) == 1


def test_form_updated_items_replaces_plate_with_food():
    plate = TableItem(0, bb(50, 50, 100, 100), "plate", 0.9)
    food  = TableItem(1, bb(50, 50, 10, 10),   "food",  0.8)

    when(bbox_utils).is_inside(food.bbox.as_xyxy(), plate.bbox.as_xyxy()).thenReturn(True)

    buckets = {
        Category.PLATE:  [plate],
        Category.FOOD:   [food],
        Category.STATE_OBJECT: [],
        Category.OTHER:  []
    }

    try:
        updated = plate_state_logic.form_updated_items(buckets)
        assert len(updated) == 1
        assert updated[0].idx   == 0
        assert updated[0].label == "food"
    finally:
        unstub()


def test_form_updated_items_marks_empty_plate():
    plate = TableItem(0, bb(50, 50, 100, 100), "plate", 0.9)
    state = TableItem(2, bb(50, 50, 20,  20),  "state", 0.7)

    when(bbox_utils).is_inside(state.bbox.as_xyxy(),
                               plate.bbox.as_xyxy(),
                               config.PLATE_STATE_IOA_THRESHOLD).thenReturn(True)

    buckets = {
        Category.PLATE:  [plate],
        Category.FOOD:   [],
        Category.STATE_OBJECT: [state],
        Category.OTHER:  []
    }

    try:
        updated = plate_state_logic.form_updated_items(buckets)
        assert updated[0].label == "empty plate"
    finally:
        unstub()

