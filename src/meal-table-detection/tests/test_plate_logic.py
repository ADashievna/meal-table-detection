from types import ModuleType
import numpy as np
from mockito import when, unstub, ANY, verify
import sys

bbox_utils = ModuleType("utils.bbox_utils")
def always_inside(*_, **__):
    return True
bbox_utils.is_inside = always_inside

config = ModuleType("config")
config.PLATE_CLASSES        = ["plate"]
config.FOOD_CLASSES         = ["steak"]
config.PLATE_STATE_CLASSES  = ["state"]
config.PLATE_STATE_IOA_THRESHOLD = 0.0

sys.modules["utils.bbox_utils"] = bbox_utils
sys.modules["config"]           = config

import importlib
plate_logic = importlib.import_module("plate_logic")

class _BoxAttr:
    def __init__(self, data):
        self._arr = np.asarray(data)
    def cpu(self):
        return self
    def numpy(self):
        return self._arr

class DummyBoxes:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = _BoxAttr(xyxy)
        self.cls  = _BoxAttr(cls)
        self.conf = _BoxAttr(conf)

class DummyResult:
    def __init__(self, xyxy, cls, conf, names, shape):
        self.boxes = DummyBoxes(xyxy, cls, conf)
        self.names = names
        self.orig_shape = shape


def test_find_food_plate_picks_smallest():
    food  = (10, 10, 40, 40)
    big   = (0, 0, 100, 100)
    small = (5, 5, 60, 60)
    plates = [(0, big, "plate", 0.9), (1, small, "plate", 0.8)]

    when(bbox_utils).is_inside(food, big, 200, 200).thenReturn(True)
    when(bbox_utils).is_inside(food, small, 200, 200).thenReturn(True)

    chosen = plate_logic.find_food_plate(food, plates, 200, 200)
    assert chosen[1] == small
    unstub()


def test_find_food_plate_returns_none_if_not_inside():
    food  = (10, 10, 40, 40)
    plate = (0, 0, 100, 100)
    when(bbox_utils).is_inside(food, plate, ANY, ANY).thenReturn(False)

    assert plate_logic.find_food_plate(food, [(0, plate, "plate", .9)], 200, 200) is None
    unstub()


def test_find_food_plate_none_for_empty_plate_list():
    assert plate_logic.find_food_plate((1,1,10,10), [], 200, 200) is None

def _run(res):
    return plate_logic.post_process_result(res)

def test_food_inside_plate_borrow_plate_bbox():
    plate_box = (0, 0, 100, 100)
    food_box = (25, 25, 75, 75)
    xyxy = [plate_box, food_box]
    cls = [0, 1]
    conf = [.9, .8]
    names = {0: "plate", 1: "steak"}

    when(bbox_utils).is_inside(ANY, ANY, ANY, ANY).thenReturn(True)

    res = DummyResult(xyxy, cls, conf, names, (200, 200))
    out = _run(res)

    assert len(out) == 1
    rec = out[0]
    assert rec["label"] == "steak"
    assert rec["bbox"] == list(plate_box)

    unstub()


def test_food_without_plate_keeps_own_bbox():
    food_box = (25, 25, 75, 75)
    xyxy = [food_box]
    cls  = [1]        # steak
    conf = [.8]
    names = {1: "steak"}

    when(bbox_utils).is_inside(...).thenReturn(False)

    out = _run(DummyResult(xyxy, cls, conf, names, (200, 200)))
    rec = out[0]
    assert rec["label"] == "steak"
    assert rec["bbox"]  == list(food_box)
    unstub()


def test_empty_plate_prefix_added():
    plate_box = (0, 0, 100, 100)
    state_box = (10, 10, 20, 20)
    xyxy = [plate_box, state_box]
    cls  = [0, 2]               # plate, state
    conf = [.9, .7]
    names = {0: "plate", 2: "state"}

    when(bbox_utils).is_inside(...).thenReturn(True)

    out = _run(DummyResult(xyxy, cls, conf, names, (200, 200)))
    rec = out[0]
    assert rec["label"] == "empty plate"
    unstub()
