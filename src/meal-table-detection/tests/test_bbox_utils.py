from utils.bbox_utils import box_ioa
from utils.bbox_utils import yolo_to_xyxy
from utils.bbox_utils import is_inside

def test_box_iou_perfect_overlap():
    box1 = [10, 10, 20, 20]
    box2 = [10, 10, 20, 20]
    assert box_ioa(box1, box2) == 1.0

def test_box_iou_no_overlap():
    box1 = [0, 0, 10, 10]
    box2 = [20, 20, 30, 30]
    assert box_ioa(box1, box2) == 0.0

def test_box_iou_partial_overlap():
    box1 = [0, 0, 10, 10]
    box2 = [5, 5, 15, 15]
    iou = box_ioa(box1, box2)
    assert 0 < iou < 1

def test_centered_box():
    box = [0.5, 0.5, 0.5, 0.5]  # cx, cy, w, h
    img_w, img_h = 100, 100
    assert yolo_to_xyxy(box, img_w, img_h) == [25, 25, 75, 75]

def test_full_image_box():
    box = [0.5, 0.5, 1.0, 1.0]
    img_w, img_h = 200, 300
    assert yolo_to_xyxy(box, img_w, img_h) == [0, 0, 200, 300]

def test_top_left_corner():
    box = [0.25, 0.25, 0.5, 0.5]
    img_w, img_h = 100, 100
    assert yolo_to_xyxy(box, img_w, img_h) == [0, 0, 50, 50]

def test_bottom_right_corner():
    box = [0.75, 0.75, 0.5, 0.5]
    img_w, img_h = 100, 100
    assert yolo_to_xyxy(box, img_w, img_h) == [50, 50, 100, 100]

def test_single_point_box():
    box = [0.5, 0.5, 0.0, 0.0]
    img_w, img_h = 100, 100
    assert yolo_to_xyxy(box, img_w, img_h) == [50, 50, 50, 50]

def test_perfectly_inside():
    outer_box = [0.26171875, 0.440625, 0.3359375, 0.20078125]
    inner_box = [0.26953125, 0.44296875, 0.13984375, 0.078125]
    img_w, img_h = 640, 640
    assert is_inside(inner_box, outer_box, img_w, img_h, threshold=0.5)

def test_not_inside():
    inner_box = [0.1, 0.1, 0.1, 0.1]    # 5,5,15,15
    outer_box = [0.9, 0.9, 0.1, 0.1]    # 85,85,95,95
    img_w, img_h = 100, 100
    assert not is_inside(inner_box, outer_box, img_w, img_h, threshold=0.1)

def test_partially_inside():
    inner_box = [0.7, 0.7, 0.4, 0.4]
    outer_box = [0.5, 0.5, 0.6, 0.6]
    img_w, img_h = 100, 100
    assert is_inside(inner_box, outer_box, img_w, img_h, threshold=0.6)
    assert not is_inside(inner_box, outer_box, img_w, img_h, threshold=0.9)

def test_identical_boxes():
    box = [0.3, 0.3, 0.2, 0.2]
    img_w, img_h = 100, 100
    assert is_inside(box, box, img_w, img_h, threshold=0.9)
    assert is_inside(box, box, img_w, img_h, threshold=1.0)

def test_threshold_edge_case():
    inner_box = [0.5, 0.5, 0.4, 0.4]
    outer_box = [0.5, 0.5, 0.4, 0.4]
    img_w, img_h = 100, 100
    assert is_inside(inner_box, outer_box, img_w, img_h, threshold=1.0)