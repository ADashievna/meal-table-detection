
def box_ioa(inner_box, outer_box):
    if len(inner_box) != 4 or len(outer_box) != 4:
        raise ValueError("Both boxes must be in format [x1, y1, x2, y2].")

    # Calculate intersection coordinates
    x_a = max(inner_box[0], outer_box[0])
    y_a = max(inner_box[1], outer_box[1])
    x_b = min(inner_box[2], outer_box[2])
    y_b = min(inner_box[3], outer_box[3])

    intersection = max(0, x_b - x_a) * max(0, y_b - y_a)
    if intersection == 0:
        return 0.0

    inner_box_area = (inner_box[2] - inner_box[0]) * (inner_box[3] - inner_box[1])

    iou = intersection / float(inner_box_area)
    return iou

def is_inside(inner_box, outer_box, img_w, img_h, threshold=0.7):
    inner_box_xyxy = yolo_to_xyxy(inner_box, img_w, img_h)
    outer_box_xyxy = yolo_to_xyxy(outer_box, img_w, img_h)
    iou = box_ioa(inner_box_xyxy, outer_box_xyxy)
    return iou >= threshold

def yolo_to_xyxy(box, img_w, img_h):
    cx, cy, w, h = box
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return [x1, y1, x2, y2]

