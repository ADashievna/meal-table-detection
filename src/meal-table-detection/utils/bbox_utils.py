
IOA_THRESHOLD = 0.7

def calc_ioa_two_boxes(inner_box, outer_box):
    left_bottom_x = max(inner_box[0], outer_box[0])
    left_bottom_y = max(inner_box[1], outer_box[1])
    right_top_x = min(inner_box[2], outer_box[2])
    right_top_y = min(inner_box[3], outer_box[3])

    intersection = max(0, right_top_x - left_bottom_x) * max(0, right_top_y - left_bottom_y)
    if intersection == 0:
        return 0.0

    inner_box_area = (inner_box[2] - inner_box[0]) * (inner_box[3] - inner_box[1])

    ioa = intersection / float(inner_box_area)
    return ioa

def is_inside(inner_box, outer_box, threshold=IOA_THRESHOLD):
    ioa = calc_ioa_two_boxes(inner_box, outer_box)
    return ioa >= threshold

