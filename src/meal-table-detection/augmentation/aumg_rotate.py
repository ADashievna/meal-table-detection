from pathlib import Path
import logging

import cv2
import albumentations as A
from tqdm import tqdm

SPLITS = ("train", "val")
IMAGES = "images"
LABELS = "labels"
LOGGER = logging.getLogger(__name__)
EPS = 1e-5

def rot_aug(angle: int) -> A.Compose:
    return A.Compose(
        [A.Rotate(limit=(angle, angle), p=1)],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"],
        check_each_transform=False),
    p=1,
    )

def _is_valid(box: list[float], eps: float(1e-5)) -> bool:
    return all(0.0 - eps <= v <= 1.0 + eps for v in box)

def _filter_boxes(classes: list[int], boxes: list[list[float]]):
    good_cls, good_boxes = [], []
    for cls, box in zip(classes, boxes):
        if _is_valid(box, EPS):
            good_cls.append(cls)
            good_boxes.append(box)
        else:
            LOGGER.warning("Bbox is out of range [0.0, 1.0]: %s, clipping it", box)
            clipped = [min(1.0, max(0.0, v)) for v in box]
            good_cls.append(cls)
            good_boxes.append(clipped)
    return good_cls, good_boxes

def read_sample(img_p: Path, lbl_p: Path):
    img = cv2.imread(str(img_p))
    classes, boxes = [], []
    if lbl_p.exists():
        with open(lbl_p) as f:
            for line in f:
                if line.strip():
                    parts = line.split()
                    classes.append(int(parts[0]))
                    boxes.append(list(map(float, parts[1:])))
    classes, boxes = _filter_boxes(classes, boxes)
    return img, classes, boxes

def write_sample(img, classes, boxes, img_p: Path, lbl_p: Path):
    cv2.imwrite(str(img_p), img)
    with open(lbl_p, "w") as f:
        for c, b in zip(classes, boxes):
            f.write(f"{c} {' '.join(f'{v:.6f}' for v in b)}\n")

def process_split(split: str, src_root: Path, dst_root: Path, angles: list[int]) -> None:
    src_img = src_root / split / IMAGES
    src_lbl = src_root / split / LABELS
    dst_img = dst_root / split / IMAGES
    dst_lbl = dst_root / split / LABELS
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Started augmentation process")

    for img_p in tqdm(src_img.glob("*.jpg"), desc=f"{split}"):
        img, classes, boxes = read_sample(img_p, src_lbl / f"{img_p.stem}.txt")
        if img is None:
            LOGGER.warning("Cannot read %s", img_p)
            continue

        write_sample(img, classes, boxes, dst_img / img_p.name, dst_lbl / f"{img_p.stem}.txt")

        if not boxes:
            continue

        for a in angles:
            aug = rot_aug(a)(image=img, bboxes=boxes, class_labels=classes)
            name = f"{img_p.stem}_rot{a}"
            write_sample(
                aug["image"],
                aug["class_labels"],
                aug["bboxes"],
                dst_img / f"{name}.jpg",
                dst_lbl / f"{name}.txt",
            )

def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    source = Path(r"C:\personal\projects\meal-table-detection\data")
    target = Path(r"C:\personal\projects\meal-table-detection\data\augmented")
    angles = [90]  #  [90, 180, 270]
    for split in SPLITS:
        process_split(split, source, target, angles)

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()