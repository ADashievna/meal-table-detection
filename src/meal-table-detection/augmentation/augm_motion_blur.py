from pathlib import Path
import random, cv2
import albumentations as A
from shutil import copy2
from tqdm import tqdm
import logging

IMAGES = "images"
LABELS = "labels"
LOGGER = logging.getLogger(__name__)

def create_dirs(dst_root: Path) -> tuple[Path, Path]:
    img_dir = dst_root / IMAGES
    lbl_dir = dst_root / LABELS
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    return img_dir, lbl_dir

def apply_motion_blur(image_path: Path, transform: A.BasicTransform) -> any:
    img = cv2.imread(str(image_path))
    return transform(image=img)["image"] if img is not None else None

def augment(src: Path, dst: Path, prob: float, blur_range: tuple[int, int]) -> None:
    LOGGER.info("Augmenting images with motion blur")

    dst_img_dir, dst_lbl_dir = create_dirs(dst)
    src_img_dir = src / IMAGES
    src_lbl_dir = src / LABELS

    tfm = A.MotionBlur(blur_limit=blur_range, p=1)

    for img_path in tqdm(src_img_dir.glob("*.jpg"), desc="Augmenting"):
        if random.random() > prob: # save only with a certain probability
            continue
        lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            LOGGER.warning("Label file not found for %s", lbl_path)
            continue

        aug_img = apply_motion_blur(img_path, tfm)
        if aug_img is None:
            LOGGER.warning("Cannot read image", img_path)
            continue

        out_img = dst_img_dir / f"{img_path.stem}_motion.jpg"
        out_lbl = dst_lbl_dir / f"{img_path.stem}_motion.txt"

        cv2.imwrite(str(out_img), aug_img)
        copy2(lbl_path, out_lbl)


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    source = Path(r"C:\personal\projects\meal-table-detection\data\train")
    target = Path(r"C:\personal\projects\meal-table-detection\data\augmented")
    probability = 0.5
    blur_level_range = (5, 31)
    augment(source, target, probability, blur_level_range)

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()