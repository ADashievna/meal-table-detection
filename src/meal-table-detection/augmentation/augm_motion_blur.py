from pathlib import Path
import random, cv2
import albumentations as A
from shutil import copy2
from tqdm import tqdm
import argparse

IMAGES = "images"
LABELS = "labels"

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

    dst_img_dir, dst_lbl_dir = create_dirs(dst)
    src_img_dir = src / IMAGES
    src_lbl_dir = src / LABELS

    tfm = A.MotionBlur(blur_limit=blur_range, p=1)

    for img_path in tqdm(src_img_dir.glob("*.jpg"), desc="Augmenting"):
        if random.random() > prob: # save only with a certain probability
            continue
        lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        aug_img = apply_motion_blur(img_path, tfm)
        if aug_img is None:
            continue

        out_img = dst_img_dir / f"{img_path.stem}_motion.jpg"
        out_lbl = dst_lbl_dir / f"{img_path.stem}_motion.txt"

        cv2.imwrite(str(out_img), aug_img)
        copy2(lbl_path, out_lbl)


def main():
    parser = argparse.ArgumentParser(description="Run motion blur augmentation script, expects source and target paths."
                                                 " In the source path, there should be 'images' and 'labels' directories with images and labels in YOLO format.")
    parser.add_argument(
        "--source_path",
        type=str,
        required=True,
        help="Path to the dataset for augmentation (should contain 'images' and 'labels' directories)."
    )
    parser.add_argument(
        "--target_path",
        type=str,
        required=True,
        help="Path to the directory to save data after augmentation"
    )
    parser.add_argument(
        "--probability",
        type=float,
        required=True,
        help="Probability of applying motion blur to each image (0.0 - 1.0)."
    )
    parser.add_argument(
        "--blur_level_range",
        type=int,
        nargs=2,
        required=True,
        help="Two integers specifying the min and max blur level (e.g., 3 7"
    )
    args = parser.parse_args()

    augment(
        Path(args.source_path),
        Path(args.target_path),
        args.probability,
        args.blur_level_range
    )



if __name__ == "__main__":
    main()