from pathlib import Path
from typing import TypedDict
from ultralytics import YOLO
import config

class Metrics(TypedDict):
    map_50_95: float
    map_50: float
    precision: float
    recall: float
    f1: float

def evaluate(model: YOLO, dataset_yaml: Path, split: str = "test", plots: bool = True) -> Metrics:
    r = model.val(
        data=dataset_yaml,
        split=split,
        plots=plots
    )
    precision, recall  = r.box.mp, r.box.mr
    f1 = 2 * precision * recall / float(precision + recall)
    return Metrics(
        precision=precision,
        recall=recall,
        f1=f1,
        map_50_95 = r.box.map,
        map_50    = r.box.map50
    )

def print_metrics(m: Metrics) -> None:
    print(f"Precision      : {m['precision']:.4f}")
    print(f"Recall         : {m['recall']:.4f}")
    print(f"F1-score       : {m['f1']:.4f}")
    print(f"mAP@0.5:0.95   : {m['map_50_95']:.4f}")
    print(f"mAP@0.5        : {m['map_50']:.4f}")

def main() -> None:
    model = YOLO(config.MODEL_PATH)
    metrics = evaluate(model, Path(config.DATASET_YAML_PATH), split="test")
    print_metrics(metrics)

if __name__ == "__main__":
    main()