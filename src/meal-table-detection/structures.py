from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
from typing import Dict, List
from enum import Enum, auto

BBOX, LABEL, SCORE = "bbox", "label", "score"

@dataclass
class BBox:
    center_x: float
    center_y: float
    width: float
    height: float

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_xyxy(self) -> Tuple[float, float, float, float]:
        left_bottom_x = self.center_x - self.width / 2
        left_bottom_y = self.center_y - self.height / 2
        right_top_x = self.center_x + self.width / 2
        right_top_y = self.center_y + self.height / 2
        return left_bottom_x, left_bottom_y, right_top_x, right_top_y

    def tolist(self) -> List[float]:
        return [self.center_x, self.center_y, self.width, self.height]

@dataclass
class TableItem:
    idx: int
    bbox: BBox
    label: str
    score: float

    def as_result(self) -> Dict[str, object]:
        return {BBOX: self.bbox, LABEL: self.label, SCORE: self.score}

class Category(Enum):
    PLATE         = auto()
    FOOD          = auto()
    STATE_OBJECT  = auto()
    OTHER         = auto()
