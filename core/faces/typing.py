import dataclasses
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Sample:
    image: np.ndarray
    landmarks: np.ndarray


def samples_to_dicts(samples: List[Sample]):
    for sample in samples:
        yield dataclasses.asdict(sample)
