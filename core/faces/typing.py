from dataclasses import dataclass
import numpy as np


@dataclass
class Sample:
    image: np.ndarray
    landmarks: np.ndarray


