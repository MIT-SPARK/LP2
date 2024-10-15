import numpy as np
from numpy.typing import ArrayLike
from typing import Union


def euclidean_distance(a: ArrayLike, b: ArrayLike) -> Union[float, np.ndarray]:
    return np.linalg.norm(np.array(a) - np.array(b))


class counter:
    def __init__(self):
        self.count = 2000

    def increment(self):
        self.count += 1
        return self.count
