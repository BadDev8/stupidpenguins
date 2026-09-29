from math import e, pi, sqrt

import numpy as np
from numpy.typing import NDArray

# the constant for converting mean deviation (faster to calculate) to standard deviation
CONVERSION_COSTANT = sqrt(pi / 2)
ONEOVERSQRTTWOPI = 1 / sqrt(2 * pi)


class NormalDistrubution:
    def __init__(self, average, StdDev):
        self.average = average
        self.StdDev = StdDev
        self.max_value = ONEOVERSQRTTWOPI / StdDev
        self.exponent_coefficient = -1 / (2 * StdDev**2)

    def _exponent(self, x: float) -> float:
        return self.exponent_coefficient * (x - self.average) ** 2

    def value(self, x) -> float:
        return self.max_value * pow(e, (self._exponent(x)))


def normalRegression(dataset: NDArray) -> NormalDistrubution:
    average: float = np.mean(dataset)
    mean_deviation: float = np.mean(np.abs(dataset - average))
    return NormalDistrubution(average, CONVERSION_COSTANT * mean_deviation)
