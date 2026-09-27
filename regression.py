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

    def PDF(self, start, end):
        return

    def value(self, point):
        return ONEOVERSQRTTWOPI / self.StdDev * pow(e, -())


class NormalRegression:
    @staticmethod
    def normalRegression(dataset: NDArray):
        average: float = np.mean(dataset)
        dataset = dataset - average
        dataset = np.abs(dataset)
        mean_deviation: float = np.mean(dataset)

        return NormalDistrubution(average, CONVERSION_COSTANT * mean_deviation)
