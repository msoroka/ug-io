import numpy
import math


def angle(v1, v2):
    return math.degrees(
        math.acos(
            scalar(v1, v2) / (length(v1) * length(v2))
        )
    )


def length(v1):
    return numpy.sqrt(numpy.dot(v1, v1))


def scalar(v1, v2):
    return numpy.dot(v1, v2)


print(angle([1, 2], [3, 4]))
