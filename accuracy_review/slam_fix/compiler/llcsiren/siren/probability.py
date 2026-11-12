"""Small probability helpers required by the grammar definitions."""

import math


def logbeta(a: float, b: float) -> float:
  return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def logcomb(n: int, k: int) -> float:
  return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

