#!/usr/bin/env python3
"""Exact posterior moments and ordered-data log evidence; standard library only.

All variances here follow Siren's convention. Wheels returns the *smoothed*
trajectory conditioned on every observation, in (velocities, angular rates) order.
"""

import json
import math
from fractions import Fraction
from itertools import product

MODELS = ("coin", "gaussian-gaussian", "gtree", "kalman", "wheels", "outlier", "slam",
          "gaussian-gaussian-hybrid")
SLAM_DATA = ((0, 1), (0, -1))
WHEELS_DATA = (
    (201.79897693, -32.779256728),
    (206.423095274, -31.9258541027),
    (207.900486767, -31.5701198412),
    (210.298303206, -32.0217383347),
    (206.010088006, -31.3001236526),
    (207.230561559, -35.9919009388),
    (207.802456076, -34.6068500903),
)
KALMAN_DATA = tuple(range(1, 10))  # Siren's upper bound is exclusive.
OUTLIER_DATA = tuple(range(1, 10))


def gaussian_hybrid_reference():
    """Sum the two Bernoulli branches and integrate x analytically in each.

    At input=0, the true branch has x|y ~ N(0, 10/11); the false branch
    leaves x ~ N(0, 10). Branch evidences are N(0;0,11) and N(0;0,100).
    The reference is independent of either system's sampled branch counts.
    """
    true_evidence = 1.0 / math.sqrt(2 * math.pi * 11.0)
    false_evidence = 1.0 / math.sqrt(2 * math.pi * 100.0)
    evidence = 0.5 * (true_evidence + false_evidence)
    true_weight = true_evidence / (true_evidence + false_evidence)
    false_weight = false_evidence / (true_evidence + false_evidence)
    variance = true_weight * (10.0 / 11.0) + false_weight * 10.0
    return {"labels": ["x"], "mean": [0.0], "variance": [variance],
            "covariance": [[variance]], "log_z": math.log(evidence),
            "posterior": "Two-component centered Gaussian mixture",
            "reference_method": "Exact enumeration of both Bernoulli branches and Gaussian conjugacy; no Monte Carlo",
            "enumerated_configurations": 2, "observation": 0.0,
            "components": [
                {"b": True, "weight": true_weight, "mean": 0.0, "variance": 10.0 / 11.0},
                {"b": False, "weight": false_weight, "mean": 0.0, "variance": 10.0}],
            "exact_true_weight": "10 / (10 + sqrt(11))",
            "exact_variance": "(100/11 + 10*sqrt(11)) / (10 + sqrt(11))",
            "metric_note": "Both component means are zero; mean error alone cannot detect incorrect mixture weights."}


def outlier_reference():
    """Enumerate all 2^9 indicators; integrate each Gaussian path exactly.

    In this source the variable called `out` is true for the INLIER branch.
    False contributes N(y; 0, 100) and leaves the predicted state unchanged.
    """
    components = []
    for indicators in product((False, True), repeat=len(OUTLIER_DATA)):
        mean, variance, log_weight = 0.0, 100.0, 0.0
        for y, inlier in zip(OUTLIER_DATA, indicators):
            variance += 1.0
            log_weight += math.log(0.9 if inlier else 0.1)
            if inlier:
                innovation = variance + 1.0
                residual = y - mean
                log_weight -= 0.5 * (math.log(2 * math.pi * innovation) + residual**2 / innovation)
                mean += variance / innovation * residual
                variance /= innovation
            else:
                log_weight -= 0.5 * (math.log(2 * math.pi * 100.0) + y*y / 100.0)
        components.append((log_weight, mean, variance))
    maximum = max(w for w, _, _ in components)
    weights = [math.exp(w - maximum) for w, _, _ in components]
    total = math.fsum(weights)
    mean = math.fsum(w * m for w, (_, m, _) in zip(weights, components)) / total
    variance = math.fsum(w * (v + (m - mean)**2)
                         for w, (_, m, v) in zip(weights, components)) / total
    return {"labels": ["x9"], "mean": [mean], "variance": [variance],
            "covariance": [[variance]], "log_z": maximum + math.log(total),
            "posterior": "512-component Gaussian mixture",
            "reference_method": "Exhaustive indicator enumeration and scalar Gaussian conditioning; no Monte Carlo",
            "enumerated_configurations": len(components)}


def slam_reference():
    """Exact rational enumeration of two map cells and two wheel-slip bits."""
    configurations = []
    for cells in product((0, 1), repeat=2):
        for slips in product((False, True), repeat=len(SLAM_DATA)):
            weight, previous, xs = Fraction(1, 4), 0, []
            for (observation, cmd), slip in zip(SLAM_DATA, slips):
                weight *= Fraction(1 if slip else 9, 10)
                cmd2 = cmd if 0 <= previous + cmd <= 1 else 0
                x = previous if slip else previous + cmd2
                weight *= Fraction(9 if cells[x] == observation else 1, 10)
                xs.append(x)
                previous = x
            configurations.append((weight, cells + tuple(xs)))
    evidence = sum(w for w, _ in configurations)
    mean = [sum(w * values[i] for w, values in configurations) / evidence for i in range(4)]
    covariance = [[sum(w * (values[i] - mean[i]) * (values[j] - mean[j])
                       for w, values in configurations) / evidence
                   for j in range(4)] for i in range(4)]
    return {"labels": ["cell0", "cell1", "x1", "x2"], "mean": list(map(float, mean)),
            "variance": [float(covariance[i][i]) for i in range(4)],
            "covariance": [list(map(float, row)) for row in covariance],
            "log_z": math.log(float(evidence)), "posterior": "Finite discrete posterior",
            "reference_method": "Exhaustive map/slip enumeration with exact rational weights; no Monte Carlo",
            "enumerated_configurations": len(configurations),
            "exact_mean": list(map(str, mean)), "exact_evidence": str(evidence),
            "exact_covariance": [list(map(str, row)) for row in covariance]}


def condition(mean, covariance, observations):
    """Condition a Gaussian on (value, linear coefficients, noise variance)."""
    mean = list(mean)
    covariance = [list(row) for row in covariance]
    log_z = 0.0
    for value, h, noise in observations:
        ph = [sum(p * hi for p, hi in zip(row, h)) for row in covariance]
        innovation_variance = noise + sum(hi * pi for hi, pi in zip(h, ph))
        residual = value - sum(hi * mi for hi, mi in zip(h, mean))
        if innovation_variance <= 0:
            raise ValueError("Nonpositive predictive variance")
        log_z -= 0.5 * (math.log(2 * math.pi * innovation_variance)
                        + residual * residual / innovation_variance)
        mean = [mi + pi * residual / innovation_variance for mi, pi in zip(mean, ph)]
        covariance = [[p - ph[i] * ph[j] / innovation_variance
                       for j, p in enumerate(row)]
                      for i, row in enumerate(covariance)]
    return mean, covariance, log_z


def gaussian_problem(name):
    """Return the joint prior, observations, returned coordinates and labels."""
    if name == "gaussian-gaussian":
        return [0.0], [[10.0]], [(3.0, [1.0], 1.0)], [0], ["x"]
    if name == "gtree":
        return ([0.0, 0.0], [[100.0, 100.0], [100.0, 110.0]],
                [(2.3, [0.0, 1.0], 1000.0), (-55.0, [1.0, 0.0], 1000.0)],
                [0], ["a"])
    if name == "kalman":
        n = len(KALMAN_DATA) + 1
        covariance = [[100.0 + min(i, j) for j in range(n)] for i in range(n)]
        observations = []
        for t, y in enumerate(KALMAN_DATA, 1):
            observations.append((float(y), [float(i == t) for i in range(n)], 1.0))
        return [0.0] * n, covariance, observations, [n - 1], [f"x{n - 1}"]
    if name == "wheels":
        n = len(WHEELS_DATA)
        covariance = [[2500.0 + min(i % n, j % n) if i // n == j // n else 0.0
                       for j in range(2 * n)] for i in range(2 * n)]
        observations = []
        for t, (left, right) in enumerate(WHEELS_DATA):
            for value, coefficient, noise in ((left, -2.0, 1.0), (right, 2.0, 0.95)):
                h = [0.0] * (2 * n)
                h[t], h[n + t] = 1.0, coefficient
                observations.append((value, h, noise))
        labels = [f"v{t + 1}" for t in range(n)] + [f"omega{t + 1}" for t in range(n)]
        return [0.0] * (2 * n), covariance, observations, list(range(2 * n)), labels
    raise ValueError(f"Unknown Gaussian model: {name}")


def reference(name):
    if name == "gaussian-gaussian-hybrid":
        return gaussian_hybrid_reference()
    if name == "outlier":
        return outlier_reference()
    if name == "slam":
        return slam_reference()
    if name == "coin":
        # Seven true observations and one false, prior Beta(1, 1).
        a, b = 8.0, 2.0
        variance = a * b / ((a + b) ** 2 * (a + b + 1))
        return {"labels": ["xt"], "mean": [a / (a + b)], "variance": [variance],
                "covariance": [[variance]], "log_z": -math.log(72.0),
                "posterior": "Beta(8, 2)"}
    mean, covariance, observations, indices, labels = gaussian_problem(name)
    mean, covariance, log_z = condition(mean, covariance, observations)
    return {"labels": labels, "mean": [mean[i] for i in indices],
            "variance": [covariance[i][i] for i in indices],
            "covariance": [[covariance[i][j] for j in indices] for i in indices],
            "log_z": log_z, "posterior": "Gaussian"}


if __name__ == "__main__":
    print(json.dumps({name: reference(name) for name in MODELS}, indent=2))
