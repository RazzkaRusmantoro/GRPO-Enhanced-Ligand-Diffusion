import math

from grpo import group_advantages


def test_group_advantages_are_standardized():
    values = group_advantages([1.0, 2.0, 3.0, 4.0])
    assert abs(sum(values)) < 1e-8
    mean = sum(values) / len(values)
    var = sum((item - mean) ** 2 for item in values) / len(values)
    assert abs(math.sqrt(var) - 1.0) < 1e-6


def test_better_sample_gets_positive_advantage():
    values = group_advantages([0.1, 0.8])
    assert values[1] > 0
    assert values[0] < 0
