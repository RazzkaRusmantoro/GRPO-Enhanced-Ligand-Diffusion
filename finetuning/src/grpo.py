from __future__ import annotations

import math


def group_advantages(
    rewards: list[float],
    eps: float = 1e-8,
) -> list[float]:
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    var = sum((item - mean) ** 2 for item in rewards) / len(rewards)
    std = math.sqrt(var)
    denom = std + eps
    return [(item - mean) / denom for item in rewards]
