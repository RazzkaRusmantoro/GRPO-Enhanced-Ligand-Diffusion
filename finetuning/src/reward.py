from __future__ import annotations

import math
from dataclasses import dataclass


def _sigmoid(value: float) -> float:
    value = max(-60.0, min(60.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def normalize_vina(
    vina: float | None,
    *,
    vina_lo: float = 0.0,
    vina_hi: float = 12.0,
) -> float | None:
    if vina is None or not math.isfinite(vina):
        return None
    span = vina_hi - vina_lo
    if span <= 0:
        raise ValueError("vina_hi must be greater than vina_lo")
    return min(1.0, max(0.0, (-float(vina) - vina_lo) / span))


def normalize_sa(sa: float | None) -> float | None:
    if sa is None or not math.isfinite(sa):
        return None
    return min(1.0, max(0.0, (10.0 - float(sa)) / 9.0))


def threshold_term(score: float, *, threshold: float, alpha: float) -> float:
    return _sigmoid(alpha * (score - threshold))


@dataclass(frozen=True)
class RewardBreakdown:
    reward: float
    vina: float | None
    s_vina: float | None
    s_qed: float | None
    s_sa: float | None
    valid: bool
    connected: bool
    failure: str | None = None


def threshold_aware_reward(
    *,
    vina: float | None,
    qed: float | None,
    sa: float | None,
    valid: bool,
    connected: bool = True,
    threshold: float = 0.5,
    alpha: float = 10.0,
    weights: dict[str, float] | None = None,
    vina_lo: float = 0.0,
    vina_hi: float = 12.0,
    vina_gate: float | None = None,
) -> RewardBreakdown:
    weights = weights or {"vina": 1.0, "qed": 1.0, "sa": 1.0}
    s_vina = normalize_vina(vina, vina_lo=vina_lo, vina_hi=vina_hi)
    s_qed = None if qed is None or not math.isfinite(qed) else min(1.0, max(0.0, float(qed)))
    s_sa = normalize_sa(sa)
    if not valid or not connected or s_vina is None or s_qed is None or s_sa is None:
        failure = "invalid" if not valid else ("disconnected" if not connected else "missing_score")
        return RewardBreakdown(0.0, vina, s_vina, s_qed, s_sa, valid, connected, failure)
    total = float(weights.get("vina", 0.0)) * threshold_term(
        s_vina, threshold=threshold, alpha=alpha)
    chem_ok = vina_gate is None or (vina is not None and vina <= float(vina_gate))
    if chem_ok:
        for name, score in (("qed", s_qed), ("sa", s_sa)):
            total += float(weights.get(name, 0.0)) * threshold_term(
                score, threshold=threshold, alpha=alpha)
    return RewardBreakdown(float(total), vina, s_vina, s_qed, s_sa, True, True, None)
