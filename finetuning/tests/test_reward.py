from reward import normalize_sa, normalize_vina, threshold_aware_reward


def test_vina_maps_more_negative_to_higher_score():
    assert normalize_vina(-12) == 1.0
    assert normalize_vina(0) == 0.0
    assert normalize_vina(-6) == 0.5


def test_sa_uses_diffsbdd_scale():
    assert abs(normalize_sa(1.0) - 1.0) < 1e-6
    assert abs(normalize_sa(10.0) - 0.0) < 1e-6


def test_invalid_gets_zero_reward():
    item = threshold_aware_reward(
        vina=-8.0, qed=0.7, sa=3.0, valid=False)
    assert item.reward == 0.0
    assert item.failure == "invalid"


def test_threshold_reward_rises_near_cutoff():
    weak = threshold_aware_reward(
        vina=-3.0, qed=0.2, sa=8.0, valid=True, alpha=10.0, threshold=0.5)
    strong = threshold_aware_reward(
        vina=-9.0, qed=0.7, sa=2.0, valid=True, alpha=10.0, threshold=0.5)
    assert strong.reward > weak.reward
    assert strong.vina == -9.0


def test_vina_hi_eight_puts_minus_four_at_midpoint():
    assert abs(normalize_vina(-4.0, vina_hi=8.0) - 0.5) < 1e-6


def test_qed_sa_gated_until_vina_is_good_enough():
    easy_junk = threshold_aware_reward(
        vina=-1.0, qed=0.9, sa=2.0, valid=True,
        vina_hi=8.0, vina_gate=-4.0,
        weights={"vina": 2.0, "qed": 0.5, "sa": 0.5})
    docked = threshold_aware_reward(
        vina=-6.0, qed=0.9, sa=2.0, valid=True,
        vina_hi=8.0, vina_gate=-4.0,
        weights={"vina": 2.0, "qed": 0.5, "sa": 0.5})
    vina_only = threshold_aware_reward(
        vina=-1.0, qed=0.9, sa=2.0, valid=True,
        vina_hi=8.0, vina_gate=-4.0,
        weights={"vina": 2.0, "qed": 0.0, "sa": 0.0})
    assert abs(easy_junk.reward - vina_only.reward) < 1e-6
    assert docked.reward > easy_junk.reward
