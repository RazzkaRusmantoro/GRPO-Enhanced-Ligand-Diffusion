import math

import torch

from policy import (
    clipped_surrogate,
    dhc_loss,
    gaussian_logp_parts,
    time_free_interpolate,
)


def test_time_free_coefficients_stay_in_unit_interval():
    mask = torch.tensor([0, 0, 1, 1])
    zt = torch.randn(4, 5)
    xhat = zt + 0.1
    out = time_free_interpolate(
        xhat, zt, mask, n_dims=3, interp_gamma=0.25, interp_eta=0.25,
        interp_scale=1.0)
    assert out.shape == zt.shape
    assert torch.isfinite(out).all()


def test_dhc_improves_when_new_policy_fits_positive_advantage():
    logp_x_old = torch.zeros(2, 3)
    logp_h_old = torch.zeros(2)
    logp_x_new = torch.ones(2, 3)
    logp_h_new = torch.ones(2)
    advantage = torch.tensor([1.0, 1.0])
    loss = dhc_loss(logp_x_new, logp_h_new, logp_x_old, logp_h_old, advantage)
    assert loss.item() < 0


def test_clip_limits_the_ratio():
    ratio = torch.tensor([5.0])
    adv = torch.tensor([1.0])
    loss = clipped_surrogate(ratio, adv, 0.2)
    assert torch.isclose(loss, torch.tensor(-(1.2)))


def test_negative_advantage_huge_ratio_is_bounded():
    ratio = torch.tensor([math.exp(20)])
    adv = torch.tensor([-1.0])
    unbounded = clipped_surrogate(ratio, adv, 0.2, dual_clip=None)
    bounded = clipped_surrogate(ratio, adv, 0.2, dual_clip=3.0)
    assert unbounded.item() > 1e6
    assert torch.isclose(bounded, torch.tensor(3.0))
    logp_new = torch.full((2, 3), 20.0)
    logp_h_new = torch.full((2,), 20.0)
    logp_old = torch.zeros(2, 3)
    logp_h_old = torch.zeros(2)
    loss = dhc_loss(
        logp_new, logp_h_new, logp_old, logp_h_old,
        torch.tensor([-1.0, -1.0]), dual_clip=3.0)
    assert loss.item() <= 6.0 + 1e-5
    assert torch.isfinite(loss)


def test_gaussian_logp_is_finite():
    z = torch.randn(5, 6)
    mu = z.clone()
    sigma = torch.ones(2, 1)
    mask = torch.tensor([0, 0, 0, 1, 1])
    logp_x, logp_h = gaussian_logp_parts(z, mu, sigma, mask, n_dims=3)
    assert logp_x.shape == (2, 3)
    assert logp_h.shape == (2,)
    assert torch.isfinite(logp_x).all()
    assert torch.isfinite(logp_h).all()
