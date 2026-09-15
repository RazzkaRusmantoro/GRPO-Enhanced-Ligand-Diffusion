from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def scatter_sum(values: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = values.new_zeros((n,) + values.shape[1:])
    out.index_add_(0, index.long(), values)
    return out


def scatter_mean(values: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    total = scatter_sum(values, index, n)
    count = scatter_sum(torch.ones_like(values), index, n).clamp_min(1)
    return total / count


def reverse_params(ddpm, s, t, zt_lig, xh_pocket, ligand_mask, pocket_mask):
    gamma_s = ddpm.gamma(s)
    gamma_t = ddpm.gamma(t)
    sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
        ddpm.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt_lig)
    sigma_s = ddpm.sigma(gamma_s, target_tensor=zt_lig)
    sigma_t = ddpm.sigma(gamma_t, target_tensor=zt_lig)
    eps_t_lig, _ = ddpm.dynamics(zt_lig, xh_pocket, t, ligand_mask, pocket_mask)
    mu_lig = zt_lig / alpha_t_given_s[ligand_mask] - (
        sigma2_t_given_s / alpha_t_given_s / sigma_t
    )[ligand_mask] * eps_t_lig
    sigma = sigma_t_given_s * sigma_s / sigma_t
    xhat = ddpm.xh_given_zt_and_epsilon(zt_lig, eps_t_lig, gamma_t, ligand_mask)
    return mu_lig, sigma, eps_t_lig, xhat


def gaussian_logp_parts(
    z_next: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    mask: torch.Tensor,
    n_dims: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_graphs = int(mask.max().item()) + 1
    scale = sigma[mask] if sigma.shape[0] != z_next.shape[0] else sigma
    if scale.ndim == 1:
        scale = scale.unsqueeze(-1)
    scale = scale.clamp_min(1e-6)
    log_scale = torch.log(scale.squeeze(-1))
    const = 0.5 * math.log(2 * math.pi)

    x_err = ((z_next[:, :n_dims] - mu[:, :n_dims]) / scale) ** 2
    h_err = ((z_next[:, n_dims:] - mu[:, n_dims:]) / scale) ** 2
    ones = torch.ones(z_next.shape[0], device=z_next.device, dtype=z_next.dtype)
    n_atoms = scatter_sum(ones, mask, n_graphs)
    sig_g = scatter_mean(log_scale.exp(), mask, n_graphs).clamp_min(1e-6)

    logp_x = []
    for dim in range(n_dims):
        sse = scatter_sum(x_err[:, dim], mask, n_graphs)
        logp_x.append(
            -0.5 * sse - n_atoms * (torch.log(sig_g) + const)
        )
    n_feat = z_next.shape[1] - n_dims
    sse_h = scatter_sum(h_err.sum(dim=-1), mask, n_graphs)
    logp_h = -0.5 * sse_h - n_atoms * n_feat * (torch.log(sig_g) + const)
    return torch.stack(logp_x, dim=-1), logp_h


def clipped_surrogate(
    ratio: torch.Tensor,
    advantage: torch.Tensor,
    clip_eps: float,
    dual_clip: float = 3.0,
) -> torch.Tensor:
    while advantage.ndim < ratio.ndim:
        advantage = advantage.unsqueeze(-1)
    unclipped = ratio * advantage
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantage
    surrogate = torch.min(unclipped, clipped)
    if dual_clip is not None and dual_clip > 1.0:
        floor = dual_clip * advantage
        surrogate = torch.where(advantage < 0, torch.max(surrogate, floor), surrogate)
    return -surrogate


def dhc_loss(
    logp_x_new: torch.Tensor,
    logp_h_new: torch.Tensor,
    logp_x_old: torch.Tensor,
    logp_h_old: torch.Tensor,
    advantage: torch.Tensor,
    clip_eps: float = 0.2,
    rectified: bool = True,
    dual_clip: float = 3.0,
) -> torch.Tensor:
    log_rx = (logp_x_new - logp_x_old).clamp(-20, 20)
    log_rh = (logp_h_new - logp_h_old).clamp(-20, 20)
    ratio_x = torch.exp(log_rx)
    ratio_h = torch.exp(log_rh)
    kwargs = {"clip_eps": clip_eps, "dual_clip": dual_clip}
    if rectified:
        loss_x = clipped_surrogate(ratio_x, advantage, **kwargs).mean(dim=-1)
    else:
        ratio_joint = torch.exp(log_rx.sum(dim=-1).clamp(-20, 20))
        loss_x = clipped_surrogate(ratio_joint, advantage, **kwargs)
    loss_h = clipped_surrogate(ratio_h, advantage, **kwargs)
    return (loss_x + loss_h).mean()


def time_free_interpolate(
    xhat: torch.Tensor,
    zt: torch.Tensor,
    mask: torch.Tensor,
    *,
    n_dims: int,
    interp_gamma: float,
    interp_eta: float,
    interp_scale: float,
) -> torch.Tensor:
    n_graphs = int(mask.max().item()) + 1
    x0, h0 = xhat[:, :n_dims], xhat[:, n_dims:]
    xt, ht = zt[:, :n_dims], zt[:, n_dims:]
    dist2 = scatter_sum(((x0 - xt) ** 2).sum(dim=-1), mask, n_graphs)
    coeff_a = 1.0 / (1.0 + interp_gamma * torch.log1p(dist2))
    p0 = F.softmax(h0, dim=-1).clamp_min(1e-8)
    pt = F.softmax(ht, dim=-1).clamp_min(1e-8)
    kl_atom = (p0 * (p0.log() - pt.log())).sum(dim=-1)
    kl = scatter_mean(kl_atom, mask, n_graphs)
    coeff_c = 1.0 / (1.0 + interp_eta * torch.log1p(kl))
    a_atom = coeff_a[mask].unsqueeze(-1)
    c_atom = coeff_c[mask].unsqueeze(-1)
    x_next = interp_scale * (a_atom * x0 + (1.0 - a_atom) * xt)
    h_next = c_atom * h0 + (1.0 - c_atom) * ht
    return torch.cat([x_next, h_next], dim=-1)


def remove_ligand_com(z_lig: torch.Tensor, mask: torch.Tensor, n_dims: int) -> torch.Tensor:
    n_graphs = int(mask.max().item()) + 1
    com = scatter_mean(z_lig[:, :n_dims], mask, n_graphs)
    out = z_lig.clone()
    out[:, :n_dims] = out[:, :n_dims] - com[mask]
    return out
