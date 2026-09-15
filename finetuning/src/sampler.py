from __future__ import annotations

from dataclasses import dataclass

import torch

from decode import latents_to_molecules
from policy import (
    gaussian_logp_parts,
    reverse_params,
    time_free_interpolate,
)


@dataclass
class GroupRollout:
    molecules: list
    transitions: list[dict]
    num_nodes: int
    pocket_com_before: torch.Tensor


def _prepare_pocket(wrapper, pdb_file: str, ref_ligand: str, repeats: int):
    from Bio.PDB import PDBParser
    import utils

    structure = PDBParser(QUIET=True).get_structure("", pdb_file)[0]
    residues = utils.get_pocket_from_ligand(structure, ref_ligand)
    return wrapper.model.prepare_pocket(residues, repeats=repeats)


def _init_latent(ddpm, pocket, num_nodes_lig):
    n_samples = len(pocket["size"])
    device = pocket["x"].device
    _, pocket = ddpm.normalize(pocket=pocket)
    xh0_pocket = torch.cat([pocket["x"], pocket["one_hot"]], dim=1)
    import utils
    lig_mask = utils.num_nodes_to_batch_mask(n_samples, num_nodes_lig, device)
    from torch_scatter import scatter_mean
    mu_lig_x = scatter_mean(pocket["x"], pocket["mask"], dim=0)
    mu_lig_h = torch.zeros((n_samples, ddpm.atom_nf), device=device)
    mu_lig = torch.cat((mu_lig_x, mu_lig_h), dim=1)[lig_mask]
    sigma = torch.ones_like(pocket["size"]).unsqueeze(1)
    z_lig, xh_pocket = ddpm.sample_normal_zero_com(
        mu_lig, xh0_pocket, sigma, lig_mask, pocket["mask"])
    return z_lig, xh_pocket, lig_mask, pocket["mask"]


def _restore_com(xh_lig, xh_pocket, lig_mask, pocket_mask, n_dims, pocket_com_before):
    from torch_scatter import scatter_mean
    pocket_com_after = scatter_mean(xh_pocket[:, :n_dims], pocket_mask, dim=0)
    shift = pocket_com_before - pocket_com_after
    xh_pocket = xh_pocket.clone()
    xh_lig = xh_lig.clone()
    xh_pocket[:, :n_dims] = xh_pocket[:, :n_dims] + shift[pocket_mask]
    xh_lig[:, :n_dims] = xh_lig[:, :n_dims] + shift[lig_mask]
    return xh_lig, xh_pocket


def sample_group(
    wrapper,
    *,
    pdb_path: str,
    ref_ligand: str,
    group_size: int,
    n_steps: int,
    kind: str,
    interp_gamma: float,
    interp_eta: float,
    interp_scale: float,
    sanitize: bool,
    largest_frag: bool,
) -> GroupRollout:
    ddpm = wrapper.ddpm
    device = wrapper.device
    n_dims = wrapper.n_dims
    pocket = _prepare_pocket(wrapper, pdb_path, ref_ligand, group_size)
    pocket = {key: (val.to(device) if torch.is_tensor(val) else val)
              for key, val in pocket.items()}
    from torch_scatter import scatter_mean
    pocket_com_before = scatter_mean(pocket["x"], pocket["mask"], dim=0)

    with torch.no_grad():
        num_nodes = ddpm.size_distribution.sample_conditional(
            n1=None, n2=pocket["size"][:1])
        num_nodes_lig = num_nodes.repeat(group_size)
        z_lig, xh_pocket, lig_mask, pocket_mask = _init_latent(
            ddpm, pocket, num_nodes_lig)
        n_samples = group_size
        transitions: list[dict] = []
        for step in range(n_steps):
            t_val = (n_steps - step) / n_steps
            s_val = (n_steps - step - 1) / n_steps
            s = torch.full((n_samples, 1), s_val, device=device)
            t = torch.full((n_samples, 1), t_val, device=device)
            mu, sigma, _, xhat = reverse_params(
                ddpm, s, t, z_lig, xh_pocket, lig_mask, pocket_mask)
            if kind == "skip_ddpm":
                noise = torch.randn_like(mu)
                sigma_atom = sigma[lig_mask] if sigma.shape[0] != mu.shape[0] else sigma
                z_next = mu + sigma_atom * noise
                z_x, x_pocket = ddpm.remove_mean_batch(
                    z_next[:, :n_dims], xh_pocket[:, :n_dims],
                    lig_mask, pocket_mask)
                z_next = torch.cat([z_x, z_next[:, n_dims:]], dim=-1)
                xh_pocket = torch.cat([x_pocket, xh_pocket[:, n_dims:]], dim=-1)
            elif kind == "time_free":
                z_next = time_free_interpolate(
                    xhat, z_lig, lig_mask, n_dims=n_dims,
                    interp_gamma=interp_gamma, interp_eta=interp_eta,
                    interp_scale=interp_scale)
                z_x, x_pocket = ddpm.remove_mean_batch(
                    z_next[:, :n_dims], xh_pocket[:, :n_dims],
                    lig_mask, pocket_mask)
                z_next = torch.cat([z_x, z_next[:, n_dims:]], dim=-1)
                xh_pocket = torch.cat([x_pocket, xh_pocket[:, n_dims:]], dim=-1)
            else:
                raise ValueError(f"unknown sampler {kind}")
            logp_x, logp_h = gaussian_logp_parts(
                z_next, mu, sigma, lig_mask, n_dims)
            transitions.append({
                "zt": z_lig.detach().cpu(),
                "zs": z_next.detach().cpu(),
                "xh_pocket": xh_pocket.detach().cpu(),
                "lig_mask": lig_mask.detach().cpu(),
                "pocket_mask": pocket_mask.detach().cpu(),
                "t": t.detach().cpu(),
                "s": s.detach().cpu(),
                "logp_x_old": logp_x.detach().cpu(),
                "logp_h_old": logp_h.detach().cpu(),
            })
            z_lig = z_next

        x_lig, h_lig, x_pocket, h_pocket = ddpm.sample_p_xh_given_z0(
            z_lig, xh_pocket, lig_mask, pocket_mask, n_samples)
        xh_lig = torch.cat([x_lig, h_lig], dim=1)
        xh_pocket_final = torch.cat([x_pocket, h_pocket], dim=1)
        xh_lig, _ = _restore_com(
            xh_lig, xh_pocket_final, lig_mask, pocket_mask, n_dims,
            pocket_com_before)

    molecules = latents_to_molecules(
        wrapper, xh_lig, lig_mask, sanitize=sanitize, largest_frag=largest_frag)
    return GroupRollout(
        molecules=molecules,
        transitions=transitions,
        num_nodes=int(num_nodes_lig[0].item()),
        pocket_com_before=pocket_com_before.detach().cpu(),
    )


def attach_advantages(transitions: list[dict], advantages: list[float]) -> list[dict]:
    adv = torch.tensor(advantages, dtype=torch.float32)
    out = []
    for item in transitions:
        packed = dict(item)
        packed["advantage"] = adv
        out.append(packed)
    return out
