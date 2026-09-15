from __future__ import annotations

import json
from pathlib import Path

import torch

from decode import score_molecules
from grpo import group_advantages
from policy import dhc_loss, gaussian_logp_parts, reverse_params
from sampler import attach_advantages, sample_group


def _reward_kwargs(cfg: dict) -> dict:
    reward = cfg["reward"]
    return {
        "threshold": float(reward.get("threshold", 0.5)),
        "alpha": float(reward.get("alpha", 10.0)),
        "weights": dict(reward.get("weights") or {"vina": 1, "qed": 1, "sa": 1}),
        "vina_lo": float(reward.get("vina_lo", 0.0)),
        "vina_hi": float(reward.get("vina_hi", 12.0)),
        "vina_gate": (
            None if reward.get("vina_gate") is None
            else float(reward["vina_gate"])
        ),
    }


def _transition_loss(wrapper, item: dict, clip_eps: float, dual_clip: float) -> torch.Tensor:
    device = wrapper.device
    n_dims = wrapper.n_dims
    zt = item["zt"].to(device)
    zs = item["zs"].to(device)
    xh_pocket = item["xh_pocket"].to(device)
    lig_mask = item["lig_mask"].to(device)
    pocket_mask = item["pocket_mask"].to(device)
    t = item["t"].to(device)
    s = item["s"].to(device)
    mu, sigma, _, _ = reverse_params(
        wrapper.ddpm, s, t, zt, xh_pocket, lig_mask, pocket_mask)
    logp_x, logp_h = gaussian_logp_parts(zs, mu, sigma, lig_mask, n_dims)
    return dhc_loss(
        logp_x, logp_h,
        item["logp_x_old"].to(device),
        item["logp_h_old"].to(device),
        item["advantage"].to(device),
        clip_eps=clip_eps,
        rectified=True,
        dual_clip=dual_clip,
    )


def train_ftdiff(
    wrapper,
    ft_rows,
    cfg: dict,
    *,
    qvina_binary: str | Path,
    obabel_binary: str | Path,
    output_dir: Path,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sampler_cfg = cfg["sampler"]
    train_cfg = cfg["train"]
    dock_cfg = cfg["docking"]
    optimizer = torch.optim.Adam(wrapper.trainable_parameters(), lr=float(train_cfg["lr"]))
    history = []
    rows = list(ft_rows.to_dict(orient="records"))
    pocket_cursor = 0

    for iteration in range(int(train_cfg["iterations"])):
        wrapper.model.eval()
        buffer: list[dict] = []
        iter_stats = []
        n_batch = int(train_cfg["batch_pockets"])
        batch = []
        for _ in range(n_batch):
            batch.append(rows[pocket_cursor % len(rows)])
            pocket_cursor += 1
        for pocket in batch:
            rollout = sample_group(
                wrapper,
                pdb_path=pocket["pdb_path"],
                ref_ligand=pocket["ref_ligand_sdf"],
                group_size=int(sampler_cfg["group_size"]),
                n_steps=int(sampler_cfg["n_steps"]),
                kind=str(sampler_cfg["kind"]),
                interp_gamma=float(sampler_cfg["interp_gamma"]),
                interp_eta=float(sampler_cfg["interp_eta"]),
                interp_scale=float(sampler_cfg["interp_scale"]),
                sanitize=bool(sampler_cfg.get("sanitize", True)),
                largest_frag=bool(sampler_cfg.get("largest_frag", True)),
            )
            breakdowns = score_molecules(
                rollout.molecules,
                pdb_path=pocket["pdb_path"],
                diffsbdd_root=cfg["paths"]["diffsbdd_root"],
                engine=dock_cfg["engine"],
                binary=qvina_binary,
                obabel=obabel_binary,
                box_size=float(dock_cfg["box_size"]),
                timeout=float(dock_cfg["timeout"]),
                reward_cfg=_reward_kwargs(cfg),
            )
            rewards = [item.reward for item in breakdowns]
            advantages = group_advantages(rewards)
            buffer.extend(attach_advantages(rollout.transitions, advantages))
            iter_stats.append({
                "pocket_id": pocket["pocket_id"],
                "mean_reward": sum(rewards) / max(len(rewards), 1),
                "mean_vina": sum(
                    (b.vina if b.vina is not None else 0.0) for b in breakdowns
                ) / max(len(breakdowns), 1),
                "n_valid": sum(1 for b in breakdowns if b.valid),
                "rewards": rewards,
            })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        wrapper.model.train()
        last_loss = 0.0
        scale = 1.0 / max(len(buffer), 1)
        clip_eps = float(train_cfg["clip_eps"])
        dual_clip = float(train_cfg.get("dual_clip", 3.0))
        has_signal = any(
            float(torch.as_tensor(item["advantage"]).abs().max()) > 1e-8
            for item in buffer
        )
        if has_signal:
            for _ in range(int(train_cfg["inner_updates"])):
                optimizer.zero_grad(set_to_none=True)
                running = 0.0
                for item in buffer:
                    loss = _transition_loss(wrapper, item, clip_eps, dual_clip)
                    (loss * scale).backward()
                    running += float(loss.detach().cpu())
                    del loss
                torch.nn.utils.clip_grad_norm_(
                    wrapper.trainable_parameters(), float(train_cfg["max_grad_norm"]))
                optimizer.step()
                last_loss = running / max(len(buffer), 1)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        record = {
            "iteration": iteration,
            "loss": last_loss,
            "pockets": iter_stats,
        }
        history.append(record)
        (output_dir / "train_log.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8")
        n_iters = int(train_cfg["iterations"])
        if iteration == 0 or (iteration + 1) % int(train_cfg.get("save_every", 10)) == 0 \
                or iteration + 1 == n_iters:
            if not hasattr(train_ftdiff, "_ckpt_template"):
                train_ftdiff._ckpt_template = torch.load(
                    cfg["paths"]["checkpoint"], map_location="cpu")
            payload = dict(train_ftdiff._ckpt_template)
            payload["state_dict"] = {k: v.detach().cpu()
                                     for k, v in wrapper.model.state_dict().items()}
            last_path = output_dir / "last.ckpt"
            torch.save(payload, last_path)
            torch.save(payload, output_dir / f"ftdiff_iter{iteration:03d}.ckpt")
        n_valid = sum(p["n_valid"] for p in iter_stats)
        n_mols = sum(len(p["rewards"]) for p in iter_stats)
        print(
            f"iter {iteration}/{n_iters - 1} loss={last_loss:.4f} "
            f"reward={sum(p['mean_reward'] for p in iter_stats)/len(iter_stats):.3f} "
            f"valid={n_valid}/{n_mols}",
            flush=True,
        )
    return {"history": history, "last_checkpoint": str(output_dir / "last.ckpt")}
