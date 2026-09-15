from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT.parent / "figures"
EVALS = ROOT / "outputs"

BASE_C = "#0072B2"
TUNE_C = "#D55E00"
GRAY = "#4D4D4D"
GRID = "#D0D0D0"
RULE = "#888888"

BASE = "DiffSBDD"
TUNED = "GRPO fine-tuned"
TUNED_TICK = "GRPO\nfine-tuned"


def _style() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "regular",
        "axes.labelsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "text.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "stixsans",
    })


def _load_mols(run: str, tag: str) -> list[dict]:
    return json.loads((EVALS / run / "eval" / f"{tag}_molecules.json").read_text(encoding="utf-8"))


def _load_summary(run: str, tag: str) -> dict:
    return json.loads((EVALS / run / "eval" / f"{tag}_summary.json").read_text(encoding="utf-8"))


def _vinas(rows: list[dict]) -> np.ndarray:
    return np.array(
        [r["vina"] for r in rows if r.get("valid") and r.get("vina") is not None],
        dtype=float,
    )


def _qed_vina(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    pairs = [
        (r["vina"], r["qed"])
        for r in rows
        if r.get("valid") and r.get("vina") is not None and r.get("qed") is not None
    ]
    if not pairs:
        return np.array([]), np.array([])
    arr = np.array(pairs, dtype=float)
    return arr[:, 0], arr[:, 1]


def _pocket_means(rows: list[dict]) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("valid") and row.get("vina") is not None:
            buckets[row["pocket_id"]].append(float(row["vina"]))
    return {key: float(np.mean(vals)) for key, vals in buckets.items()}


def _short_pocket(name: str) -> str:
    return name.split("-")[0]


def _panel(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(False)
    ax.tick_params(length=3.2)


def _save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def fig_summary(u: dict, f: dict) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(10.6, 3.35))
    panels = [
        ("a  Mean Vina (kcal/mol)", u["mean_vina"], f["mean_vina"], True, "{:.2f}"),
        ("b  Validity (%)", 100 * u["validity"], 100 * f["validity"], False, "{:.1f}"),
        ("c  QED", u["mean_qed"], f["mean_qed"], False, "{:.2f}"),
        ("d  SA", u["mean_sa"], f["mean_sa"], False, "{:.2f}"),
    ]
    for ax, (title, a, b, signed, fmt) in zip(axes, panels):
        _panel(ax)
        ax.set_title(title, loc="left", pad=6)
        bars = ax.bar(
            [0, 1], [a, b], width=0.64, color=[BASE_C, TUNE_C],
            edgecolor="black", linewidth=0.5, zorder=3,
        )
        span = abs(b - a) if b != a else (abs(a) * 0.08 + 0.08)
        for bar, val in zip(bars, (a, b)):
            if signed:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, val - 0.08 * span,
                    fmt.format(val), ha="center", va="top", fontsize=8, color="black",
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, val + 0.03 * (max(a, b) + 1e-6),
                    fmt.format(val), ha="center", va="bottom", fontsize=8, color="black",
                )
        ax.set_xticks([0, 1], [BASE, TUNED_TICK])
        if signed:
            ax.set_ylim(min(a, b) - 0.45, 0.2)
            ax.axhline(0, color="black", lw=0.6)
        else:
            ax.set_ylim(0, max(a, b) * 1.22 + (6 if "Validity" in title else 0.05))
    fig.tight_layout()
    _save(fig, "01_summary_bars")


def fig_vina_dist(u_rows: list[dict], f_rows: list[dict]) -> None:
    uv, fv = _vinas(u_rows), _vinas(f_rows)
    rng = np.random.default_rng(17)
    fig, ax = plt.subplots(figsize=(4.6, 4.8))
    _panel(ax)
    vp = ax.violinplot(
        [uv, fv], positions=[0, 1], widths=0.7,
        showmeans=False, showextrema=False, vert=True,
    )
    for body, color in zip(vp["bodies"], (BASE_C, TUNE_C)):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.28)
        body.set_linewidth(0.6)
    for x, data, color in ((0, uv, BASE_C), (1, fv, TUNE_C)):
        jitter = x + rng.normal(0, 0.04, size=len(data))
        ax.scatter(jitter, data, s=8, color=color, alpha=0.45, zorder=3, linewidths=0)
        q1, med, q3 = np.percentile(data, [25, 50, 75])
        ax.plot([x - 0.12, x + 0.12], [med, med], color="black", lw=1.4, zorder=4)
        ax.plot([x, x], [q1, q3], color="black", lw=1.0, zorder=4)
    ax.axhline(-4.0, color=RULE, lw=0.8, ls="--")
    ax.text(1.42, -4.0, "−4 kcal/mol\nthreshold", color=GRAY, fontsize=7, va="center")
    ax.set_xticks([0, 1], [BASE, TUNED_TICK])
    ax.set_ylabel("Vina (kcal/mol)")
    ax.set_xlim(-0.65, 1.7)
    ax.set_ylim(-12.5, 2.0)
    fig.tight_layout()
    _save(fig, "02_vina_distribution")


def fig_paired(u_rows: list[dict], f_rows: list[dict]) -> None:
    um, fm = _pocket_means(u_rows), _pocket_means(f_rows)
    keys = sorted(set(um) & set(fm))
    xs = np.array([um[k] for k in keys])
    ys = np.array([fm[k] for k in keys])
    fig, ax = plt.subplots(figsize=(4.7, 4.7))
    _panel(ax)
    lo = min(xs.min(), ys.min()) - 0.5
    hi = max(xs.max(), ys.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], color=RULE, lw=0.8, ls="--", zorder=1)
    improved = ys < xs
    ax.scatter(
        xs[improved], ys[improved], s=36, color=TUNE_C, zorder=3,
        edgecolors="black", linewidths=0.4, label=f"Improved ({improved.sum()})",
    )
    ax.scatter(
        xs[~improved], ys[~improved], s=36, color=BASE_C, zorder=3,
        edgecolors="black", linewidths=0.4, label=f"Not improved ({(~improved).sum()})",
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{BASE} mean Vina (kcal/mol)")
    ax.set_ylabel(f"{TUNED} mean Vina (kcal/mol)")
    ax.legend(loc="upper left", handletextpad=0.35, borderaxespad=0.4)
    order = np.argsort(ys - xs)
    for idx in list(order[:2]) + list(order[-1:]):
        ax.annotate(
            _short_pocket(keys[idx]), (xs[idx], ys[idx]),
            textcoords="offset points", xytext=(5, 3), fontsize=7, color="black",
        )
    fig.tight_layout()
    _save(fig, "03_per_pocket_paired")


def fig_vina_qed(u_rows: list[dict], f_rows: list[dict]) -> None:
    ux, uy = _qed_vina(u_rows)
    fx, fy = _qed_vina(f_rows)
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    _panel(ax)
    ax.scatter(uy, ux, s=16, alpha=0.7, color=BASE_C, label=BASE, zorder=2, linewidths=0)
    ax.scatter(fy, fx, s=16, alpha=0.7, color=TUNE_C, label=TUNED, zorder=3, linewidths=0)
    ax.axhline(-4.0, color=RULE, lw=0.8, ls="--")
    ax.set_xlabel("QED")
    ax.set_ylabel("Vina (kcal/mol)")
    ax.legend(loc="lower right")
    ax.set_ylim(-12.5, 3.2)
    fig.tight_layout()
    _save(fig, "04_vina_vs_qed")


def fig_training() -> None:
    log = json.loads((EVALS / "sufficient_vina" / "train_log.json").read_text(encoding="utf-8"))
    iters, valid, reward = [], [], []
    for rec in log:
        iters.append(int(rec["iteration"]))
        n_valid = sum(p["n_valid"] for p in rec["pockets"])
        n_mols = sum(len(p["rewards"]) for p in rec["pockets"])
        valid.append(100.0 * n_valid / max(n_mols, 1))
        reward.append(float(np.mean([p["mean_reward"] for p in rec["pockets"]])))
    iters = np.array(iters)
    valid = np.array(valid)
    reward = np.array(reward)

    def roll(x, w=10):
        out = np.empty_like(x, dtype=float)
        for i in range(len(x)):
            out[i] = x[max(0, i - w + 1): i + 1].mean()
        return out

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.2), sharex=True)
    specs = (
        (axes[0], valid, roll(valid), "a  Validity on sampled training groups (%)", BASE_C),
        (axes[1], reward, roll(reward), "b  Mean training reward", TUNE_C),
    )
    for ax, series, smooth, title, color in specs:
        _panel(ax)
        ax.set_title(title, loc="left", pad=4)
        ax.plot(iters, series, color="#B0B0B0", lw=0.7)
        ax.plot(iters, smooth, color=color, lw=1.6)
        ax.set_ylabel(title.split("  ", 1)[1] if "  " in title else title)
        ax.set_xlim(0, 99)
    axes[0].set_ylabel("Validity (%)")
    axes[1].set_ylabel("Mean reward")
    axes[1].set_xlabel("GRPO iteration (one pocket per iteration, group size 8)")
    fig.tight_layout()
    _save(fig, "05_training_curves")


def fig_ablation() -> None:
    runs = [
        ("sufficient", "Unclipped DHC"),
        ("sufficient_dualclip", "Dual-clip"),
        ("sufficient_vina", "Vina-gated"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.6))
    width = 0.36
    x = np.arange(len(runs))
    for ax, key, title, ylabel in (
        (axes[0], "mean_vina", "a  Mean Vina", "Vina (kcal/mol)"),
        (axes[1], "validity", "b  Validity", "Validity (%)"),
    ):
        _panel(ax)
        ax.set_title(title, loc="left", pad=4)
        u_vals = [_load_summary(run, "unguided")[key] for run, _ in runs]
        f_vals = [_load_summary(run, "ftdiff")[key] for run, _ in runs]
        if key == "validity":
            u_vals = [100.0 * v for v in u_vals]
            f_vals = [100.0 * v for v in f_vals]
        ax.bar(x - width / 2, u_vals, width, color=BASE_C, edgecolor="black",
               linewidth=0.5, label=BASE, zorder=3)
        ax.bar(x + width / 2, f_vals, width, color=TUNE_C, edgecolor="black",
               linewidth=0.5, label=TUNED, zorder=3)
        ax.set_xticks(x, [label for _, label in runs])
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")
        fmt = "{:.2f}" if key == "mean_vina" else "{:.1f}"
        for i, (u, f) in enumerate(zip(u_vals, f_vals)):
            if key == "mean_vina":
                ax.text(i - width / 2, u, fmt.format(u), ha="center", va="top", fontsize=7)
                ax.text(i + width / 2, f, fmt.format(f), ha="center", va="top", fontsize=7)
            else:
                ax.text(i - width / 2, u, fmt.format(u), ha="center", va="bottom", fontsize=7)
                ax.text(i + width / 2, f, fmt.format(f), ha="center", va="bottom", fontsize=7)
        if key == "mean_vina":
            lo = min(u_vals + f_vals)
            ax.set_ylim(lo - 0.9, 0.35)
            ax.axhline(0, color="black", lw=0.6)
        else:
            ax.set_ylim(0, 112)
    fig.tight_layout()
    _save(fig, "06_three_run_ablation")


def fig_board() -> None:
    paths = [
        OUT / "01_summary_bars.png",
        OUT / "02_vina_distribution.png",
        OUT / "03_per_pocket_paired.png",
        OUT / "04_vina_vs_qed.png",
        OUT / "05_training_curves.png",
        OUT / "06_three_run_ablation.png",
    ]
    fig = plt.figure(figsize=(11.5, 7.6), facecolor="white")
    fig.suptitle("GRPO fine-tuning of DiffSBDD", fontsize=12, color="black", y=0.98)
    for i, path in enumerate(paths):
        ax = fig.add_subplot(2, 3, i + 1)
        ax.imshow(plt.imread(path))
        ax.axis("off")
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.95))
    fig.savefig(OUT / "00_all_figures.png", dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main() -> None:
    _style()
    u_rows = _load_mols("sufficient_vina", "unguided")
    f_rows = _load_mols("sufficient_vina", "ftdiff")
    fig_summary(_load_summary("sufficient_vina", "unguided"), _load_summary("sufficient_vina", "ftdiff"))
    fig_vina_dist(u_rows, f_rows)
    fig_paired(u_rows, f_rows)
    fig_vina_qed(u_rows, f_rows)
    fig_training()
    fig_ablation()
    fig_board()
    print(f"Wrote figures -> {OUT}")
    for path in sorted(OUT.glob("*.png")):
        print(" ", path.name)


if __name__ == "__main__":
    main()
