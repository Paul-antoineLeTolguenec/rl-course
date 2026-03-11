"""Generate Bellman operator convergence GIFs for two policies."""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image
import io

# Allow importing markov_env from this directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from practical_case.markov_env import (
    induced_chain, spiral_ccw, rightward, WALL_STATES, GRID, N, _mean_arrows
)

# ── Config ────────────────────────────────────────────────────────────────────
GAMMA      = 0.95
N_ITER     = 60
FIG_W_PX   = 900
FIG_H_PX   = 360
DPI        = 100
FRAME_MS   = 80
DARK_BG    = "#070b14"
WALL_RGBA  = (167/255, 139/255, 250/255, 0.7)

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── Reward: goal at top-right corner ─────────────────────────────────────────
def build_reward() -> np.ndarray:
    R = np.zeros(N)
    for idx in range(N):
        i, j = divmod(idx, GRID)
        if i <= 2 and j >= GRID - 3:
            R[idx] = 1.0
    return R

# ── Wall mask for overlay (GRID x GRID) ──────────────────────────────────────
def wall_mask() -> np.ndarray:
    z = np.zeros((GRID, GRID))
    for s in WALL_STATES:
        i, j = divmod(s, GRID)
        z[i, j] = 1.0
    return z

_WALL_MASK = wall_mask()

# ── Single frame rendering ────────────────────────────────────────────────────
def render_frame(
    Vs: list[np.ndarray],
    V_star: np.ndarray,
    labels: list[str],
    policy_name: str,
    k: int,
    vmin: float,
    vmax: float,
    arrows: tuple[np.ndarray, np.ndarray] | None = None,
) -> Image.Image:
    fig, axes = plt.subplots(
        1, 2,
        figsize=(FIG_W_PX / DPI, FIG_H_PX / DPI),
        dpi=DPI,
        facecolor=DARK_BG,
    )
    fig.suptitle(
        f"{policy_name} — iteration k = {k}",
        color="#e2e8f0", fontsize=11, y=0.98,
    )

    for ax, V, label in zip(axes, Vs, labels):
        ax.set_facecolor(DARK_BG)
        grid = V.reshape(GRID, GRID)

        # Value heatmap
        im = ax.imshow(
            grid,
            cmap="plasma",
            vmin=vmin, vmax=vmax,
            origin="upper",
            interpolation="nearest",
            aspect="equal",
        )

        # Policy vector field
        if arrows is not None:
            U, V_arr = arrows
            xs = np.arange(GRID)
            ys = np.arange(GRID)
            mask = _WALL_MASK == 0
            ax.quiver(
                xs[np.newaxis, :].repeat(GRID, 0)[mask],
                ys[:, np.newaxis].repeat(GRID, 1)[mask],
                U[mask], -V_arr[mask],  # flip V for image coords (origin=upper)
                color="#a78bfa", alpha=0.45,
                scale=28, width=0.004, headwidth=3, headlength=3,
            )

        # Wall overlay
        wall_rgba_arr = np.zeros((GRID, GRID, 4))
        wall_rgba_arr[..., 0] = WALL_RGBA[0]
        wall_rgba_arr[..., 1] = WALL_RGBA[1]
        wall_rgba_arr[..., 2] = WALL_RGBA[2]
        wall_rgba_arr[..., 3] = _WALL_MASK * WALL_RGBA[3]
        ax.imshow(wall_rgba_arr, origin="upper", aspect="equal", interpolation="nearest")

        # Error text
        err = np.max(np.abs(V - V_star))
        ax.text(
            0.03, 0.03,
            f"‖Vₖ − V*‖∞ = {err:.3f}",
            transform=ax.transAxes,
            color="#e2e8f0", fontsize=7,
            verticalalignment="bottom",
            bbox=dict(facecolor=DARK_BG, alpha=0.6, edgecolor="none", pad=1),
        )

        ax.set_title(label, color="#e2e8f0", fontsize=9, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor=DARK_BG, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ── Main GIF generation ───────────────────────────────────────────────────────
def generate_gif(policy_name: str, pi: np.ndarray, out_path: str) -> None:
    print(f"  Building P^π for {policy_name}…")
    P_pi = induced_chain(pi, alpha=0.4, eps=0.05)

    R = build_reward()

    print(f"  Solving for V* ({policy_name})…")
    V_star = np.linalg.solve(np.eye(N) - GAMMA * P_pi, R)

    vmin, vmax = V_star.min(), V_star.max()
    arrows = _mean_arrows(P_pi)

    # Three initializations
    rng = np.random.default_rng(42)
    inits = [
        np.zeros(N),
        rng.uniform(0, 10, N),
    ]
    labels = ["V₀ = 0", "V₀ ~ U[0, 10]"]
    Vs = [v.copy() for v in inits]

    frames: list[Image.Image] = []
    print(f"  Rendering {N_ITER} frames…")
    for k in range(N_ITER + 1):
        frame = render_frame(Vs, V_star, labels, policy_name, k, vmin, vmax, arrows)
        frames.append(frame)
        # Bellman step
        Vs = [R + GAMMA * P_pi @ V for V in Vs]

    # Save as GIF
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_MS,
        loop=0,
        optimize=False,
    )
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    policies = {
        "spiral_ccw": (spiral_ccw(), os.path.join(ASSETS_DIR, "value_ccw.gif")),
        "rightward":  (rightward(),  os.path.join(ASSETS_DIR, "value_rightward.gif")),
    }
    for name, (pi, path) in policies.items():
        print(f"\nGenerating GIF for policy: {name}")
        generate_gif(name, pi, path)

    print("\nDone.")
