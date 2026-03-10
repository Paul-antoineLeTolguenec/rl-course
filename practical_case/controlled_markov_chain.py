"""
Controlled Markov chain on a 20×20 grid — random (uniform) policy.

Reuses kernels from markov_chain.py to show that P^π depends on π.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.insert(0, os.path.dirname(__file__))
from markov_chain import (
    N, A_DIM, GRID, WALL_CELLS, WALL_STATES, ACTIONS,
    ALPHA_SPR, BETA_SPR, CTR_THRESH, TEMPERATURE,
    build_deterministic_kernel,
    build_transition_matrix,
    compute_mean_arrows,
    cmap,
)

OUTPUT_PATH = "assets/practical_case/controlled_mc.gif"


def rightward_policy() -> np.ndarray:
    """Policy that always goes right — action (0,+1)."""
    # ACTIONS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    # index 4 = (0, 1) = right
    pi = np.zeros((N, A_DIM))
    pi[:, 4] = 1.0
    for s in WALL_STATES:
        pi[s, :] = 1.0 / A_DIM
    return pi


def clockwise_policy() -> np.ndarray:
    """Clockwise spiral policy — mirror of the CCW spiral in markov_chain.py."""
    cx = cy = 9.5
    pi = np.zeros((N, A_DIM))
    for i in range(GRID):
        for j in range(GRID):
            s = i * GRID + j
            if s in WALL_STATES:
                pi[s, :] = 1.0 / A_DIM
                continue
            outward = np.array([j - cx, i - cy], dtype=float)
            dist    = np.linalg.norm(outward)
            if dist < CTR_THRESH:
                weights = np.ones(A_DIM)
            else:
                o_norm  = outward / dist
                tangent = np.array([o_norm[1], -o_norm[0]])   # CW in screen coords
                drift   = ALPHA_SPR * tangent + BETA_SPR * (-o_norm)
                drift  /= np.linalg.norm(drift)
                weights = np.array([
                    np.exp(
                        np.dot(np.array([dj, di], dtype=float) / np.linalg.norm([dj, di]), drift)
                        / TEMPERATURE
                    )
                    for (di, dj) in ACTIONS
                ])
            pi[s, :] = weights / weights.sum()
    return pi


def main() -> None:
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("Building kernels …")
    P_sa = build_deterministic_kernel()
    pi   = rightward_policy()

    print("Building transition matrix …")
    P  = build_transition_matrix(pi, P_sa, alpha=0.2)
    PT = P.T

    print("Computing mean arrows …")
    U_flat, V_flat = compute_mean_arrows(P)
    U_grid = U_flat.reshape(GRID, GRID)
    V_grid = V_flat.reshape(GRID, GRID)

    norm = np.sqrt(U_grid**2 + V_grid**2)
    norm[norm < 1e-8] = 1.0
    U_norm =  U_grid / norm
    V_norm =  V_grid / norm

    for wi, wj in WALL_CELLS:
        U_norm[wi, wj] = V_norm[wi, wj] = 0.0

    jj_g, ii_g = np.meshgrid(np.arange(GRID), np.arange(GRID))

    # Initial distribution: Dirac at bottom-left
    p = np.zeros(N)
    p[19 * GRID + 0] = 1.0

    fig, ax = plt.subplots(figsize=(6, 6), dpi=80)
    fig.patch.set_facecolor("#070b14")
    ax.set_facecolor("#070b14")
    ax.axis("off")

    im = ax.imshow(
        p.reshape(GRID, GRID),
        origin="upper",
        cmap=cmap,
        vmin=0,
        vmax=max(p.max(), 1e-6),
        interpolation="nearest",
    )

    wall_rgba = np.zeros((GRID, GRID, 4), dtype=float)
    for wi, wj in WALL_CELLS:
        wall_rgba[wi, wj] = [0.655, 0.545, 0.980, 0.85]
    ax.imshow(wall_rgba, origin="upper", interpolation="nearest")

    ax.quiver(
        jj_g, ii_g, U_norm, -V_norm,
        color="#a78bfa",
        alpha=0.7,
        scale=2,
        scale_units="xy",
        width=0.003,
        headwidth=4,
        headlength=5,
    )

    plt.tight_layout(pad=0)

    def update(frame: int):
        nonlocal p
        im.set_data(p.reshape(GRID, GRID))
        im.set_clim(vmin=0, vmax=max(p.max(), 1e-6))
        p[:] = PT @ p
        return [im]

    print("Rendering animation (100 frames) …")
    anim = animation.FuncAnimation(
        fig, update, frames=100, interval=1000 // 15, blit=True, repeat=True
    )
    anim.save(OUTPUT_PATH, writer=animation.PillowWriter(fps=15))
    plt.close(fig)

    size = os.path.getsize(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH} ({size:,} bytes)")


if __name__ == "__main__":
    main()
