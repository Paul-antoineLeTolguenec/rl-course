"""
MDP = Controlled Markov Chain + reward signal.

Same dynamics as controlled_mc, but with:
  - a fixed start cell (green, bottom-left)
  - a fixed goal cell (gold, top-right)
  - a reward flash on the goal cell proportional to probability mass there
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.insert(0, os.path.dirname(__file__))
from markov_chain import (
    N, A_DIM, GRID, WALL_CELLS, WALL_STATES, ACTIONS,
    build_deterministic_kernel,
    build_transition_matrix,
    compute_mean_arrows,
    cmap,
)
from controlled_markov_chain import rightward_policy

OUTPUT_PATH = "assets/practical_case/mdp.gif"

START_CELL = (19, 0)   # bottom-left  (row, col)
GOAL_CELL  = (0, 19)   # top-right

START_STATE = START_CELL[0] * GRID + START_CELL[1]
GOAL_STATE  = GOAL_CELL[0]  * GRID + GOAL_CELL[1]


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

    # Initial distribution: Dirac at start cell
    p = np.zeros(N)
    p[START_STATE] = 1.0

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

    # Wall overlay
    wall_rgba = np.zeros((GRID, GRID, 4), dtype=float)
    for wi, wj in WALL_CELLS:
        wall_rgba[wi, wj] = [0.655, 0.545, 0.980, 0.85]
    ax.imshow(wall_rgba, origin="upper", interpolation="nearest")

    # Start cell overlay (blue-ish, neutral)
    start_rgba = np.zeros((GRID, GRID, 4), dtype=float)
    start_rgba[START_CELL[0], START_CELL[1]] = [0.239, 0.510, 0.965, 1.0]  # #3d82f6
    ax.imshow(start_rgba, origin="upper", interpolation="nearest")

    # Goal cell overlay (green = reward) — dynamic alpha to flash with reward
    goal_rgba = np.zeros((GRID, GRID, 4), dtype=float)
    goal_rgba[GOAL_CELL[0], GOAL_CELL[1]] = [0.133, 0.773, 0.369, 1.0]  # #22c55e
    goal_im = ax.imshow(goal_rgba, origin="upper", interpolation="nearest")

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

        # Reward flash: alpha of goal cell scales with probability mass there
        goal_mass = float(p[GOAL_STATE])
        base_alpha = 0.85
        flash_alpha = min(1.0, base_alpha + goal_mass * 6.0)
        new_goal = np.zeros((GRID, GRID, 4), dtype=float)
        new_goal[GOAL_CELL[0], GOAL_CELL[1]] = [0.133, 0.773, 0.369, flash_alpha]
        goal_im.set_data(new_goal)

        p[:] = PT @ p
        return [im, goal_im]

    print("Rendering animation (120 frames) …")
    anim = animation.FuncAnimation(
        fig, update, frames=120, interval=1000 // 15, blit=True, repeat=True
    )
    anim.save(OUTPUT_PATH, writer=animation.PillowWriter(fps=15))
    plt.close(fig)

    size = os.path.getsize(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH} ({size:,} bytes)")


if __name__ == "__main__":
    main()
