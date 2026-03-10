"""
Markov chain vortex dynamics on a 20×20 grid.

Architecture:
  P(s'|s,a) : deterministic kernel  (N × N × A)
  π(a|s)    : policy                (N × A)
  Pᵖ(s'|s)  = (1-ε) Σₐ π(a|s) P(s'|s,a)  +  ε · uniform
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

# ── Grid / model constants ──────────────────────────────────────────────────
GRID       = 20
N          = GRID * GRID
TEMPERATURE = 2.5
EPSILON     = 0.05
ALPHA_SPR   = 0.7   # CCW tangent weight
BETA_SPR    = 0.5   # inward-radial weight
CTR_THRESH  = 1.5   # uniform zone near center
OUTPUT_PATH = "assets/practical_case/mc_dynamics.gif"

# ── Action space: 8 directions (di, dj) ────────────────────────────────────
ACTIONS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
A_DIM   = len(ACTIONS)

# ── Wall cells (impassable) ─────────────────────────────────────────────────
#   top wall:    j=9, i=2..5   (passage above: i=0,1)
#   bottom wall: j=9, i=14..17 (passage below: i=18,19)
WALL_CELLS  = frozenset(
    [(i, 9) for i in range(2, 6)] +
    [(i, 9) for i in range(14, 18)]
)
WALL_STATES = frozenset(i * GRID + j for (i, j) in WALL_CELLS)

cmap = LinearSegmentedColormap.from_list("rl", ["#070b14", "#00d4ff"])


# ── Kernels ─────────────────────────────────────────────────────────────────

def build_deterministic_kernel() -> np.ndarray:
    """P(s'|s,a) of shape (N, N, A).

    Action a from state s → go to s+a if in-bounds and not a wall, else self-loop.
    """
    P_sa = np.zeros((N, N, A_DIM))
    for i in range(GRID):
        for j in range(GRID):
            s = i * GRID + j
            if s in WALL_STATES:
                P_sa[s, s, :] = 1.0
                continue
            for a_idx, (di, dj) in enumerate(ACTIONS):
                ni, nj = i + di, j + dj
                if 0 <= ni < GRID and 0 <= nj < GRID and (ni, nj) not in WALL_CELLS:
                    P_sa[s, ni * GRID + nj, a_idx] = 1.0
                else:
                    P_sa[s, s, a_idx] = 1.0   # self-loop
    return P_sa


def spiral_policy() -> np.ndarray:
    """Vortex spiral policy π(a|s) of shape (N, A).

    Softmax over (CCW-tangent + inward-radial) drift vector.
    """
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
                tangent = np.array([-o_norm[1], o_norm[0]])   # CCW in screen coords
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


def build_transition_matrix(
    policy:  np.ndarray,
    P_sa:    np.ndarray,
    epsilon: float = EPSILON,
    alpha:   float = 1.0,
) -> np.ndarray:
    """Pᵖ(s'|s) = (1-ε) Σₐ π_eff(a|s) P(s'|s,a)  +  ε · uniform.

    Args:
        policy:  (N, A) agent policy — rows sum to 1
        P_sa:    (N, N, A) deterministic kernel
        epsilon: ergodicity mixing weight
        alpha:   blend weight — π_eff = (1-α)·π_vortex + α·policy
                 (α=1 → pure agent, α=0 → pure vortex)
    Returns:
        P: (N, N) transition matrix — rows sum to 1
    """
    pi_base = spiral_policy()
    pi_eff  = (1 - alpha) * pi_base + alpha * policy

    # Controlled kernel: P_ctrl[s,s'] = Σ_a π_eff[s,a] · P_sa[s,s',a]
    P_ctrl = np.einsum('sa,sna->sn', pi_eff, P_sa)

    # Uniform over valid (non-wall) neighbors + self
    P_unif = np.zeros((N, N))
    for i in range(GRID):
        for j in range(GRID):
            s = i * GRID + j
            if s in WALL_STATES:
                P_unif[s, s] = 1.0
                continue
            valid = [s] + [
                (i + di) * GRID + (j + dj)
                for di, dj in ACTIONS
                if 0 <= i + di < GRID and 0 <= j + dj < GRID
                and (i + di, j + dj) not in WALL_CELLS
            ]
            for ns in valid:
                P_unif[s, ns] = 1.0 / len(valid)

    return (1 - epsilon) * P_ctrl + epsilon * P_unif


# ── Helpers ─────────────────────────────────────────────────────────────────

def compute_mean_arrows(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized mean displacement (U=dx col, V=dy row) per state."""
    jj = np.arange(N) % GRID
    ii = np.arange(N) // GRID
    return P @ jj - jj, P @ ii - ii   # E[j']-j, E[i']-i


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("Building kernels …")
    P_sa = build_deterministic_kernel()
    pi   = spiral_policy()

    print("Building transition matrix …")
    P  = build_transition_matrix(pi, P_sa)
    PT = P.T   # p_{t+1} = PT @ p_t

    print("Computing mean arrows …")
    U_flat, V_flat = compute_mean_arrows(P)
    U_grid = U_flat.reshape(GRID, GRID)
    V_grid = V_flat.reshape(GRID, GRID)

    norm = np.sqrt(U_grid**2 + V_grid**2)
    norm[norm < 1e-8] = 1.0
    U_norm =  U_grid / norm
    V_norm =  V_grid / norm

    # Zero arrows on wall cells
    for wi, wj in WALL_CELLS:
        U_norm[wi, wj] = V_norm[wi, wj] = 0.0

    jj_g, ii_g = np.meshgrid(np.arange(GRID), np.arange(GRID))

    # Initial distribution: Dirac at bottom-left
    p = np.zeros(N)
    p[19 * GRID + 0] = 1.0

    # ── Figure ───────────────────────────────────────────────────────────────
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

    # Wall overlay: #a78bfa with transparency on non-wall pixels
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
    animation.PillowWriter(fps=15)
    anim.save(OUTPUT_PATH, writer=animation.PillowWriter(fps=15))
    plt.close(fig)

    size = os.path.getsize(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH} ({size:,} bytes)")


if __name__ == "__main__":
    main()
