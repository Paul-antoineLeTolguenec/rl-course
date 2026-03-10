"""
markov_env — Controlled Markov chain API for the RL course notebook.

Public API
----------
Policies:
    spiral_ccw()  spiral_cw()  rightward()

Chain:
    induced_chain(pi, alpha=1.0, eps=0.05) -> P^pi  (N×N)

Analysis:
    stationary(P)               -> mu_inf
    tv_curve(P, n_steps=150)    -> TV(mu_t, mu_inf) over time
    occupation(P, gamma)        -> d^pi_gamma

Visualisation (return go.Figure):
    show_grid(mu=None, P=None, title='')
    animate(P, n_frames=80)
    rho_animation(P, n_frames=120)
    mixing_plot(policies: dict[str, ndarray])
    occupation_plot(P, gammas=(0.5, 0.9, 0.99))
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff

# ── Grid constants ────────────────────────────────────────────────────────────
GRID = 20
N    = GRID * GRID
ACTIONS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
A_DIM   = len(ACTIONS)

WALL_CELLS  = frozenset([(i,9) for i in range(2,6)] + [(i,9) for i in range(14,18)])
WALL_STATES = frozenset(i*GRID+j for i,j in WALL_CELLS)

# ── Colour palette ────────────────────────────────────────────────────────────
_DARK   = "#070b14"
_FG     = "#e2e8f0"
_MUTED  = "#7a8fa6"
_ACCENT = "#00d4ff"
_VIOLET = "#a78bfa"
_WARM   = "#ffc850"

_DIST_CS = [[0, _DARK],              [1, _ACCENT]]
_WALL_CS = [[0, "rgba(0,0,0,0)"],    [1, "rgba(167,139,250,0.85)"]]
_POLICY_COLORS = [_ACCENT, _VIOLET, _WARM]

# ── Kernel (built once) ───────────────────────────────────────────────────────
def _build_kernel() -> np.ndarray:
    P = np.zeros((N, N, A_DIM))
    for i in range(GRID):
        for j in range(GRID):
            s = i*GRID + j
            if s in WALL_STATES:
                P[s, s, :] = 1.0; continue
            for a, (di, dj) in enumerate(ACTIONS):
                ni, nj = i+di, j+dj
                if 0 <= ni < GRID and 0 <= nj < GRID and (ni,nj) not in WALL_CELLS:
                    P[s, ni*GRID+nj, a] = 1.0
                else:
                    P[s, s, a] = 1.0
    return P

_P_SA: np.ndarray | None = None

def _kernel() -> np.ndarray:
    global _P_SA
    if _P_SA is None:
        _P_SA = _build_kernel()
    return _P_SA


# ── Policies ──────────────────────────────────────────────────────────────────
def _spiral(cw: bool = False, temp: float = 2.5) -> np.ndarray:
    cx = cy = 9.5
    pi = np.zeros((N, A_DIM))
    for i in range(GRID):
        for j in range(GRID):
            s = i*GRID + j
            if s in WALL_STATES: pi[s] = 1.0/A_DIM; continue
            out = np.array([j-cx, i-cy], float)
            d = np.linalg.norm(out)
            if d < 1.5: pi[s] = 1.0/A_DIM; continue
            o = out/d
            t = np.array([o[1],-o[0]] if cw else [-o[1],o[0]])
            drift = 0.7*t + 0.5*(-o); drift /= np.linalg.norm(drift)
            w = np.array([np.exp(np.dot([dj,di]/np.linalg.norm([dj,di]), drift)/temp)
                          for di,dj in ACTIONS])
            pi[s] = w/w.sum()
    return pi

def spiral_ccw() -> np.ndarray:
    """Counter-clockwise spiral policy."""
    return _spiral(cw=False)

def spiral_cw() -> np.ndarray:
    """Clockwise spiral policy."""
    return _spiral(cw=True)

def rightward() -> np.ndarray:
    """Always-right policy."""
    pi = np.zeros((N, A_DIM))
    pi[:, 4] = 1.0  # action (0,+1)
    for s in WALL_STATES: pi[s] = 1.0/A_DIM
    return pi


# ── Chain ─────────────────────────────────────────────────────────────────────
def induced_chain(pi: np.ndarray, alpha: float = 0.2, eps: float = 0.05) -> np.ndarray:
    """Compute P^pi.

    P^pi(s'|s) = (1-eps) * [(1-alpha)*pi_vortex + alpha*pi](a|s) * P(s'|s,a)  +  eps * uniform

    alpha=1  → pure policy pi
    alpha=0  → pure CCW vortex (base dynamics)
    """
    P_sa   = _kernel()
    pi_eff = (1 - alpha) * _spiral() + alpha * pi
    P_ctrl = np.einsum("sa,sna->sn", pi_eff, P_sa)
    P_unif = np.zeros((N, N))
    for i in range(GRID):
        for j in range(GRID):
            s = i*GRID + j
            if s in WALL_STATES: P_unif[s, s] = 1.0; continue
            nb = [s] + [(i+di)*GRID+(j+dj) for di,dj in ACTIONS
                        if 0<=i+di<GRID and 0<=j+dj<GRID and (i+di,j+dj) not in WALL_CELLS]
            for ns in nb: P_unif[s, ns] = 1.0/len(nb)
    return (1 - eps)*P_ctrl + eps*P_unif


# ── Analysis ──────────────────────────────────────────────────────────────────
def stationary(P: np.ndarray, n_iter: int = 2000) -> np.ndarray:
    """Stationary distribution mu_inf via power iteration."""
    mu, PT = np.ones(N)/N, P.T
    for _ in range(n_iter): mu = PT @ mu; mu /= mu.sum()
    return mu

def tv_curve(P: np.ndarray, n_steps: int = 150) -> np.ndarray:
    """TV distance ||mu_t - mu_inf||_1 / 2 from a Dirac at s0=(19,0)."""
    mu_inf = stationary(P)
    mu, PT = np.zeros(N), P.T
    mu[19*GRID] = 1.0
    tv = []
    for _ in range(n_steps):
        tv.append(0.5 * np.abs(mu - mu_inf).sum())
        mu = PT @ mu
    return np.array(tv)

def occupation(P: np.ndarray, gamma: float, tol: float = 1e-6) -> np.ndarray:
    """Discounted occupation measure d^pi_gamma from s0=(19,0).
    Runs until gamma^t < tol (adaptive n_steps)."""
    n_steps = int(np.ceil(np.log(tol) / np.log(gamma)))
    mu, PT, d, w = np.zeros(N), P.T, np.zeros(N), 1.0
    mu[19*GRID] = 1.0
    for _ in range(n_steps):
        d += w*mu; mu = PT @ mu; w *= gamma
    return (1 - gamma) * d


# ── Visualisation helpers ─────────────────────────────────────────────────────
def _wall_z() -> np.ndarray:
    z = np.zeros((GRID, GRID))
    for wi, wj in WALL_CELLS: z[wi, wj] = 1.0
    return z

def _mean_arrows(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    jj = np.arange(N) % GRID
    ii = np.arange(N) // GRID
    U = (P @ jj - jj).reshape(GRID, GRID)
    V = (P @ ii - ii).reshape(GRID, GRID)
    norm = np.sqrt(U**2 + V**2).clip(min=1e-8)
    U, V = U/norm, V/norm
    for wi, wj in WALL_CELLS: U[wi,wj] = V[wi,wj] = 0.0
    return U, V

def _quiver(U: np.ndarray, V: np.ndarray) -> list:
    jj_g, ii_g = np.meshgrid(np.arange(GRID), np.arange(GRID))
    mask = np.ones((GRID, GRID), bool)
    for wi, wj in WALL_CELLS: mask[wi, wj] = False
    q = ff.create_quiver(
        jj_g[mask].ravel(), ii_g[mask].ravel(),
        U[mask].ravel(), V[mask].ravel(),
        scale=0.38, arrow_scale=0.22,
        line=dict(color=_VIOLET, width=0.9)
    )
    for t in q.data: t.showlegend = False; t.hoverinfo = "skip"
    return list(q.data)

def _grid_layout(title: str = "", w: int = 420, h: int = 420) -> dict:
    return dict(
        title=dict(text=title, font_color=_FG),
        paper_bgcolor=_DARK, plot_bgcolor=_DARK, font_color=_FG,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   scaleanchor="x", autorange="reversed"),
        width=w, height=h, margin=dict(l=5, r=5, t=40, b=5)
    )


# ── Public visualisation ──────────────────────────────────────────────────────
def show_grid(mu: np.ndarray | None = None, P: np.ndarray | None = None,
              title: str = "") -> go.Figure:
    """Heatmap of distribution mu and/or mean-drift arrows from P."""
    traces = []
    if mu is not None:
        traces.append(go.Heatmap(z=mu.reshape(GRID,GRID), colorscale=_DIST_CS,
                                  showscale=False, zmin=0, hoverinfo="skip"))
    traces.append(go.Heatmap(z=_wall_z(), colorscale=_WALL_CS,
                              showscale=False, zmin=0, zmax=1, hoverinfo="skip"))
    if P is not None:
        traces += _quiver(*_mean_arrows(P))
    fig = go.Figure(data=traces)
    fig.update_layout(**_grid_layout(title))
    return fig


def animate(P: np.ndarray, n_frames: int = 80) -> go.Figure:
    """Animation of mu_t evolving under P^pi, starting from delta_{s0}."""
    PT  = P.T
    mu0 = np.zeros(N); mu0[19*GRID] = 1.0
    mu  = mu0.copy()

    frames = []
    for i in range(n_frames):
        frames.append(go.Frame(
            data=[go.Heatmap(z=mu.reshape(GRID,GRID), colorscale=_DIST_CS,
                             showscale=False, zmin=0, hoverinfo="skip")],
            traces=[0], name=str(i)
        ))
        mu = PT @ mu

    fig = go.Figure(
        data=[
            go.Heatmap(z=mu0.reshape(GRID,GRID), colorscale=_DIST_CS,
                       showscale=False, zmin=0, hoverinfo="skip"),
            go.Heatmap(z=_wall_z(), colorscale=_WALL_CS,
                       showscale=False, zmin=0, zmax=1, hoverinfo="skip"),
        ],
        frames=frames
    )
    fig.update_layout(
        **_grid_layout(),
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.5, xanchor="center", y=-0.05,
            buttons=[dict(label="Play", method="animate",
                          args=[None, dict(frame=dict(duration=70, redraw=True),
                                          fromcurrent=True)])]
        )]
    )
    return fig


def rho_animation(P: np.ndarray, n_frames: int = 120) -> go.Figure:
    """Animation of the running average rho_T = (1/T) sum_{t=0}^{T-1} mu_t."""
    PT  = P.T
    mu0 = np.zeros(N); mu0[19*GRID] = 1.0
    mu  = mu0.copy()
    rho = np.zeros(N)

    frames = []
    for t in range(1, n_frames + 1):
        rho = rho + (mu - rho) / t  # online mean
        frames.append(go.Frame(
            data=[go.Heatmap(z=rho.reshape(GRID,GRID), colorscale=_DIST_CS,
                             showscale=False, zmin=0, hoverinfo="skip")],
            traces=[0], name=str(t)
        ))
        mu = PT @ mu

    fig = go.Figure(
        data=[
            go.Heatmap(z=mu0.reshape(GRID,GRID), colorscale=_DIST_CS,
                       showscale=False, zmin=0, hoverinfo="skip"),
            go.Heatmap(z=_wall_z(), colorscale=_WALL_CS,
                       showscale=False, zmin=0, zmax=1, hoverinfo="skip"),
        ],
        frames=frames
    )
    fig.update_layout(
        **_grid_layout("Occupation measure ρ_T → μ_∞"),
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.5, xanchor="center", y=-0.05,
            buttons=[dict(label="Play", method="animate",
                          args=[None, dict(frame=dict(duration=60, redraw=True),
                                          fromcurrent=True)])]
        )]
    )
    return fig


def mixing_plot(policies: dict[str, np.ndarray]) -> go.Figure:
    """TV distance ||mu_t - mu_inf||_1/2 for each policy."""
    fig = go.Figure()
    for (name, pi), color in zip(policies.items(), _POLICY_COLORS):
        P = induced_chain(pi)
        fig.add_trace(go.Scatter(y=tv_curve(P), mode="lines", name=name,
                                 line=dict(color=color, width=2)))
    fig.update_layout(
        paper_bgcolor=_DARK, plot_bgcolor="#0d1220", font_color=_FG,
        xaxis=dict(title="step t", color=_MUTED, gridcolor="#1e2a3a"),
        yaxis=dict(title="TV(μ_t, μ_∞)", color=_MUTED, gridcolor="#1e2a3a"),
        title=dict(text="Mixing time", font_color=_FG),
        legend=dict(bgcolor="#0d1220", bordercolor="#1e2a3a"),
        width=700, height=400, margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig


def occupation_plot(P: np.ndarray,
                    gammas: tuple[float, ...] = (0.5, 0.9, 0.99)) -> go.Figure:
    """d^pi_gamma for each gamma value, side by side."""
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=len(gammas),
                        subplot_titles=[f"γ = {g}" for g in gammas])
    wall = _wall_z()
    for col, g in enumerate(gammas, 1):
        d = occupation(P, g)
        fig.add_trace(go.Heatmap(z=d.reshape(GRID,GRID), colorscale=_DIST_CS,
                                  showscale=False, zmin=0, hoverinfo="skip"), row=1, col=col)
        fig.add_trace(go.Heatmap(z=wall, colorscale=_WALL_CS,
                                  showscale=False, zmin=0, zmax=1, hoverinfo="skip"), row=1, col=col)
        ax = "" if col == 1 else str(col)
        fig.update_layout(**{
            f"xaxis{ax}": dict(showticklabels=False, showgrid=False, zeroline=False),
            f"yaxis{ax}": dict(showticklabels=False, showgrid=False, zeroline=False,
                               autorange="reversed"),
        })
    fig.update_layout(
        paper_bgcolor=_DARK, plot_bgcolor=_DARK, font_color=_FG,
        title=dict(text="Discounted occupation measure d^π_γ", font_color=_FG),
        width=1100, height=420, margin=dict(l=5, r=5, t=50, b=5)
    )
    return fig
