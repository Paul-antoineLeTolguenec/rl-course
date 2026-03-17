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

Reward:
    GOAL_CELL, GOAL_STATE
    default_reward()            -> r  (N,)

Dynamic Programming:
    policy_eval(pi, r, gamma)   -> V^pi  (N,)
    greedy_policy(V, r, gamma)  -> pi    (N, A_DIM)
    policy_iteration(r, gamma)  -> [(pi_k, V_k)]
    value_iteration(r, gamma)   -> [V_0, V_1, ..., V_*]

Visualisation (DP):
    show_value(V, pi=None, title='')
    vi_pi_convergence(vi_hist, pi_hist, V_star)

Model-Free:
    td_eval(pi, r, gamma, ...)                       -> [(ep, V_snapshot)]
    td_convergence(pi, r, gamma, V_ref, alphas, ...) -> {label: error_curve}
    sarsa(r, gamma, ...)                             -> (Q, episode_returns)
    q_learning(r, gamma, ...)                        -> (Q, episode_returns)
    policy_from_q(Q)                                 -> pi  (N, A_DIM)
    td_lambda_convergence(pi, r, gamma, V_ref, ...)  -> {label: error_curve}

Visualisation (Model-Free):
    show_td_snapshots(snapshots, V_true)
    show_mf_convergence(curves, ylabel, title, smooth)
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


def tv_incremental_curve(P: np.ndarray, n_steps: int = 150) -> np.ndarray:
    """TV distance ||mu_{t+1} - mu_t||_1 / 2 from a Dirac at s0=(19,0)."""
    mu, PT = np.zeros(N), P.T
    mu[19*GRID] = 1.0
    tv = []
    for _ in range(n_steps):
        mu_next = PT @ mu
        tv.append(0.5 * np.abs(mu_next - mu).sum())
        mu = mu_next
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
    """Two subplots: TV(mu_t, mu_inf) and TV(mu_{t+1}, mu_t) for each policy."""
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["TV(μ<sub>t</sub>, μ<sub>∞</sub>)",
                                        "TV(μ<sub>t+1</sub>, μ<sub>t</sub>)"])
    for (name, pi), color in zip(policies.items(), _POLICY_COLORS):
        P = induced_chain(pi)
        fig.add_trace(go.Scatter(y=tv_curve(P), mode="lines", name=name,
                                 line=dict(color=color, width=2)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(y=tv_incremental_curve(P), mode="lines",
                                 name=name, showlegend=False,
                                 line=dict(color=color, width=2)),
                      row=1, col=2)
    fig.update_xaxes(title_text="step t", color=_MUTED, gridcolor="#1e2a3a")
    fig.update_yaxes(color=_MUTED, gridcolor="#1e2a3a")
    fig.update_layout(
        paper_bgcolor=_DARK, plot_bgcolor="#0d1220", font_color=_FG,
        title=dict(text="Mixing time", font_color=_FG),
        legend=dict(bgcolor="#0d1220", bordercolor="#1e2a3a"),
        width=900, height=400, margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig


_VALUE_CS = [[0.0, "#070b14"], [0.4, "#0d3d6b"], [0.75, "#f59e0b"], [1.0, "#fffbeb"]]


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


# ── Goal & reward ──────────────────────────────────────────────────────────────
GOAL_CELL  = (2, 17)
GOAL_STATE = GOAL_CELL[0] * GRID + GOAL_CELL[1]


def default_reward() -> np.ndarray:
    """Sparse reward: +1 at GOAL_STATE, 0 elsewhere."""
    r = np.zeros(N)
    r[GOAL_STATE] = 1.0
    return r


# ── Dynamic Programming ────────────────────────────────────────────────────────
def policy_eval(
    pi: np.ndarray, r: np.ndarray, gamma: float,
    tol: float = 1e-6, max_iter: int = 5000,
) -> np.ndarray:
    """Iterative policy evaluation — V^π via Bellman backups."""
    P = induced_chain(pi, alpha=1.0, eps=0.0)
    V = np.zeros(N)
    for _ in range(max_iter):
        V_new = r + gamma * (P @ V)
        if np.max(np.abs(V_new - V)) < tol:
            return V_new
        V = V_new
    return V


def greedy_policy(V: np.ndarray, r: np.ndarray, gamma: float) -> np.ndarray:
    """Greedy policy: π(s) = argmax_a [r(s) + γ Σ_{s'} P(s'|s,a) V(s')]."""
    P_sa = _kernel()                                       # (N, N, A_DIM)
    Q    = r[:, None] + gamma * np.einsum("sna,n->sa", P_sa, V)
    pi   = np.zeros((N, A_DIM))
    pi[np.arange(N), np.argmax(Q, axis=1)] = 1.0
    for s in WALL_STATES:
        pi[s] = 1.0 / A_DIM
    return pi


def policy_iteration(
    r: np.ndarray, gamma: float,
    tol: float = 1e-6, max_iter: int = 50,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Policy Iteration. Returns [(π_k, V^{π_k})] for each step."""
    pi      = np.ones((N, A_DIM)) / A_DIM
    history: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(max_iter):
        V      = policy_eval(pi, r, gamma, tol)
        history.append((pi.copy(), V.copy()))
        pi_new = greedy_policy(V, r, gamma)
        if np.allclose(pi_new, pi):
            break
        pi = pi_new
    return history


def value_iteration(
    r: np.ndarray, gamma: float,
    tol: float = 1e-6, max_iter: int = 2000,
) -> list[np.ndarray]:
    """Value Iteration. Returns [V_0, V_1, ..., V_*]."""
    P_sa    = _kernel()
    V       = np.zeros(N)
    history = [V.copy()]
    for _ in range(max_iter):
        Q     = r[:, None] + gamma * np.einsum("sna,n->sa", P_sa, V)
        V_new = Q.max(axis=1)
        history.append(V_new.copy())
        if np.max(np.abs(V_new - V)) < tol:
            break
        V = V_new
    return history


# ── Visualisation (DP) ────────────────────────────────────────────────────────
def show_value(
    V: np.ndarray, pi: np.ndarray | None = None, title: str = "",
) -> go.Figure:
    """Value heatmap + goal marker + optional policy arrows."""
    traces = [
        go.Heatmap(z=V.reshape(GRID, GRID), colorscale=_VALUE_CS,
                   showscale=True, hoverinfo="skip"),
        go.Heatmap(z=_wall_z(), colorscale=_WALL_CS,
                   showscale=False, zmin=0, zmax=1, hoverinfo="skip"),
        go.Scatter(x=[GOAL_CELL[1]], y=[GOAL_CELL[0]], mode="markers",
                   marker=dict(symbol="star", size=16, color=_WARM),
                   showlegend=False, hoverinfo="skip"),
    ]
    if pi is not None:
        P_pi = induced_chain(pi, alpha=1.0, eps=0.0)
        traces += _quiver(*_mean_arrows(P_pi))
    fig = go.Figure(data=traces)
    fig.update_layout(**_grid_layout(title))
    return fig


def vi_pi_convergence(
    vi_hist: list[np.ndarray],
    pi_hist: list[tuple[np.ndarray, np.ndarray]],
    V_star: np.ndarray,
) -> go.Figure:
    """‖V_k − V*‖∞ convergence curves for VI and PI."""
    vi_err = [float(np.max(np.abs(V - V_star))) for V in vi_hist]
    pi_err = [float(np.max(np.abs(V - V_star))) for _, V in pi_hist]
    fig = go.Figure([
        go.Scatter(y=vi_err, mode="lines", name="Value Iteration",
                   line=dict(color=_ACCENT, width=2)),
        go.Scatter(y=pi_err, mode="lines+markers", name="Policy Iteration",
                   line=dict(color=_WARM, width=2), marker=dict(size=7)),
    ])
    fig.update_layout(
        paper_bgcolor=_DARK, plot_bgcolor="#0d1220", font_color=_FG,
        title=dict(text="Convergence  ‖V<sub>k</sub> − V*‖<sub>∞</sub>",
                   font_color=_FG),
        xaxis=dict(title="iteration k", color=_MUTED, gridcolor="#1e2a3a"),
        yaxis=dict(title="error (log scale)", type="log",
                   color=_MUTED, gridcolor="#1e2a3a"),
        legend=dict(bgcolor="#0d1220", bordercolor="#1e2a3a"),
        width=700, height=380, margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ── Model-Free helpers ─────────────────────────────────────────────────────────
_NEXT_SA: np.ndarray | None = None
_FREE_ARR: np.ndarray | None = None


def _ns_matrix() -> np.ndarray:
    """Deterministic next-state table — shape (N, A_DIM), dtype int."""
    global _NEXT_SA
    if _NEXT_SA is None:
        _NEXT_SA = np.argmax(_kernel(), axis=1)  # argmax over s' axis
    return _NEXT_SA


def _free_arr() -> np.ndarray:
    """States that are neither walls nor the goal."""
    global _FREE_ARR
    if _FREE_ARR is None:
        _FREE_ARR = np.array([s for s in range(N)
                               if s not in WALL_STATES and s != GOAL_STATE])
    return _FREE_ARR


def _td0_run(
    pi: np.ndarray, r: np.ndarray, gamma: float,
    n_episodes: int, alpha: float, max_steps: int,
    V_ref: np.ndarray | None,
    checkpoints: set[int],
    seed: int,
) -> tuple[list[tuple[int, np.ndarray]], np.ndarray | None]:
    """Internal TD(0) runner — continuing task, reward r[s] at current state."""
    rng   = np.random.default_rng(seed)
    NS    = _ns_matrix()
    free  = _free_arr()
    V     = np.zeros(N)
    snaps: list[tuple[int, np.ndarray]] = []
    errs  = np.empty(n_episodes) if V_ref is not None else None
    for ep in range(1, n_episodes + 1):
        s = int(rng.choice(free))
        for _ in range(max_steps):
            a  = int(rng.choice(A_DIM, p=pi[s]))
            s2 = NS[s, a]
            V[s] += alpha * (r[s] + gamma * V[s2] - V[s])
            s = s2
        if ep in checkpoints:
            snaps.append((ep, V.copy()))
        if errs is not None:
            errs[ep - 1] = float(np.max(np.abs(V - V_ref)))
    return snaps, errs


# ── Model-Free — Policy Evaluation ────────────────────────────────────────────
def td_eval(
    pi: np.ndarray, r: np.ndarray, gamma: float,
    n_episodes: int = 3000, alpha: float = 0.05, max_steps: int = 500,
    checkpoints: tuple[int, ...] = (100, 500, 1500, 3000),
    seed: int = 42,
) -> list[tuple[int, np.ndarray]]:
    """TD(0) policy evaluation. Returns [(episode, V_snapshot)] at checkpoints."""
    snaps, _ = _td0_run(pi, r, gamma, n_episodes, alpha, max_steps,
                         None, set(checkpoints), seed)
    return snaps


def td_convergence(
    pi: np.ndarray, r: np.ndarray, gamma: float, V_ref: np.ndarray,
    alphas: tuple[float, ...] = (0.01, 0.05, 0.2),
    n_episodes: int = 3000, seed: int = 42,
) -> dict[str, np.ndarray]:
    """TD(0) ‖V − V_ref‖∞ error curve for each α. Returns {label: error_array}."""
    return {
        f"α = {a}": _td0_run(pi, r, gamma, n_episodes, a, 500,
                               V_ref, set(), seed)[1]
        for a in alphas
    }


def td_lambda_convergence(
    pi: np.ndarray, r: np.ndarray, gamma: float, V_ref: np.ndarray,
    lambdas: tuple[float, ...] = (0.0, 0.5, 0.9),
    n_episodes: int = 3000, alpha: float = 0.05,
    max_steps: int = 500, seed: int = 42,
) -> dict[str, np.ndarray]:
    """TD(λ) ‖V − V_ref‖∞ error curve for each λ. Returns {label: error_array}."""
    NS   = _ns_matrix()
    free = _free_arr()
    out: dict[str, np.ndarray] = {}
    for lam in lambdas:
        rng  = np.random.default_rng(seed)
        V    = np.zeros(N)
        errs = np.empty(n_episodes)
        for ep in range(n_episodes):
            s = int(rng.choice(free))
            e = np.zeros(N)
            for _ in range(max_steps):
                a     = int(rng.choice(A_DIM, p=pi[s]))
                s2    = NS[s, a]
                delta = r[s] + gamma * V[s2] - V[s]
                e[s] += 1.0
                V    += alpha * delta * e
                e    *= gamma * lam
                s     = s2
            errs[ep] = float(np.max(np.abs(V - V_ref)))
        out[f"λ = {lam}"] = errs
    return out


# ── Model-Free — Control ───────────────────────────────────────────────────────
def sarsa(
    r: np.ndarray, gamma: float,
    n_episodes: int = 5000, alpha: float = 0.1, eps: float = 0.1,
    max_steps: int = 1000, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """SARSA (on-policy TD control). Returns (Q, episode_returns)."""
    rng  = np.random.default_rng(seed)
    NS   = _ns_matrix()
    free = _free_arr()
    Q    = np.zeros((N, A_DIM))
    rets = np.zeros(n_episodes)
    for ep in range(n_episodes):
        s      = int(rng.choice(free))
        a      = int(np.argmax(Q[s]) if rng.random() > eps else rng.integers(A_DIM))
        ep_ret = 0.0
        disc   = 1.0
        for _ in range(max_steps):
            s2  = NS[s, a]
            rew = r[s]
            ep_ret += disc * rew
            disc   *= gamma
            if s2 == GOAL_STATE:
                Q[s, a] += alpha * (rew + gamma * r[GOAL_STATE] - Q[s, a])
                ep_ret  += disc * r[GOAL_STATE]
                break
            a2 = int(np.argmax(Q[s2]) if rng.random() > eps else rng.integers(A_DIM))
            Q[s, a] += alpha * (rew + gamma * Q[s2, a2] - Q[s, a])
            s, a = s2, a2
        rets[ep] = ep_ret
    return Q, rets


def q_learning(
    r: np.ndarray, gamma: float,
    n_episodes: int = 5000, alpha: float = 0.1, eps: float = 0.1,
    max_steps: int = 1000, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Q-Learning (off-policy TD control). Returns (Q, episode_returns)."""
    rng  = np.random.default_rng(seed)
    NS   = _ns_matrix()
    free = _free_arr()
    Q    = np.zeros((N, A_DIM))
    rets = np.zeros(n_episodes)
    for ep in range(n_episodes):
        s      = int(rng.choice(free))
        ep_ret = 0.0
        disc   = 1.0
        for _ in range(max_steps):
            a  = int(np.argmax(Q[s]) if rng.random() > eps else rng.integers(A_DIM))
            s2 = NS[s, a]
            rew = r[s]
            ep_ret += disc * rew
            disc   *= gamma
            if s2 == GOAL_STATE:
                Q[s, a] += alpha * (rew + gamma * r[GOAL_STATE] - Q[s, a])
                ep_ret  += disc * r[GOAL_STATE]
                break
            Q[s, a] += alpha * (rew + gamma * np.max(Q[s2]) - Q[s, a])
            s = s2
        rets[ep] = ep_ret
    return Q, rets


def policy_from_q(Q: np.ndarray) -> np.ndarray:
    """Greedy deterministic policy derived from Q-values."""
    pi = np.zeros((N, A_DIM))
    pi[np.arange(N), np.argmax(Q, axis=1)] = 1.0
    for s in WALL_STATES:
        pi[s] = 1.0 / A_DIM
    return pi


# ── Visualisation (Model-Free) ─────────────────────────────────────────────────
def show_td_snapshots(
    snapshots: list[tuple[int, np.ndarray]], V_true: np.ndarray,
) -> go.Figure:
    """Subplots: V̂ at each TD checkpoint + exact V^π (rightmost panel)."""
    from plotly.subplots import make_subplots
    panels = list(snapshots) + [(-1, V_true)]
    titles = [f"TD — ep {ep}" for ep, _ in snapshots] + ["V<sup>π</sup> exact"]
    n      = len(panels)
    fig    = make_subplots(rows=1, cols=n, subplot_titles=titles)
    wall   = _wall_z()
    vmax   = float(V_true.max())
    for col, (_, V) in enumerate(panels, 1):
        ax = "" if col == 1 else str(col)
        fig.add_trace(go.Heatmap(z=V.reshape(GRID, GRID), colorscale=_VALUE_CS,
                                  showscale=False, zmin=0, zmax=vmax,
                                  hoverinfo="skip"), row=1, col=col)
        fig.add_trace(go.Heatmap(z=wall, colorscale=_WALL_CS,
                                  showscale=False, zmin=0, zmax=1,
                                  hoverinfo="skip"), row=1, col=col)
        fig.add_trace(go.Scatter(x=[GOAL_CELL[1]], y=[GOAL_CELL[0]], mode="markers",
                                  marker=dict(symbol="star", size=12, color=_WARM),
                                  showlegend=False, hoverinfo="skip"),
                       row=1, col=col)
        fig.update_layout(**{
            f"xaxis{ax}": dict(showticklabels=False, showgrid=False, zeroline=False),
            f"yaxis{ax}": dict(showticklabels=False, showgrid=False, zeroline=False,
                               autorange="reversed"),
        })
    fig.update_layout(
        paper_bgcolor=_DARK, plot_bgcolor=_DARK, font_color=_FG,
        title=dict(text="TD(0) estimates vs exact V<sup>π</sup>", font_color=_FG),
        width=min(n * 260, 1200), height=320,
        margin=dict(l=5, r=5, t=60, b=5),
    )
    return fig


def show_mf_convergence(
    curves: dict[str, np.ndarray],
    ylabel: str = "",
    title: str = "",
    smooth: int = 1,
) -> go.Figure:
    """Line chart of named learning curves (error or episode return)."""
    colors = [_ACCENT, _WARM, _VIOLET, "#4ade80", "#f87171"]
    fig = go.Figure()
    for (name, y), col in zip(curves.items(), colors):
        y_plot = np.convolve(y, np.ones(smooth) / smooth, mode="valid") if smooth > 1 else y
        fig.add_trace(go.Scatter(
            y=y_plot, mode="lines", name=name,
            line=dict(color=col, width=2),
        ))
    fig.update_layout(
        paper_bgcolor=_DARK, plot_bgcolor="#0d1220", font_color=_FG,
        title=dict(text=title, font_color=_FG),
        xaxis=dict(title="episode", color=_MUTED, gridcolor="#1e2a3a"),
        yaxis=dict(title=ylabel, color=_MUTED, gridcolor="#1e2a3a"),
        legend=dict(bgcolor="#0d1220", bordercolor="#1e2a3a"),
        width=700, height=380, margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig
