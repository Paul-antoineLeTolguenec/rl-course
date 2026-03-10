# Course Conventions

## Slide Design

- **One idea per slide** — no bullet-point overload
- **Minimal text** — short statements, key equations, one figure max
- No decorative redundancy — every element must earn its place
- Use `<strong>` for key terms, display math for formulas

---

## Figures & SVG

### Principles
- All figures are **hand-coded SVGs** — no external diagram tools
- Every figure must respect the site **color palette** (see below)
- **Reuse before creating** — before proposing a new figure, check if an existing one can be adapted (same base structure, modified labels/highlights)
- The RL loop figure (agent ↔ environment) is the **canonical base figure**; MDP figure must derive from it

### Color palette (from `css/theme.css`)
| Role | Value |
|------|-------|
| Background | `#070b14` |
| Text | `#e2e8f0` |
| Muted / labels | `#7a8fa6` |
| Accent (cyan) | `#00d4ff` |
| Accent 2 (violet) | `#a78bfa` |
| Border | `rgba(0,212,255,0.12)` |
| Warm highlight | `#ffc850` |

### Figure consistency rules
- Node fills: `rgba(13,18,32,0.75)` with `1px` stroke in accent color
- Arrows: `#00d4ff` (primary flow) or `#a78bfa` (secondary)
- Labels: `#e2e8f0`, font `Inter`, size consistent across figures
- Rounded rects: `rx="8"`
- When a concept is introduced visually, its **shape and color are fixed** for the rest of the course

---

## Mathematical Notation

### Principles
- **Lightweight notation** — prefer short symbols, avoid subscript/superscript stacks
- **No re-introduction** — once a symbol is defined, it is never redefined with a different meaning
- **No redundant re-statement** — do not re-display a definition that has already appeared on a previous slide

### Mandatory check before introducing any new notation
Before writing any new symbol or equation, always:
1. Check the **notation table** below — if the concept is already covered, reuse the existing symbol
2. If a new symbol is needed, add it to the table **before** writing it into a slide or notebook
3. If unsure whether a new symbol is needed, default to **reusing** an existing one

### Persistent notation table
*(updated each time a new symbol is introduced)*

| Symbol | Meaning | Introduced |
|--------|---------|-----------|
| $t$ | time step | Ch1 |
| $r_t$ | reward at step $t$ | Ch1 |
| $G_t$ | cumulative return from step $t$ | Ch1 |
| $k$ | arm index (bandit) | Ch2 |
| $K$ | number of arms | Ch2 |
| $\mu_k$ | expected reward of arm $k$ — **scalar, bandit context only** | Ch2 |
| $\hat{\mu}_k$ | empirical estimate of $\mu_k$ | Ch2 |
| $n_k$ | number of pulls of arm $k$ | Ch2 |
| $R$ | cumulative regret | Ch2 |
| $\mathcal{S}$ | state space | Ch2 |
| $\mathcal{A}$ | action space | Ch2 |
| $S_t$ | state random variable at step $t$ | Ch2 |
| $A_t$ | action random variable at step $t$ | Ch2 |
| $\pi$ | policy $\mathcal{S} \to \Delta(\mathcal{A})$ | Ch2 |
| $P$ | controlled kernel $\mathcal{S} \times \mathcal{A} \to \Delta(\mathcal{S})$ | Ch2 |
| $P^\pi$ | induced kernel $P^\pi(s'\mid s) = \sum_a \pi(a\mid s) P(s'\mid s,a)$ | Ch2 |
| $\mu_t$ | instantaneous distribution $\mu_t \in \Delta(\mathcal{S})$, $[\mu_t]_s = \mathbb{P}(S_t=s)$ — **vector, MDP context** | Ch2 |
| $\mu_t^\pi$ | instantaneous distribution under policy $\pi$, $\mu_{t+1}^\pi = \mu_t^\pi P^\pi$ | Ch2 |
| $\mu_\infty$ | stationary distribution $\mu_\infty = \lim_{t\to\infty} \delta_{s_0} (P^\pi)^t$ | Ch2 |
| $\rho$ | (undiscounted) occupation measure $\rho = \lim_{T\to\infty} \frac{1}{T}\sum_{t=0}^{T-1} \mu_t$ | Ch2 |
| $d^\pi_\gamma$ | discounted occupation measure $(1-\gamma)\sum_{t=0}^\infty \gamma^t \mu_t^\pi$ | Ch2 |
| $r$ | reward function $\mathcal{S} \times \mathcal{A} \to \mathbb{R}$ | Ch2 |
| $\gamma$ | discount factor | Ch2 |
| $\mathcal{M}$ | MDP tuple $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$ | Ch2 |
