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
| $\mu_k$ | expected reward of arm $k$ | Ch2 |
| $\hat{\mu}_k$ | empirical estimate of $\mu_k$ | Ch2 |
| $n_k$ | number of pulls of arm $k$ | Ch2 |
| $R$ | cumulative regret | Ch2 |
