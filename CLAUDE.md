# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Static site for an interactive Reinforcement Learning course, deployed to GitHub Pages via GitHub Actions. No build step — pure HTML/CSS/JS served as-is.

**Language rule:** all site content (slides, labels, UI text) must be written in **English**.

## Code Standards

- Minimal, non-redundant, production-quality code
- No helper wrappers around one-liners, no speculative abstractions
- CSS: use existing variables (`--bg`, `--accent`, etc.) — never hardcode color values

## Deployment

Push to `main` triggers automatic deployment via `.github/workflows/static.yml`. The entire repo root is the site root.

## Architecture

Three-section single-page app built on **Reveal.js 5.1** (CDN):

- **Landing** (`section.landing-section`) — title + author, Game of Life canvas background
- **Slides** (`section > section`) — nested vertical slides
- **Notebooks** (`section.notebooks-section`) — grid of notebook/resource cards

Navigation is remapped in `index.html`:
- `↓` / wheel-down → next major section (`Reveal.right()`)
- `↑` / wheel-up → previous major section (`Reveal.left()`)
- `←` / `→` → navigate slides within the slides section (`Reveal.up()` / `Reveal.down()`)

**Background canvas** (`#bg`, `js/automata.js`) — Conway's Game of Life at ~12fps, double-buffer `Uint8Array`, toroidal boundary, mouse perturbation, auto-respawn below 2% density.

**CSS** (`css/theme.css`) — CSS variables on `:root` for the dark theme. Canvas at `z-index: 0`, Reveal at `z-index: 1`.

## Math / LaTeX

Use **KaTeX** via the `RevealMath.KaTeX` plugin (CDN). Syntax: `$...$` inline, `$$...$$` display block. Plugin config goes in the `Reveal.initialize()` call:

```js
plugins: [ RevealMath.KaTeX ],
math: { katex: { /* options */ } }
```

Add the plugin script tag before `Reveal.initialize()`:
```html
<script src="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/plugin/math/math.js"></script>
```

## Slide Design Rules

- Each slide = one focused idea; avoid bullet-point overload
- Prefer short statements and key equations over long prose
- Use `<strong>` for key terms, display math for formulas
- `.slide-nav` hint stays at the bottom of every slide

## Adding Content

**New slide:** add a `<section>` inside the nested slides `<section>` block (between `SLIDES` comments).

**New notebook card:** add `<a class="notebook-card">` inside `.notebook-grid` with `.nb-num`, `h3`, and `p` children.

## Local Preview

```sh
python3 -m http.server
# or
npx serve .
```
