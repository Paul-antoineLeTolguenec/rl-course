// Game of Life — fixed grid, double-buffer Uint8Array, mouse perturbation, auto-respawn

const CELL         = 8;     // px per cell
const RADIUS       = 55;    // mouse influence radius (px)
const PERTURB_RATE = 0.12;  // max perturbation probability at center
const DENSITY_MIN  = 0.02;  // respawn threshold (2% alive)
const RESPAWN_EVERY = 4000; // ms between respawn checks
const STEP_EVERY   = 80;    // ms between GoL steps (~12 fps)

// Classic patterns used for respawn injection
const PATTERNS = [
  [[1,0],[2,1],[0,2],[1,2],[2,2]],          // glider
  [[1,0],[2,0],[0,1],[1,1],[1,2]],          // R-pentomino
  [[0,0],[1,0],[2,0],[3,0],[4,0]],          // blinker (5-cell)
  [[0,0],[1,0],[2,0],[0,1],[1,2]],          // boat
];

class GameOfLife {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx    = canvas.getContext('2d');
    this.mx     = -9999;
    this.my     = -9999;
    this.lastStep    = 0;
    this.lastRespawn = 0;

    this.resize();
    this.bindEvents();
    requestAnimationFrame(t => this.frame(t));
  }

  resize() {
    const { canvas } = this;
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
    this.W = Math.ceil(canvas.width  / CELL);
    this.H = Math.ceil(canvas.height / CELL);
    const n = this.W * this.H;
    this.cur = new Uint8Array(n);
    this.nxt = new Uint8Array(n);
    this.age = new Uint8Array(n);
    this.seed();
  }

  seed() {
    for (let i = 0; i < this.cur.length; i++) {
      this.cur[i] = Math.random() < 0.28 ? 1 : 0;
      this.age[i] = this.cur[i] ? (Math.random() * 30 | 0) : 0;
    }
  }

  // Locally perturb cells within mouse radius — only if at least one cell is alive in the radius
  perturb() {
    const { cur, W, H, mx, my } = this;
    const cr  = Math.ceil(RADIUS / CELL);
    const mcx = mx / CELL | 0;
    const mcy = my / CELL | 0;

    // Check if at least one alive cell exists in the radius
    let hasAlive = false;
    outer: for (let dy = -cr; dy <= cr; dy++) {
      const cy = mcy + dy;
      if (cy < 0 || cy >= H) continue;
      for (let dx = -cr; dx <= cr; dx++) {
        const cx = mcx + dx;
        if (cx < 0 || cx >= W) continue;
        const d = Math.sqrt(dx * dx + dy * dy) * CELL;
        if (d > RADIUS) continue;
        if (cur[cy * W + cx]) { hasAlive = true; break outer; }
      }
    }
    if (!hasAlive) return;

    for (let dy = -cr; dy <= cr; dy++) {
      const cy = mcy + dy;
      if (cy < 0 || cy >= H) continue;
      for (let dx = -cr; dx <= cr; dx++) {
        const cx = mcx + dx;
        if (cx < 0 || cx >= W) continue;
        const d = Math.sqrt(dx * dx + dy * dy) * CELL;
        if (d > RADIUS) continue;
        if (Math.random() < PERTURB_RATE * (1 - d / RADIUS))
          cur[cy * W + cx] = Math.random() < 0.55 ? 1 : 0;
      }
    }
  }

  // One Conway step with toroidal boundary
  step() {
    this.perturb();
    const { cur, nxt, age, W, H } = this;

    for (let y = 0; y < H; y++) {
      const ym = ((y - 1 + H) % H) * W;
      const yc = y * W;
      const yp = ((y + 1) % H) * W;

      for (let x = 0; x < W; x++) {
        const xm = (x - 1 + W) % W;
        const xp = (x + 1) % W;
        const n  =
          cur[ym + xm] + cur[ym + x] + cur[ym + xp] +
          cur[yc + xm]               + cur[yc + xp] +
          cur[yp + xm] + cur[yp + x] + cur[yp + xp];

        const alive = cur[yc + x];
        const next  = alive ? (n === 2 || n === 3 ? 1 : 0) : (n === 3 ? 1 : 0);
        nxt[yc + x] = next;
        age[yc + x] = next ? Math.min(255, age[yc + x] + 1) : 0;
      }
    }
    // Swap buffers
    const tmp = this.cur; this.cur = this.nxt; this.nxt = tmp;
  }

  // checkRespawn disabled — no auto-injection of patterns
  // checkRespawn(now) { ... }

  render() {
    const { ctx, canvas, cur, age, W, H, mx, my } = this;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const mcx = mx / CELL | 0;
    const mcy = my / CELL | 0;
    const mr  = RADIUS / CELL;

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = y * W + x;
        if (!cur[i]) continue;

        // t: 0 = young cell, 1 = old cell
        const t    = Math.min(1, age[i] / 45);
        const dist = Math.sqrt((x - mcx) ** 2 + (y - mcy) ** 2);
        const mf   = Math.max(0, 1 - dist / mr);

        // Base gradient: white-blue (young) → sky blue → deep teal (old)
        let r = Math.round(210 * (1 - t) +  8 * t);
        let g = Math.round(235 * (1 - t) + 145 * t);
        let b = Math.round(255 - t * 75);
        let a = 0.55 + 0.3 * (1 - t);

        // Mouse proximity: blend toward violet (#a78bfa)
        if (mf > 0) {
          r = Math.round(r + (167 - r) * mf * 0.88);
          g = Math.round(g + (139 - g) * mf * 0.88);
          b = Math.round(b + (250 - b) * mf * 0.88);
          a = Math.min(1, a + mf * 0.4);
        }

        ctx.fillStyle = `rgba(${r},${g},${b},${a})`;
        ctx.fillRect(x * CELL + 1, y * CELL + 1, CELL - 2, CELL - 2);
      }
    }
  }

  frame(t) {
    if (t - this.lastStep >= STEP_EVERY) {
      this.step();
      this.lastStep = t;
    }
    this.render();
    requestAnimationFrame(ts => this.frame(ts));
  }

  bindEvents() {
    window.addEventListener('mousemove',  e => { this.mx = e.clientX; this.my = e.clientY; });
    window.addEventListener('mouseleave', () => { this.mx = -9999;    this.my = -9999; });
    let t;
    window.addEventListener('resize', () => {
      clearTimeout(t);
      t = setTimeout(() => this.resize(), 150);
    });
  }
}

window.addEventListener('DOMContentLoaded', () => {
  new GameOfLife(document.getElementById('bg'));
});
