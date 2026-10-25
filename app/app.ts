/**
 * hopf-hough interactive app: real-time exact retarded Liénard–Wiechert fields with
 * live comparison against an independent double-precision reference.
 */
import {
  DEFAULT_CONFIG, Simulation, exactField, legacyPreset, pulseCrossingSweep, runCase,
  circularOrbit, harmonicOscillator, linearMotion, hyperbolicMotion, precessingEllipse,
  expressionTrajectory, csvTrajectory, assertSubluminal, shellRadius,
  type Kinematics, type SolverConfig, type Trajectory, type CaseResult,
} from '../src/core/index.js';
import { GpuSolver, kernelShaders, gpuBackend, type KernelName } from '../src/gpu/index.js';

// ---------------------------------------------------------------- DOM helpers
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const fmt = (v: number, d = 3) => (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(d - 1) : v.toFixed(d));
const sci = (v: number) => (v === 0 ? '0' : v.toExponential(2));

// ---------------------------------------------------------------- state
type Scenario = 'orbit' | 'ensemble' | 'oscillator' | 'linear' | 'hyperbolic' | 'custom' | 'csv';

const C = DEFAULT_CONFIG.c;
const cfg: SolverConfig = { ...DEFAULT_CONFIG, mT: 128, kBin: 9, softEps: 0.02, prefilter: false };
const ui = {
  scenario: 'ensemble' as Scenario,
  beta: 0.8,
  n: 16,
  omega: 1.6,
  channel: 0,
  wavefronts: true,
  mask: true,
  impostors: false,
  panMode: false,
  paused: false,
  view: { cx: 0, cy: 0, halfH: 1.45 },
  probe: { x: 0.95, y: 0.4 },
  customX: '0.8*cos(2.2*t)',
  customY: '0.45*sin(4.4*t)',
  csvText: '' as string,
  csvRange: null as null | [number, number],
};

let sim: Simulation;
let solver: GpuSolver | null = null;
/** Production kernel: v6f (default) or v6 (?kernel=v6). */
let kernel: KernelName = 'v6f';
let benchV6: GpuSolver | null = null;
let trajectories: Trajectory[] = [];
let reseedPeriod = 0;           // for linear/hyperbolic: reseed so charges re-enter the view
let reseedClock = 0;
let lastOverrideTime = -Infinity;
const drag = { active: false, returning: false, target: { x: 0, y: 0 }, what: '' as '' | 'source' | 'probe' | 'pan', start: { x: 0, y: 0, cx: 0, cy: 0 } };

// ---------------------------------------------------------------- trajectories
function sliderToBeta(v: number) { const u = Math.max(0, Math.min(1, v / 1000)); return 1 - 0.95 * Math.pow(1e-4 / 0.95, u); }
function betaToSlider(b: number) { return Math.round((Math.log((1 - b) / 0.95) / Math.log(1e-4 / 0.95)) * 1000); }

function orbitRadius(beta: number, omega: number) { return Math.min(0.85, (beta * C) / Math.max(0.5, omega)); }

function buildTrajectories(): Trajectory[] {
  const { scenario, beta, omega } = ui;
  const n = scenario === 'custom' || scenario === 'csv' ? 1 : ui.n;
  const out: Trajectory[] = [];
  reseedPeriod = 0;
  const yOff = (i: number, span: number) => (n > 1 ? (i / (n - 1) - 0.5) * span : 0);
  switch (scenario) {
    case 'orbit': {
      const r = orbitRadius(beta, omega);
      for (let i = 0; i < n; i++) out.push(circularOrbit({ radius: r, beta, c: C, phase: (2 * Math.PI * i) / n }));
      break;
    }
    case 'ensemble':
      for (let i = 0; i < n; i++) out.push(precessingEllipse({ index: i, beta, omega, c: C }));
      break;
    case 'oscillator': {
      const A = 0.6, w = (beta * C) / A;
      for (let i = 0; i < n; i++) out.push(harmonicOscillator({ amplitude: A, omega: w, c: C, phase: (2 * Math.PI * i) / n, y0: yOff(i, 1.1) }));
      break;
    }
    case 'linear': {
      const span = 3.4, v = beta * C;
      reseedPeriod = span / v;
      for (let i = 0; i < n; i++) out.push(linearMotion({ beta, c: C, x0: -span / 2, y0: yOff(i, 1.2), t0: sim ? sim.time : 2.5 }));
      break;
    }
    case 'hyperbolic': {
      const al = 2.5, gb = beta / Math.sqrt(1 - beta * beta);
      reseedPeriod = (C / al) * gb;
      for (let i = 0; i < n; i++) out.push(hyperbolicMotion({ properAccel: al, c: C, x0: -1.6, y0: yOff(i, 1.2), t0: sim ? sim.time : 2.5 }));
      break;
    }
    case 'custom':
      out.push(expressionTrajectory({ x: ui.customX, y: ui.customY, c: C }));
      break;
    case 'csv':
      out.push(csvTrajectory(ui.csvText, C));
      break;
  }
  return out;
}

function lookbackSpan() { return (cfg.ringSize * cfg.dt); }

function applyScenario(): boolean {
  const err = $('scenario-error');
  try {
    const trs = buildTrajectories();
    const t0 = ui.scenario === 'csv' && ui.csvRange ? ui.csvRange[0] + 0.25 * (ui.csvRange[1] - ui.csvRange[0]) : (sim ? sim.time : 2.5);
    // Refuse superluminal motion over the window the history will hold.
    for (const tr of trs) assertSubluminal(tr, t0 - Math.min(lookbackSpan(), 40), t0 + 10, 4096);
    trajectories = trs;
    if (!sim) sim = new Simulation(cfg, trs, t0);
    else { sim.config = { ...cfg }; sim.time = t0; sim.setTrajectories(trs); }
    sim.override = dragOverride;
    reseedClock = 0;
    lastOverrideTime = -Infinity;
    err.classList.add('hidden');
    updateNotes();
    return true;
  } catch (e) {
    err.textContent = (e as Error).message;
    err.classList.remove('hidden');
    return false;
  }
}

// ---------------------------------------------------------------- dragging (consistent kinematics)
function dragOverride(i: number, t: number, dt: number, prev: Kinematics, prescribed: Kinematics): Kinematics | null {
  if (i !== 0 || !(drag.active && drag.what === 'source') && !drag.returning) return null;
  const goal = drag.active ? drag.target : { x: prescribed.x, y: prescribed.y };
  const goalB = drag.active ? { x: 0, y: 0 } : { x: prescribed.bx, y: prescribed.by };
  const dx = goal.x - prev.x, dy = goal.y - prev.y;
  if (!drag.active && Math.hypot(dx, dy) < 0.006 && Math.hypot(prev.bx - goalB.x, prev.by - goalB.y) < 0.02) {
    drag.returning = false;
    return null;
  }
  lastOverrideTime = t;
  const k = 8.5;
  let tbx = goalB.x + (dx * k) / C, tby = goalB.y + (dy * k) / C;
  const tb = Math.hypot(tbx, tby), bmax = Math.min(0.9, Math.max(ui.beta, 0.5));
  if (tb > bmax) { tbx *= bmax / tb; tby *= bmax / tb; }
  let nbx = prev.bx + 0.085 * (tbx - prev.bx), nby = prev.by + 0.085 * (tby - prev.by);
  // Limit acceleration, then derive the stored acceleration from the velocity change actually made.
  const amax = 1.15 * C * dt;
  const dbx = nbx - prev.bx, dby = nby - prev.by, db = Math.hypot(dbx, dby);
  if (db > amax) { nbx = prev.bx + (dbx * amax) / db; nby = prev.by + (dby * amax) / db; }
  const bdx = (nbx - prev.bx) / (C * dt), bdy = (nby - prev.by) / (C * dt);   // a/c² = (Δβ/Δt)/c
  return { x: prev.x + 0.5 * (prev.bx + nbx) * C * dt, y: prev.y + 0.5 * (prev.by + nby) * C * dt, bx: nbx, by: nby, bdx, bdy, charge: prescribed.charge };
}

/** Trajectory that reads source 0's recorded history (used as the reference after dragging). */
function ringTrajectory(idx: number): Trajectory {
  const s = sim.sources[idx];
  return {
    kinematics(t) {
      const back = (sim.time - t) / cfg.dt;
      const h = s.ring.sample(back, C * cfg.dt);
      return { ...h, charge: s.current.charge };
    },
  };
}

function referenceTrajectories(probeDist: number) {
  const analytic = sim.time - lastOverrideTime > (probeDist + 2) / C;
  const trs = sim.trajectories.slice();
  if (!analytic) trs[0] = ringTrajectory(0);
  return { trs, analytic };
}

// ---------------------------------------------------------------- views / coordinates
const glCanvas = $<HTMLCanvasElement>('glCanvas');
const overlay = $<HTMLCanvasElement>('overlay');
const octx = overlay.getContext('2d')!;
let W = 1, H = 1, dpr = 1;

function resize() {
  const r = overlay.getBoundingClientRect();
  dpr = 1; // CSS-pixel resolution, as in the original app (rendering at device resolution costs 2–4× per frame)
  W = Math.max(1, Math.round(r.width * dpr));
  H = Math.max(1, Math.round(r.height * dpr));
  if (glCanvas.width !== W || glCanvas.height !== H) { glCanvas.width = W; glCanvas.height = H; }
  if (overlay.width !== W || overlay.height !== H) { overlay.width = W; overlay.height = H; }
}
const aspect = () => W / H;
function worldToScreen(x: number, y: number) {
  const v = ui.view;
  return { x: ((x - v.cx) / (v.halfH * aspect()) * 0.5 + 0.5) * W, y: (0.5 - (y - v.cy) / v.halfH * 0.5) * H };
}
function screenToWorld(px: number, py: number) {
  const v = ui.view;
  return { x: v.cx + (px / W - 0.5) * 2 * v.halfH * aspect(), y: v.cy + (0.5 - py / H) * 2 * v.halfH };
}
const pxPerUnit = () => H / (2 * ui.view.halfH);

// ---------------------------------------------------------------- GPU
function initGpu() {
  try {
    solver = new GpuSolver(glCanvas, kernelShaders(kernel));
  } catch (e) {
    const el = $('gl-error');
    el.textContent = 'This tool needs WebGL2 with float render targets (EXT_color_buffer_float): ' + (e as Error).message;
    el.classList.remove('hidden');
    solver = null;
  }
}

function gainForDisplay() { return 1.0; }

function renderField() {
  if (!solver) return;
  solver.upload(sim.shellHistories, cfg.mT);
  solver.render(W, H, cfg, {
    u_channel: ui.channel,
    u_showWavePool: true,
    u_showImpostors: ui.impostors,
    u_showMask: ui.mask,
    u_viewCenter: [ui.view.cx, ui.view.cy],
  }, {
    u_aspect: aspect(),
    u_viewHalfHeight: ui.view.halfH,
    u_time: sim.time,
    u_gain: gainForDisplay(),
    u_maskThreshold: 1.5,
  });
}

// ---------------------------------------------------------------- overlay
function niceStep(span: number) {
  const raw = span / 6, p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p;
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
}

function drawOverlay() {
  octx.clearRect(0, 0, W, H);
  const s0 = sim.sources[0];
  // Exact wavefronts of source 0: circles of radius R centred on the emission point at t − R/c.
  if (ui.wavefronts) {
    const spacing = 0.55, phase = ((sim.time * C) / spacing) % 1;
    octx.lineWidth = 1.2 * dpr;
    const maxR = ui.view.halfH * aspect() * 2.2;
    for (let k = 0; k < 40; k++) {
      const R = (k + phase) * spacing;
      if (R < 0.04) continue;
      if (R > maxR) break;
      const h = s0.ring.sample(R / (C * cfg.dt), C * cfg.dt);
      const c = worldToScreen(h.x, h.y);
      octx.strokeStyle = `rgba(0,0,0,${Math.max(0.12, 0.6 - R / maxR * 0.5).toFixed(2)})`;
      octx.beginPath();
      octx.arc(c.x, c.y, R * pxPerUnit(), 0, Math.PI * 2);
      octx.stroke();
    }
  }
  // Domain edge (outermost shell) and Nyquist radius around source 0.
  const rMax = shellRadius(cfg.mT - 1, cfg).r64;
  const c0 = worldToScreen(s0.current.x, s0.current.y);
  octx.setLineDash([6 * dpr, 5 * dpr]);
  octx.strokeStyle = 'rgba(251,113,133,0.7)';
  octx.lineWidth = 1.2 * dpr;
  octx.beginPath(); octx.arc(c0.x, c0.y, rMax * pxPerUnit(), 0, Math.PI * 2); octx.stroke();
  octx.setLineDash([]);

  // Sources.
  const rad = (sim.sources.length > 32 ? 3.5 : 5.5) * dpr;
  for (let i = sim.sources.length - 1; i >= 0; i--) {
    const s = sim.sources[i], p = worldToScreen(s.current.x, s.current.y);
    octx.fillStyle = i === 0 ? (drag.active && drag.what === 'source' ? '#ffffff' : '#38bdf8') : s.current.charge >= 0 ? '#38bdf8' : '#fb7185';
    octx.strokeStyle = '#0f172a';
    octx.lineWidth = 1.6 * dpr;
    octx.beginPath(); octx.arc(p.x, p.y, i === 0 ? rad + 1.2 * dpr : rad, 0, Math.PI * 2); octx.fill(); octx.stroke();
  }
  // Probe.
  const pp = worldToScreen(ui.probe.x, ui.probe.y);
  octx.strokeStyle = '#34d399'; octx.lineWidth = 2 * dpr;
  octx.beginPath(); octx.arc(pp.x, pp.y, 7.5 * dpr, 0, Math.PI * 2); octx.stroke();

  // Axes with world-coordinate ticks.
  const v = ui.view, xSpan = 2 * v.halfH * aspect(), step = niceStep(xSpan);
  octx.fillStyle = 'rgba(203,213,225,0.75)'; octx.strokeStyle = 'rgba(203,213,225,0.5)';
  octx.font = `${10 * dpr}px ui-monospace, monospace`;
  octx.lineWidth = 1 * dpr;
  const x0 = Math.ceil((v.cx - xSpan / 2) / step) * step;
  for (let x = x0; x <= v.cx + xSpan / 2; x += step) {
    const p = worldToScreen(x, 0);
    octx.beginPath(); octx.moveTo(p.x, H); octx.lineTo(p.x, H - 6 * dpr); octx.stroke();
    octx.fillText(+x.toFixed(6) + '', p.x + 2 * dpr, H - 8 * dpr);
  }
  const y0 = Math.ceil((v.cy - v.halfH) / step) * step;
  for (let y = y0; y <= v.cy + v.halfH; y += step) {
    const p = worldToScreen(0, y);
    octx.beginPath(); octx.moveTo(0, p.y); octx.lineTo(6 * dpr, p.y); octx.stroke();
    octx.fillText(+y.toFixed(6) + '', 8 * dpr, p.y - 2 * dpr);
  }
}

// ---------------------------------------------------------------- colorbar
function drawColorbar() {
  const cb = $<HTMLCanvasElement>('colorbar');
  const r = cb.getBoundingClientRect();
  cb.width = Math.max(1, Math.round(r.width)); cb.height = 12;
  const ctx = cb.getContext('2d')!;
  const comp = (m: number) => { const lm = Math.log(1 + 1.85 * m); return lm / (1.35 + lm); };
  for (let i = 0; i < cb.width; i++) {
    const c = i / (cb.width - 1);
    let col: string;
    if (ui.channel === 3) {
      const v = c * 2 - 1, a = Math.abs(v);
      const rgb = v < 0 ? [0.04 + (0.23 - 0.04) * a, 0.06 + (0.55 - 0.06) * a, 0.10 + (0.98 - 0.10) * a] : [0.04 + (0.98 - 0.04) * a, 0.06 + (0.42 - 0.06) * a, 0.10 + (0.30 - 0.10) * a];
      col = `rgb(${rgb.map((x) => Math.round(x * 255)).join(',')})`;
    } else {
      const val = Math.min(0.96, 0.14 + 0.82 * Math.pow(c, 0.75));
      const g = Math.round(val * 255);
      col = `rgb(${g},${g},${g})`;
    }
    ctx.fillStyle = col; ctx.fillRect(i, 0, 1, 12);
  }
  // Value at the right end (compressed = 0.95), per-source display gain included.
  const gEff = gainForDisplay() / Math.pow(Math.max(1, sim.sources.length), 0.42);
  const cMax = 0.95, lm = (1.35 * cMax) / (1 - cMax), mMax = (Math.exp(lm) - 1) / 1.85 / gEff;
  const labels = ['|E| brightness · hue = direction', 'E_rad brightness · hue = direction', 'E_vel brightness · hue = direction', 'B_z: blue < 0 < red', '|S| = |E||B_z|'];
  $('colorbar-label').textContent = labels[ui.channel];
  $('colorbar-max').textContent = (ui.channel === 3 ? '±' : '') + fmt(ui.channel === 4 ? mMax * mMax : mMax, 3) + ' (log scale)';
  void comp;
}

// ---------------------------------------------------------------- probe + accuracy
function probeReadout() {
  if (!solver) return;
  const g = solver.readFull([{ x: ui.probe.x, y: ui.probe.y }], cfg)[0];
  const dist = Math.hypot(ui.probe.x - sim.sources[0].current.x, ui.probe.y - sim.sources[0].current.y);
  const { trs, analytic } = referenceTrajectories(dist + 3);
  const ref = exactField(trs, Math.fround(ui.probe.x), Math.fround(ui.probe.y), sim.time, { c: C, softEps: cfg.softEps, maxLookback: lookbackSpan() * C * 0.98 });
  let bz = 0;
  // Reference B_z: sum over sources of (n × E)_z, recomputed per source.
  for (const tr of trs) {
    const one = exactField([tr], Math.fround(ui.probe.x), Math.fround(ui.probe.y), sim.time, { c: C, softEps: cfg.softEps, maxLookback: lookbackSpan() * C * 0.98 });
    const dx = ui.probe.x - one.emitX, dy = ui.probe.y - one.emitY, r = Math.hypot(dx, dy) || 1;
    bz += (dx / r) * one.ey - (dy / r) * one.ex;
  }
  const dE = Math.hypot(g.ex - ref.ex, g.ey - ref.ey);
  const rel = dE / Math.max(1e-30, ref.mag);
  $('probe-pos').textContent = `(${fmt(ui.probe.x, 3)}, ${fmt(ui.probe.y, 3)})`;
  $('probe-gpu').innerHTML = `E = (${sci(g.ex)}, ${sci(g.ey)})<br>B_z = ${sci(g.bz)}`;
  $('probe-ref-title').textContent = analytic ? 'Reference (float64, analytic trajectory)' : 'Reference (float64, recorded history)';
  $('probe-ref').innerHTML = `E = (${sci(ref.ex)}, ${sci(ref.ey)})<br>B_z = ${sci(bz)}`;
  const q = g.quality;
  const qText = q < 0 ? '<span class="text-rose-300">outside stored history</span>' : q > 1.5 ? `<span class="text-amber-300">under-resolved (ΔRγ/ρ = ${q.toFixed(2)})</span>` : `resolved (ΔRγ/ρ = ${q.toFixed(3)})`;
  $('probe-err').innerHTML = `|ΔE|/|E| = ${(rel * 100).toPrecision(3)}%<br>${qText}`;
}

function checkAccuracy() {
  if (!solver) return;
  const pts: { x: number; y: number }[] = [];
  for (let k = 0; k < 64; k++) {
    const a = (k / 64) * 2 * Math.PI, r = 0.05 + 0.1 * ((k * 7) % 8) / 8;
    pts.push({ x: Math.fround(ui.probe.x + r * Math.cos(a)), y: Math.fround(ui.probe.y + r * Math.sin(a)) });
  }
  const g = solver.readFull(pts, cfg);
  const { trs, analytic } = referenceTrajectories(Math.hypot(ui.probe.x, ui.probe.y) + 3);
  let num = 0, den = 0, bad = 0;
  pts.forEach((p, i) => {
    const ref = exactField(trs, p.x, p.y, sim.time, { c: C, softEps: cfg.softEps, maxLookback: lookbackSpan() * C * 0.98 });
    num += (g[i].ex - ref.ex) ** 2 + (g[i].ey - ref.ey) ** 2;
    den += ref.mag ** 2;
    if (g[i].quality < 0 || g[i].quality > 1.5) bad++;
  });
  const l2 = Math.sqrt(num / Math.max(1e-300, den)) * 100;
  const el = $('stat-check');
  el.textContent = `${l2.toPrecision(3)}%${analytic ? '' : ' (vs history)'}${bad ? ` · ${bad}/64 flagged` : ''}`;
  el.className = (l2 < 1 ? 'text-emerald-400' : l2 < 5 ? 'text-amber-300' : 'text-rose-300') + ' font-semibold';
}

// ---------------------------------------------------------------- notes / labels
function updateNotes() {
  const g = 1 / Math.sqrt((1 - ui.beta) * (1 + ui.beta));
  $('stat-beta').textContent = ui.beta.toFixed(ui.beta > 0.999 ? 5 : 4);
  $('stat-gamma').textContent = g.toFixed(2);
  $('val-beta').textContent = `${ui.beta.toFixed(ui.beta > 0.999 ? 5 : 4)} (γ = ${g.toFixed(2)})`;
  let note = '';
  if (ui.scenario === 'orbit') {
    const r = orbitRadius(ui.beta, ui.omega), w = (ui.beta * C) / r;
    note = `orbit radius ${r.toFixed(3)}, angular frequency ${w.toFixed(3)}${r === 0.85 && w.toFixed(2) !== ui.omega.toFixed(2) ? ' (radius capped at 0.85 to stay on screen)' : ''}. `;
    const width = r / (g * g * g);
    if (width * pxPerUnit() < 1) note += `The pulse is ${width.toExponential(1)} wide, narrower than a pixel: the image samples it sparsely; use the probe or export for values.`;
  }
  $('beta-note').textContent = note;
  const rMax = shellRadius(cfg.mT - 1, cfg).r64;
  $('domain-note').textContent = `history reach R ≤ ${rMax.toFixed(1)} (dashed red circle) · ring buffer ${lookbackSpan().toFixed(0)} time units`;
  const nIn = $<HTMLInputElement>('slider-n');
  nIn.disabled = ui.scenario === 'custom' || ui.scenario === 'csv';
}

// ---------------------------------------------------------------- verification suite
let lastSuite: unknown = null;
async function runVerification() {
  if (!solver) return;
  const report = $('report');
  const btn = $<HTMLButtonElement>('btn-suite');
  btn.disabled = true;
  ui.paused = true;
  const backend = gpuBackend(solver);
  const betas = [0.8, 0.95, 0.99, 0.999, 0.9999, 0.99999];
  const Ms: [number, number][] = [[128, 9], [512, 11], [2048, 13]];
  const prod = { ...DEFAULT_CONFIG, softEps: 0, prefilter: false };
  const legacy = { ...DEFAULT_CONFIG, newton: 2, certMode: 0 as const, rootMode: 0 as const, precision: 0 as const, betaMax: 0.99995, softEps: 0, prefilter: false };
  const rows: { beta: number; gamma: number; M: number; prod: CaseResult; prodDf: CaseResult; legacy: CaseResult | null }[] = [];
  const tick = () => new Promise((r) => setTimeout(r, 0));
  for (const b of betas) {
    for (const [M, kb] of Ms) {
      report.textContent = `Running β = ${b}, M_T = ${M} …`;
      await tick();
      const spec = { beta: b, motion: 'synchrotron' as const, mT: M, kBin: kb, velMode: 2 as const, nShock: 256 };
      const p = runCase(spec, backend, { base: prod });
      const pd = runCase(spec, backend, { base: prod, queryMode: 'df64' });
      const l = b <= 0.9999 ? runCase(spec, backend, { base: legacy }) : null;
      rows.push({ beta: b, gamma: 1 / Math.sqrt(1 - b * b), M, prod: p, prodDf: pd, legacy: l });
    }
  }
  lastSuite = { generated: new Date().toISOString(), renderer: rendererName(), kernel, settings: { production: prod, legacy }, rows };
  const pct = (v: number) => (v < 0.001 ? v.toExponential(1) : v.toPrecision(3)) + '%';
  let html = `<div class="text-emerald-400 mb-1">GPU readback vs exact Liénard–Wiechert (no softening), synchrotron pulse sweep, 256 observers · kernel: ${kernel} · renderer: ${rendererName()}</div>`;
  html += '<table class="report-table w-full"><tr><th>β</th><th>γ</th><th>M_T</th><th>production</th><th>production, df64 query</th><th>legacy algorithm</th><th>ring (production)</th></tr>';
  for (const r of rows) {
    html += `<tr><td>${r.beta}</td><td>${r.gamma.toFixed(1)}</td><td>${r.M}</td><td class="text-emerald-300">${pct(r.prod.shockL2)}</td><td>${pct(r.prodDf.shockL2)}</td><td class="text-slate-400">${r.legacy ? pct(r.legacy.shockL2) : '—'}</td><td>${pct(r.prod.bulkL2)}</td></tr>`;
  }
  html += '</table><div class="mt-2 text-slate-400">"production" rounds observers to float32 and compares with the exact field at the rounded point (the preprint\'s metric). "df64 query" places observers with double-float coordinates. Legacy = the original algorithm (float32 root, Theorem 1 limiter, spline bisection). Above γ ≈ 100, float32 observer coordinates cannot resolve the pulse.</div>';
  report.innerHTML = html;
  $('btn-export-json').classList.remove('hidden');
  btn.disabled = false;
  ui.paused = false;
}

function rendererName() {
  if (!solver) return 'none';
  const gl = solver.gl, ext = gl.getExtension('WEBGL_debug_renderer_info');
  return String(ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER));
}

// ---------------------------------------------------------------- benchmark
function benchmark() {
  if (!solver) return;
  // The precision / certificate / root-search switches are those of the v6 kernel: run them on a v6 solver
  // (own canvas and context) when the main kernel is v6f.
  if (kernel === 'v6f' && !benchV6) {
    try { benchV6 = new GpuSolver(document.createElement('canvas'), kernelShaders('v6')); } catch { benchV6 = null; }
  }
  const v6s = kernel === 'v6' ? solver : benchV6;
  const time = (c: SolverConfig, frames: number, s: GpuSolver) => {
    const cv = s.gl.canvas as HTMLCanvasElement;
    if (s !== solver && (cv.width !== W || cv.height !== H)) { cv.width = W; cv.height = H; }
    const sg = s.gl, sSync = () => { const px = new Uint8Array(4); sg.readPixels(0, 0, 1, 1, sg.RGBA, sg.UNSIGNED_BYTE, px); };
    s.upload(sim.shellHistories, c.mT);
    s.render(W, H, c, { u_channel: 0, u_showWavePool: true, u_showImpostors: false, u_showMask: false, u_viewCenter: [ui.view.cx, ui.view.cy] }, { u_aspect: aspect(), u_viewHalfHeight: ui.view.halfH, u_time: sim.time, u_gain: 1, u_maskThreshold: 1.5 });
    sSync();
    const t0 = performance.now();
    for (let f = 0; f < frames; f++) s.render(W, H, c, {}, {});
    sSync();
    return (performance.now() - t0) / frames;
  };
  const variants: [string, Partial<SolverConfig>, GpuSolver | null][] = [
    [kernel === 'v6f' ? 'production, v6f (float32 incremental root, CPU chord deficit, float32 knot search + df64 check)' : 'production, v6 (double-float, |z′|² certificate, knots + 3 Newton)', {}, solver],
    ...(kernel === 'v6f' ? [['v6 (double-float, |z′|² certificate, knots + 3 Newton)', {}, v6s] as [string, Partial<SolverConfig>, GpuSolver | null]] : []),
    ['v6, float32 root, |z′|² certificate, knots + 3 Newton', { precision: 0 }, v6s],
    ['legacy algorithm (v6 switches: float32, Theorem 1, spline bisection + 2 Newton)', { precision: 0, certMode: 0, rootMode: 0, newton: 2 }, v6s],
  ];
  const frames = 5;
  const n = sim.sources.length;
  let html = `<div class="text-emerald-400 mb-1">Frame time at ${W}×${H} px, ${n} sources, M_T = ${cfg.mT} · renderer: ${rendererName()}</div><table class="report-table w-full"><tr><th>variant</th><th>ms/frame</th><th>pixel·source evaluations / s</th></tr>`;
  for (const [name, over, s] of variants) {
    if (!s) continue;
    const ms = time({ ...cfg, ...over }, frames, s);
    html += `<tr><td>${name}</td><td>${ms.toFixed(1)}</td><td>${((W * H * n) / (ms / 1000)).toExponential(2)}</td></tr>`;
  }
  html += '</table><div class="mt-2 text-slate-400">Software renderers (SwiftShader) are far slower than real GPUs; compare variants, not absolute numbers.</div>';
  $('report').innerHTML = html;
}

// ---------------------------------------------------------------- export
function download(name: string, text: string, type = 'text/csv') {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

function exportField() {
  if (!solver) return;
  const nx = 256, ny = Math.round(256 / aspect());
  const pts: { x: number; y: number }[] = [];
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const w = screenToWorld(((i + 0.5) / nx) * W, ((j + 0.5) / ny) * H);
    pts.push(w);
  }
  const f = solver.readFull(pts, cfg);
  const lines = [
    `# hopf-hough field export · t = ${sim.time} · c = ${C} · softening eps = ${cfg.softEps} · M_T = ${cfg.mT} · scenario = ${ui.scenario} · beta = ${ui.beta}`,
    '# quality: -1 = outside stored history; > 1.5 = under-resolved; otherwise resolved',
    'x,y,Ex,Ey,Bz,quality',
  ];
  pts.forEach((p, i) => lines.push(`${p.x},${p.y},${f[i].ex},${f[i].ey},${f[i].bz},${f[i].quality}`));
  download(`hopf-hough-field-t${sim.time.toFixed(3)}.csv`, lines.join('\n'));
}

// ---------------------------------------------------------------- input wiring
function wire() {
  const slider = $<HTMLInputElement>('slider-beta');
  slider.value = String(betaToSlider(ui.beta));
  slider.addEventListener('input', () => { ui.beta = sliderToBeta(+slider.value); applyScenario(); });
  const bindRange = (id: string, label: string, set: (v: number) => void, digits = 2) => {
    const el = $<HTMLInputElement>(id);
    el.addEventListener('input', () => { set(+el.value); $(label).textContent = (+el.value).toFixed(digits).replace(/\.?0+$/, (m) => (digits ? m : '')); });
  };
  bindRange('slider-n', 'val-n', (v) => { ui.n = v; applyScenario(); }, 0);
  bindRange('slider-omega', 'val-omega', (v) => { ui.omega = v; cfg.omega = v; applyScenario(); }, 1);
  bindRange('slider-L', 'val-L', (v) => { cfg.scaleL = v; sim.config.scaleL = v; sim.repopulate(); updateNotes(); }, 2);
  bindRange('slider-eps', 'val-eps', (v) => { cfg.softEps = v; sim.config.softEps = v; }, 3);

  $<HTMLSelectElement>('sel-channel').addEventListener('change', (e) => { ui.channel = +(e.target as HTMLSelectElement).value; drawColorbar(); });
  $<HTMLInputElement>('chk-wavefronts').addEventListener('change', (e) => { ui.wavefronts = (e.target as HTMLInputElement).checked; });
  $<HTMLInputElement>('chk-mask').addEventListener('change', (e) => { ui.mask = (e.target as HTMLInputElement).checked; });
  $<HTMLInputElement>('chk-impostors').addEventListener('change', (e) => { ui.impostors = (e.target as HTMLInputElement).checked; });
  $<HTMLInputElement>('chk-taper').addEventListener('change', (e) => { cfg.prefilter = (e.target as HTMLInputElement).checked; sim.config.prefilter = cfg.prefilter; sim.repopulate(); });

  const kbFor: Record<number, number> = { 128: 9, 512: 11, 1024: 12, 2048: 13 };
  $<HTMLSelectElement>('sel-mt').addEventListener('change', (e) => {
    cfg.mT = +(e.target as HTMLSelectElement).value; cfg.kBin = kbFor[cfg.mT];
    sim.config.mT = cfg.mT; sim.config.kBin = cfg.kBin; sim.repopulate(); updateNotes();
  });
  $<HTMLSelectElement>('sel-precision').addEventListener('change', (e) => { cfg.precision = +(e.target as HTMLSelectElement).value as 0 | 1; });
  $<HTMLSelectElement>('sel-cert').addEventListener('change', (e) => { cfg.certMode = +(e.target as HTMLSelectElement).value as 0 | 1 | 2; });
  $<HTMLSelectElement>('sel-root').addEventListener('change', (e) => {
    cfg.rootMode = +(e.target as HTMLSelectElement).value as 0 | 1; cfg.newton = cfg.rootMode === 1 ? 3 : 2;
  });

  $<HTMLSelectElement>('sel-scenario').addEventListener('change', (e) => {
    ui.scenario = (e.target as HTMLSelectElement).value as Scenario;
    $('custom-box').classList.toggle('hidden', ui.scenario !== 'custom');
    $('csv-box').classList.toggle('hidden', ui.scenario !== 'csv');
    if (ui.scenario === 'csv' && !ui.csvText) { updateNotes(); return; }
    applyScenario();
  });
  $('btn-apply-custom').addEventListener('click', () => {
    ui.customX = $<HTMLInputElement>('in-x').value; ui.customY = $<HTMLInputElement>('in-y').value; applyScenario();
  });
  $<HTMLInputElement>('in-csv').addEventListener('change', async (e) => {
    const f = (e.target as HTMLInputElement).files?.[0];
    if (!f) return;
    ui.csvText = await f.text();
    const ts = ui.csvText.split(/\r?\n/).map((l) => +l.split(/[,\s;]+/)[0]).filter((v) => Number.isFinite(v));
    ui.csvRange = ts.length ? [Math.min(...ts), Math.max(...ts)] : null;
    applyScenario();
  });

  $('btn-pan-mode').addEventListener('click', () => { ui.panMode = !ui.panMode; $('btn-pan-mode').textContent = ui.panMode ? 'Mode: pan view' : 'Mode: drag source'; });
  $('btn-reset-view').addEventListener('click', () => { ui.view = { cx: 0, cy: 0, halfH: 1.45 }; updateNotes(); });
  $('btn-check').addEventListener('click', checkAccuracy);
  $('btn-suite').addEventListener('click', () => { void runVerification(); });
  $('btn-bench').addEventListener('click', benchmark);
  $('btn-export').addEventListener('click', exportField);
  $('btn-export-json').addEventListener('click', () => { if (lastSuite) download('hopf-hough-suite.json', JSON.stringify(lastSuite, null, 2), 'application/json'); });

  overlay.addEventListener('pointerdown', (e) => {
    const r = overlay.getBoundingClientRect();
    const px = (e.clientX - r.left) * dpr, py = (e.clientY - r.top) * dpr;
    const w = screenToWorld(px, py);
    const pp = worldToScreen(ui.probe.x, ui.probe.y);
    if (Math.hypot(px - pp.x, py - pp.y) < 12 * dpr) { drag.what = 'probe'; }
    else if (ui.panMode) { drag.what = 'pan'; drag.start = { x: px, y: py, cx: ui.view.cx, cy: ui.view.cy }; }
    else { drag.what = 'source'; drag.active = true; drag.returning = true; drag.target = w; }
    overlay.setPointerCapture(e.pointerId);
  });
  overlay.addEventListener('pointermove', (e) => {
    if (!drag.what) return;
    const r = overlay.getBoundingClientRect();
    const px = (e.clientX - r.left) * dpr, py = (e.clientY - r.top) * dpr;
    const w = screenToWorld(px, py);
    if (drag.what === 'probe') ui.probe = w;
    else if (drag.what === 'source') drag.target = w;
    else if (drag.what === 'pan') {
      const s = (2 * ui.view.halfH) / H;
      ui.view.cx = drag.start.cx - (px - drag.start.x) * s;
      ui.view.cy = drag.start.cy + (py - drag.start.y) * s;
    }
  });
  const end = () => { if (drag.what === 'source') drag.active = false; drag.what = ''; };
  overlay.addEventListener('pointerup', end);
  overlay.addEventListener('pointercancel', end);
  overlay.addEventListener('wheel', (e) => {
    e.preventDefault();
    const r = overlay.getBoundingClientRect();
    const px = (e.clientX - r.left) * dpr, py = (e.clientY - r.top) * dpr;
    const before = screenToWorld(px, py);
    ui.view.halfH = Math.min(500, Math.max(1e-3, ui.view.halfH * Math.exp(e.deltaY * 0.0015)));
    const after = screenToWorld(px, py);
    ui.view.cx += before.x - after.x; ui.view.cy += before.y - after.y;
    updateNotes();
  }, { passive: false });
  window.addEventListener('resize', () => { resize(); drawColorbar(); });
}

// ---------------------------------------------------------------- main loop
let frames = 0, fpsT = performance.now(), probeT = 0;
function frame(now: number) {
  resize();
  if (!ui.paused) {
    const sub = 1;
    for (let k = 0; k < sub; k++) sim.step(cfg.dt, k < sub - 1);
    if (reseedPeriod > 0) {
      reseedClock += sub * cfg.dt;
      if (reseedClock > reseedPeriod) { trajectories = buildTrajectories(); sim.setTrajectories(trajectories); reseedClock = 0; }
    }
    if (ui.scenario === 'csv' && ui.csvRange && sim.time >= ui.csvRange[1]) ui.paused = true;
  }
  renderField();
  drawOverlay();
  if (now - probeT > 120) { probeReadout(); probeT = now; }
  frames++;
  if (now - fpsT > 500) { $('stat-fps').textContent = `${((now - fpsT) / frames).toFixed(1)} ms`; frames = 0; fpsT = now; }
  requestAnimationFrame(frame);
}

/** Shareable settings: ?scenario=orbit&beta=0.99&n=1&mt=2048&omega=1.6&eps=0&kernel=v6 */
function readUrlParams() {
  const q = new URLSearchParams(location.search);
  const num = (k: string) => (q.has(k) && Number.isFinite(+q.get(k)!) ? +q.get(k)! : null);
  const sc = q.get('scenario');
  if (sc && ['orbit', 'ensemble', 'oscillator', 'linear', 'hyperbolic', 'custom'].includes(sc)) ui.scenario = sc as Scenario;
  const b = num('beta'); if (b !== null && b > 0 && b < 1) ui.beta = b;
  const n = num('n'); if (n !== null) ui.n = Math.max(1, Math.min(256, Math.round(n)));
  const w = num('omega'); if (w !== null && w > 0) { ui.omega = w; cfg.omega = w; }
  const e = num('eps'); if (e !== null && e >= 0) cfg.softEps = e;
  const mt = num('mt'); const kb: Record<number, number> = { 128: 9, 512: 11, 1024: 12, 2048: 13 };
  if (mt !== null && kb[mt]) { cfg.mT = mt; cfg.kBin = kb[mt]; }
  const kn = q.get('kernel'); if (kn === 'v6' || kn === 'v6f') kernel = kn;
  if (q.get('x')) ui.customX = q.get('x')!;
  if (q.get('y')) ui.customY = q.get('y')!;
  // Reflect into the controls.
  ($('sel-scenario') as HTMLSelectElement).value = ui.scenario;
  $('custom-box').classList.toggle('hidden', ui.scenario !== 'custom');
  ($('in-x') as HTMLInputElement).value = ui.customX;
  ($('in-y') as HTMLInputElement).value = ui.customY;
  ($('slider-n') as HTMLInputElement).value = String(ui.n); $('val-n').textContent = String(ui.n);
  ($('slider-omega') as HTMLInputElement).value = String(ui.omega); $('val-omega').textContent = String(ui.omega);
  ($('slider-eps') as HTMLInputElement).value = String(cfg.softEps); $('val-eps').textContent = String(cfg.softEps);
  ($('sel-mt') as HTMLSelectElement).value = String(cfg.mT);
}

function main() {
  readUrlParams();
  resize();
  initGpu();
  applyScenario();
  wire();
  drawColorbar();
  // Exposed for scripting and automated tests.
  (window as unknown as Record<string, unknown>).hopfHough = { sim: () => sim, solver: () => solver, cfg, ui, runVerification, benchmark, checkAccuracy, exportField, legacyPreset, pulseCrossingSweep };
  requestAnimationFrame(frame);
}
main();
