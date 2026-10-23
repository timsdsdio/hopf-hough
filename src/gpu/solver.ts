/**
 * WebGL2 evaluator: uploads each source's shell history to RGBA32F textures and
 * evaluates the retarded LW field per fragment, either to the screen or to a
 * float framebuffer that is read back for verification and export.
 */
import type { ShellHistory } from '../core/shells.js';
import { MAX_M_T, MAX_SOURCES, type SolverConfig } from '../core/types.js';
import type { PointField } from '../core/suite.js';
import { LEGACY_FRAGMENT_SHADER, LEGACY_VERTEX_SHADER } from './legacyShader.js';
import { FRAGMENT_SHADER, VERTEX_SHADER } from './shader.js';

export interface ShaderSources { vertex: string; fragment: string; production: boolean }

export const LEGACY_SHADERS: ShaderSources = { vertex: LEGACY_VERTEX_SHADER, fragment: LEGACY_FRAGMENT_SHADER, production: false };
export const PRODUCTION_SHADERS: ShaderSources = { vertex: VERTEX_SHADER, fragment: FRAGMENT_SHADER, production: true };

export type UniformValue = number | boolean | [number, number] | [number, number, number, number];

/** Field, magnetic field and resolution indicator at one point (production readback). */
export interface FullField {
  ex: number;
  ey: number;
  /** Out-of-plane magnetic field B_z (Gaussian units, same scale as E). */
  bz: number;
  /**
   * Resolution indicator: max over sources of ΔR·γ/ρ at the retarded point
   * (shell spacing × γ over the trajectory's radius of curvature). Values above ~1.5
   * mean the history is too coarse there; −1 means the retarded point is outside the
   * stored history (answer not valid).
   */
  quality: number;
}

export class GpuSolver {
  readonly gl: WebGL2RenderingContext;
  readonly program: WebGLProgram;
  readonly production: boolean;
  /** Readback tile size (points per draw = width × height). */
  readonly readW: number;
  readonly readH: number;
  private readonly uniformCache = new Map<string, WebGLUniformLocation | null>();
  private readonly texPosVel: WebGLTexture;
  private readonly texAccChg: WebGLTexture;
  private readonly texQuery: WebGLTexture;
  private readonly texPosLo: WebGLTexture | null;
  private readonly fbo: WebGLFramebuffer;
  private readonly posVelData = new Float32Array(MAX_M_T * MAX_SOURCES * 4);
  private readonly accChgData = new Float32Array(MAX_M_T * MAX_SOURCES * 4);
  private readonly posLoData: Float32Array | null;
  private readonly queryData: Float32Array;
  private readonly readData: Float32Array;
  uploadedCount = 0;
  uploadedM = 0;

  constructor(canvas: HTMLCanvasElement | OffscreenCanvas, shaders: ShaderSources = PRODUCTION_SHADERS) {
    const gl = canvas.getContext('webgl2', { antialias: false, depth: false, stencil: false, preserveDrawingBuffer: true }) as WebGL2RenderingContext | null;
    if (!gl) throw new Error('WebGL2 is not available in this browser.');
    if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('EXT_color_buffer_float is required for float readback.');
    this.gl = gl;
    this.production = shaders.production;
    this.readW = 256;
    this.readH = shaders.production ? 64 : 1;
    this.queryData = new Float32Array(this.readW * this.readH * 4);
    this.readData = new Float32Array(this.readW * this.readH * 4);
    this.posLoData = shaders.production ? new Float32Array(MAX_M_T * MAX_SOURCES * 4) : null;
    this.program = this.link(shaders);
    gl.useProgram(this.program);

    const quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(this.program, 'a_pos');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

    this.texPosVel = this.dataTexture(0);
    this.texAccChg = this.dataTexture(1);
    this.texQuery = this.dataTexture(2);
    const verifyTex = this.dataTexture(3);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, this.readW, this.readH, 0, gl.RGBA, gl.FLOAT, null);
    this.texPosLo = shaders.production ? this.dataTexture(4) : null;
    const fbo = gl.createFramebuffer();
    if (!fbo) throw new Error('Could not create framebuffer.');
    this.fbo = fbo;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, verifyTex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    this.set('u_texPosVel', 0);
    this.set('u_texAccChg', 1);
    this.set('u_texQueryCoords', 2);
    if (shaders.production) { this.set('u_texPosLo', 4); this.setFloat('u_one', 1.0); this.setUint('u_zero', 0); }
  }

  private link(src: ShaderSources): WebGLProgram {
    const gl = this.gl;
    const compile = (type: number, text: string) => {
      const sh = gl.createShader(type)!;
      gl.shaderSource(sh, text);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error('Shader compile failed: ' + gl.getShaderInfoLog(sh));
      return sh;
    };
    const prog = gl.createProgram()!;
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, src.vertex));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, src.fragment));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error('Shader link failed: ' + gl.getProgramInfoLog(prog));
    return prog;
  }

  private dataTexture(unit: number): WebGLTexture {
    const gl = this.gl;
    const tex = gl.createTexture()!;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
  }

  /** Set a uniform by name (int for integers/booleans unless `asFloat`). Unknown names are ignored. */
  set(name: string, value: UniformValue, asFloat = false): void {
    const gl = this.gl;
    let loc = this.uniformCache.get(name);
    if (loc === undefined) { loc = gl.getUniformLocation(this.program, name); this.uniformCache.set(name, loc); }
    if (loc === null) return;
    if (typeof value === 'boolean') gl.uniform1i(loc, value ? 1 : 0);
    else if (typeof value === 'number') { if (asFloat) gl.uniform1f(loc, value); else gl.uniform1i(loc, value); }
    else if (value.length === 2) gl.uniform2f(loc, value[0], value[1]);
    else gl.uniform4f(loc, value[0], value[1], value[2], value[3]);
  }

  setFloat(name: string, value: number): void { this.set(name, value, true); }

  setUint(name: string, value: number): void {
    const gl = this.gl;
    let loc = this.uniformCache.get(name);
    if (loc === undefined) { loc = gl.getUniformLocation(this.program, name); this.uniformCache.set(name, loc); }
    if (loc !== null) gl.uniform1ui(loc, value);
  }

  /** Upload shell histories (one texture row per source). */
  upload(histories: readonly ShellHistory[], M: number): void {
    const gl = this.gl;
    const rows = Math.max(1, histories.length);
    const lo = this.posLoData;
    for (let s = 0; s < histories.length; s++) {
      const h = histories[s];
      const off = s * M * 4;
      for (let m = 0; m < M; m++) {
        const i = off + m * 4;
        this.posVelData[i] = h.emitX[m];
        this.posVelData[i + 1] = h.emitY[m];
        this.posVelData[i + 2] = h.emitUx[m];
        this.posVelData[i + 3] = h.emitUy[m];
        this.accChgData[i] = h.emitBdx[m];
        this.accChgData[i + 1] = h.emitBdy[m];
        this.accChgData[i + 2] = h.charge;
        this.accChgData[i + 3] = h.emitR[m];
        if (lo) { lo[i] = h.emitXlo[m]; lo[i + 1] = h.emitYlo[m]; lo[i + 2] = 0; lo[i + 3] = 0; }
      }
    }
    const n = M * rows * 4;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.texPosVel);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, M, rows, 0, gl.RGBA, gl.FLOAT, this.posVelData.subarray(0, n));
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.texAccChg);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, M, rows, 0, gl.RGBA, gl.FLOAT, this.accChgData.subarray(0, n));
    if (lo && this.texPosLo) {
      gl.activeTexture(gl.TEXTURE4);
      gl.bindTexture(gl.TEXTURE_2D, this.texPosLo);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, M, rows, 0, gl.RGBA, gl.FLOAT, lo.subarray(0, n));
    }
    this.uploadedCount = histories.length;
    this.uploadedM = M;
  }

  private setSolverUniforms(cfg: SolverConfig): void {
    this.set('u_activeCount', this.uploadedCount);
    this.set('u_mT', cfg.mT);
    this.set('u_kBin', cfg.kBin);
    this.set('u_velMode', cfg.velMode);
    this.setFloat('u_scaleL', cfg.scaleL);
    this.setFloat('u_softEps', cfg.softEps);
    if (this.production) {
      this.setFloat('u_c', cfg.c);
      this.set('u_newton', cfg.newton);
      this.set('u_certMode', cfg.certMode);
      this.set('u_rootMode', cfg.rootMode);
      this.set('u_precision', cfg.precision);
      this.setFloat('u_betaMax', cfg.betaMax);
      this.setFloat('u_one', 1.0);
      this.setUint('u_zero', 0);
      this.set('u_taperOn', cfg.prefilter && cfg.gridMode === 'compactified');
      this.setFloat('u_omega', cfg.omega);
      this.setFloat('u_ringReach', cfg.c * cfg.dt * (cfg.ringSize - 4));
    }
  }

  /** Raw readback: 4 floats per point in the requested layout. */
  readRaw(points: readonly { x: number; y: number; x64?: number; y64?: number }[], cfg: SolverConfig, layout: 0 | 1 | 2): Float32Array {
    const gl = this.gl;
    gl.useProgram(this.program);
    this.setSolverUniforms(cfg);
    this.set('u_readbackMode', 1);
    this.set('u_readbackLayout', layout);
    const tile = this.readW * this.readH;
    const out = new Float32Array(points.length * 4);
    for (let start = 0; start < points.length; start += tile) {
      const count = Math.min(tile, points.length - start);
      const rowsUsed = Math.ceil(count / this.readW);
      this.queryData.fill(0);
      for (let i = 0; i < count; i++) {
        const p = points[start + i];
        const hx = Math.fround(p.x), hy = Math.fround(p.y);
        this.queryData[i * 4] = hx;
        this.queryData[i * 4 + 1] = hy;
        // Low parts let callers place observers to better than float32 resolution (production shader only).
        this.queryData[i * 4 + 2] = p.x64 !== undefined ? Math.fround(p.x64 - hx) : 0;
        this.queryData[i * 4 + 3] = p.y64 !== undefined ? Math.fround(p.y64 - hy) : 0;
      }
      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_2D, this.texQuery);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, this.readW, this.readH, 0, gl.RGBA, gl.FLOAT, this.queryData);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
      gl.viewport(0, 0, this.readW, rowsUsed);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.readPixels(0, 0, this.readW, rowsUsed, gl.RGBA, gl.FLOAT, this.readData);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      out.set(this.readData.subarray(0, count * 4), start * 4);
    }
    return out;
  }

  /** Legacy-compatible readback: E and the first source's reconstructed emission point. */
  readback(points: readonly { x: number; y: number; x64?: number; y64?: number }[], cfg: SolverConfig): PointField[] {
    const raw = this.readRaw(points, cfg, 0);
    const out: PointField[] = new Array(points.length);
    for (let i = 0; i < points.length; i++) out[i] = { ex: raw[i * 4], ey: raw[i * 4 + 1], emitX: raw[i * 4 + 2], emitY: raw[i * 4 + 3] };
    return out;
  }

  /** E, B_z and the resolution indicator at arbitrary points (production shader). */
  readFull(points: readonly { x: number; y: number; x64?: number; y64?: number }[], cfg: SolverConfig): FullField[] {
    if (!this.production) throw new Error('readFull needs the production shader.');
    const raw = this.readRaw(points, cfg, 1);
    const out: FullField[] = new Array(points.length);
    for (let i = 0; i < points.length; i++) out[i] = { ex: raw[i * 4], ey: raw[i * 4 + 1], bz: raw[i * 4 + 2], quality: raw[i * 4 + 3] };
    return out;
  }

  /** Draw the field to the canvas. Extra uniforms (view, channel, time …) are passed through. */
  render(width: number, height: number, cfg: SolverConfig, uniforms: Record<string, UniformValue> = {}, floats: Record<string, number> = {}): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, width, height);
    gl.useProgram(this.program);
    this.setSolverUniforms(cfg);
    this.set('u_readbackMode', 0);
    for (const [k, v] of Object.entries(uniforms)) this.set(k, v);
    for (const [k, v] of Object.entries(floats)) this.setFloat(k, v);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }
}
