/**
 * WebGL2 evaluator: uploads each source's shell history to RGBA32F textures and
 * evaluates the retarded LW field per fragment, either to the screen or to a
 * float framebuffer that is read back for verification and export.
 */
import type { ShellHistory } from '../core/shells.js';
import { MAX_M_T, MAX_SOURCES, type SolverConfig } from '../core/types.js';
import type { PointField } from '../core/suite.js';
import { LEGACY_FRAGMENT_SHADER, LEGACY_VERTEX_SHADER } from './legacyShader.js';

export interface ShaderSources { vertex: string; fragment: string }

export const LEGACY_SHADERS: ShaderSources = { vertex: LEGACY_VERTEX_SHADER, fragment: LEGACY_FRAGMENT_SHADER };

/** Width of the readback framebuffer (points per draw call). */
export const READBACK_WIDTH = 256;

export type UniformValue = number | boolean | [number, number] | [number, number, number, number];

export class GpuSolver {
  readonly gl: WebGL2RenderingContext;
  readonly program: WebGLProgram;
  private readonly uniformCache = new Map<string, WebGLUniformLocation | null>();
  private readonly texPosVel: WebGLTexture;
  private readonly texAccChg: WebGLTexture;
  private readonly texQuery: WebGLTexture;
  private readonly fbo: WebGLFramebuffer;
  private readonly posVelData = new Float32Array(MAX_M_T * MAX_SOURCES * 4);
  private readonly accChgData = new Float32Array(MAX_M_T * MAX_SOURCES * 4);
  private readonly queryData = new Float32Array(READBACK_WIDTH * 4);
  private readonly readData = new Float32Array(READBACK_WIDTH * 4);
  /** Optional extra textures uploaded by subclasses/extensions (unit → texture). */
  uploadedCount = 0;
  uploadedM = 0;

  constructor(canvas: HTMLCanvasElement | OffscreenCanvas, shaders: ShaderSources = LEGACY_SHADERS) {
    const gl = canvas.getContext('webgl2', { antialias: false, depth: false, stencil: false, preserveDrawingBuffer: true }) as WebGL2RenderingContext | null;
    if (!gl) throw new Error('WebGL2 is not available in this browser.');
    if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('EXT_color_buffer_float is required for float readback.');
    this.gl = gl;
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
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, READBACK_WIDTH, 1, 0, gl.RGBA, gl.FLOAT, null);
    const fbo = gl.createFramebuffer();
    if (!fbo) throw new Error('Could not create framebuffer.');
    this.fbo = fbo;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, verifyTex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    this.set('u_texPosVel', 0);
    this.set('u_texAccChg', 1);
    this.set('u_texQueryCoords', 2);
  }

  private link(src: ShaderSources): WebGLProgram {
    const gl = this.gl;
    const compile = (type: number, text: string) => {
      const sh = gl.createShader(type)!;
      gl.shaderSource(sh, text);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        throw new Error('Shader compile failed: ' + gl.getShaderInfoLog(sh));
      }
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

  /** Set a uniform by name (int for integers/booleans unless `asFloat`). */
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

  /** Upload shell histories (one texture row per source). */
  upload(histories: readonly ShellHistory[], M: number): void {
    const gl = this.gl;
    const rows = Math.max(1, histories.length);
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
      }
    }
    const n = M * rows * 4;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.texPosVel);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, M, rows, 0, gl.RGBA, gl.FLOAT, this.posVelData.subarray(0, n));
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.texAccChg);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, M, rows, 0, gl.RGBA, gl.FLOAT, this.accChgData.subarray(0, n));
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
    this.setFloat('u_c', cfg.c);
  }

  /**
   * Evaluate the field at arbitrary points through a float framebuffer.
   * Returns E and the first source's reconstructed emission point per query.
   */
  readback(points: readonly { x: number; y: number }[], cfg: SolverConfig): PointField[] {
    const gl = this.gl;
    gl.useProgram(this.program);
    this.setSolverUniforms(cfg);
    this.set('u_readbackMode', 1);
    const out: PointField[] = [];
    for (let start = 0; start < points.length; start += READBACK_WIDTH) {
      const count = Math.min(READBACK_WIDTH, points.length - start);
      this.queryData.fill(0);
      for (let i = 0; i < count; i++) {
        this.queryData[i * 4] = Math.fround(points[start + i].x);
        this.queryData[i * 4 + 1] = Math.fround(points[start + i].y);
      }
      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_2D, this.texQuery);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, READBACK_WIDTH, 1, 0, gl.RGBA, gl.FLOAT, this.queryData);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
      gl.viewport(0, 0, READBACK_WIDTH, 1);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.readPixels(0, 0, READBACK_WIDTH, 1, gl.RGBA, gl.FLOAT, this.readData);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      for (let i = 0; i < count; i++) {
        out.push({ ex: this.readData[i * 4], ey: this.readData[i * 4 + 1], emitX: this.readData[i * 4 + 2], emitY: this.readData[i * 4 + 3] });
      }
    }
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
