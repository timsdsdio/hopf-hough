'use strict';
        const MAX_M_T = 2048;
        const MAX_PARTICLES = 256;
        const C_LIGHT = 2.35;
        const RING_SIZE = 16384;
        const DT_STEP = 0.00675;
        const f32 = Math.fround;

        // Strict IEEE-754 Float32 2D Euclidean norm (matches GLSL length(vec2) without Float64 intermediates!)
        function f32Len(x, y) {
            const xx = f32(x * x);
            const yy = f32(y * y);
            return f32(Math.sqrt(f32(xx + yy)));
        }

        // Smooth Slider [0, 1000] <-> Beta [0.05, 0.90]
        function sliderToBeta(sliderVal) {
            const u = Math.max(0, Math.min(1, sliderVal / 1000.0));
            const oneMinusBeta = 0.95 * Math.pow(0.10 / 0.95, u);
            return Math.min(0.90, Math.max(0.05, 1.0 - oneMinusBeta));
        }

        function betaToSlider(beta) {
            const clamped = Math.min(0.90, Math.max(0.05, beta));
            const u = Math.log((1.0 - clamped) / 0.95) / Math.log(0.10 / 0.95);
            return Math.round(Math.max(0, Math.min(1000, u * 1000)));
        }

        const state = {
            mT: 128,
            kBin: 9,
            numParticles: 32,
            beta: 0.80,
            gamma: 1 / Math.sqrt((1 - 0.80) * (1 + 0.80)),
            omega: 1.6,
            scaleL: 1.35,
            softEps: 0.06,
            motionMode: 'swarm',
            channel: 'total',
            subviewMode: 'shock_crest',
            velMode: 2, // 0: Raw Beta, 1: Arc-Length Norm Beta, 2: Proper 4-Vel u = gamma*beta + Spline-Consistent Direction
            useDiffusiveEulerian: false,
            stressTestMode: true,
            showWavePool: true,
            showImpostors: true,
            showDilatingFibers: true,
            paused: false,
            time: 2.50,
            fps: 60,
            probePos: { x: 0.95, y: 0.40 },
            interactionTarget: null,
            draggedSourceTarget: { x: 0, y: 0 },
            sources: []
        };

        function getExactTrajectoryKinematics(partIdx, totalParts, t, betaOverride, motionOverride) {
            const betaVal = betaOverride !== undefined ? betaOverride : state.beta;
            const modeVal = motionOverride !== undefined ? motionOverride : state.motionMode;
            const phaseOffset = (partIdx / Math.max(1, totalParts)) * 2 * Math.PI;
            const charge = (partIdx % 2 === 0 || modeVal === 'synchrotron') ? 1.0 : -1.0;

            if (modeVal === 'synchrotron') {
                const omega = state.omega;
                const v = betaVal * C_LIGHT;
                const rOrb = Math.min(0.85, v / Math.max(0.5, omega));
                const actualOmega = v / Math.max(0.05, rOrb);
                const ang = actualOmega * t + phaseOffset;
                const cosA = Math.cos(ang);
                const sinA = Math.sin(ang);

                return {
                    x: rOrb * cosA,
                    y: rOrb * sinA,
                    bx: -betaVal * sinA,
                    by: betaVal * cosA,
                    bdx: (-betaVal * actualOmega * cosA) / C_LIGHT,
                    bdy: (-betaVal * actualOmega * sinA) / C_LIGHT,
                    charge: 1.0
                };
            } else if (modeVal === 'uniform') {
                const period = 4.4;
                const wrappedT = ((t % period) + period) % period - 0.5 * period;
                const yOff = totalParts > 1 ? ((partIdx / (totalParts - 1)) - 0.5) * 1.2 : 0.0;
                const bEff = betaVal * 0.55;
                const vx = bEff * C_LIGHT;
                return {
                    x: vx * wrappedT,
                    y: yOff,
                    bx: bEff,
                    by: 0.0,
                    bdx: 0.0,
                    bdy: 0.0,
                    charge
                };
            } else if (modeVal === 'swarm') {
                // Multi-body Keplerian / Atomic Rosette Swarm:
                // Each particle orbits on a golden-ratio tilted rosette / precessing ellipse
                const goldenAngle = partIdx * 2.399963229728653;
                const shellLayer = (partIdx % 4) / 3.0;
                const rBase = 0.28 + 0.58 * shellLayer;
                const ecc = 0.28 * Math.sin(partIdx * 1.7 + 0.5);
                const orbitFreq = state.omega * (0.75 + 0.55 * (1.0 - shellLayer * 0.5));
                const precessFreq = 0.22 * state.omega * ((partIdx % 2 === 0) ? 1.0 : -1.0);

                const theta = orbitFreq * t + goldenAngle;
                const phiTilt = precessFreq * t + goldenAngle * 0.5;

                // Unrotated ellipse in local orbital plane (u, v)
                const aSemi = rBase;
                const bSemi = rBase * Math.sqrt(Math.max(0.25, 1.0 - ecc * ecc));
                const cosT = Math.cos(theta), sinT = Math.sin(theta);
                const u = aSemi * cosT;
                const v = bSemi * sinT;
                const du = -aSemi * orbitFreq * sinT;
                const dv =  bSemi * orbitFreq * cosT;
                const d2u = -aSemi * orbitFreq * orbitFreq * cosT;
                const d2v = -bSemi * orbitFreq * orbitFreq * sinT;

                const cosP = Math.cos(phiTilt), sinP = Math.sin(phiTilt);
                const x = u * cosP - v * sinP;
                const y = u * sinP + v * cosP;

                const vxRaw = (du * cosP - dv * sinP) - precessFreq * y;
                const vyRaw = (du * sinP + dv * cosP) + precessFreq * x;
                const axRaw = (d2u * cosP - d2v * sinP) - 2.0 * precessFreq * vyRaw - precessFreq * precessFreq * x;
                const ayRaw = (d2u * sinP + d2v * cosP) + 2.0 * precessFreq * vxRaw - precessFreq * precessFreq * y;

                // Scale speed smoothly so max swarm speed respects betaVal
                const speedRaw = Math.hypot(vxRaw, vyRaw);
                const maxAllowedV = betaVal * C_LIGHT;
                const vScale = speedRaw > maxAllowedV ? (maxAllowedV / speedRaw) : 1.0;

                return {
                    x,
                    y,
                    bx: (vxRaw * vScale) / C_LIGHT,
                    by: (vyRaw * vScale) / C_LIGHT,
                    bdx: (axRaw * vScale) / (C_LIGHT * C_LIGHT),
                    bdy: (ayRaw * vScale) / (C_LIGHT * C_LIGHT),
                    charge
                };
            } else {
                const omega = state.omega;
                const maxA = Math.min(0.75, (betaVal * C_LIGHT) / Math.max(0.4, omega));
                const ang = omega * t + phaseOffset;
                const sinA = Math.sin(ang);
                const cosA = Math.cos(ang);
                const yOff = totalParts > 1 ? ((partIdx / (totalParts - 1)) - 0.5) * 1.1 : 0.0;
                return {
                    x: maxA * sinA,
                    y: yOff,
                    bx: (maxA * omega * cosA) / C_LIGHT,
                    by: 0.0,
                    bdx: (-maxA * omega * omega * sinA) / (C_LIGHT * C_LIGHT),
                    bdy: 0.0,
                    charge
                };
            }
        }

        // Exact Retarded-Tangent Coordinate Construction (Eq. 8):
        function getExactShockCrestPoint(rRetDesired, tObs, betaOverride) {
            const tRetStar = tObs - rRetDesired / C_LIGHT;
            const kin = getExactTrajectoryKinematics(0, 1, tRetStar, betaOverride, 'synchrotron');
            const bMag = Math.hypot(kin.bx, kin.by);
            const bHatX = bMag > 1e-12 ? kin.bx / bMag : 1.0;
            const bHatY = bMag > 1e-12 ? kin.by / bMag : 0.0;
            const crestX = kin.x + rRetDesired * bHatX;
            const crestY = kin.y + rRetDesired * bHatY;
            return {
                x: crestX,
                y: crestY,
                rObs: Math.hypot(crestX, crestY),
                phiObs: Math.atan2(crestY, crestX),
                emitX: kin.x,
                emitY: kin.y,
                bHatX,
                bHatY,
                tRetStar
            };
        }

        function getPulseCrossingSweepPoints(rRetDesired, tObs, betaVal, nSamples) {
            const gammaVal = 1.0 / Math.sqrt((1.0 - betaVal) * (1.0 + betaVal));
            const crest = getExactShockCrestPoint(rRetDesired, tObs, betaVal);
            const halfSweep = Math.min(0.45, Math.max(1e-6, 4.0 * Math.pow(gammaVal, -3.0)));
            const pts = [];
            for (let k = 0; k < nSamples; k++) {
                const u = (k / (nSamples - 1)) * 2.0 - 1.0;
                const ang = crest.phiObs + u * halfSweep;
                const x64 = crest.rObs * Math.cos(ang);
                const y64 = crest.rObs * Math.sin(ang);
                pts.push({
                    x: f32(x64),
                    y: f32(y64),
                    x64,
                    y64
                });
            }
            return { pts, halfSweep, crest };
        }

        function createCompactifiedSource(idx, totalParts, t0) {
            const src = {
                x: 0, y: 0, bx: 0, by: 0, bdx: 0, bdy: 0, charge: 1.0,
                 returningToOrbit: false,
                emitX: new Float32Array(MAX_M_T),
                emitY: new Float32Array(MAX_M_T),
                emitUx: new Float32Array(MAX_M_T),
                emitUy: new Float32Array(MAX_M_T),
                emitBdx: new Float32Array(MAX_M_T),
                emitBdy: new Float32Array(MAX_M_T),
                emitR: new Float32Array(MAX_M_T),
                emitX64: new Float64Array(MAX_M_T),
                emitY64: new Float64Array(MAX_M_T),
                emitUx64: new Float64Array(MAX_M_T),
                emitUy64: new Float64Array(MAX_M_T),
                emitBdx64: new Float64Array(MAX_M_T),
                emitBdy64: new Float64Array(MAX_M_T),
                emitR64: new Float64Array(MAX_M_T),
                nextX: new Float32Array(MAX_M_T),
                nextY: new Float32Array(MAX_M_T),
                nextUx: new Float32Array(MAX_M_T),
                nextUy: new Float32Array(MAX_M_T),
                nextBdx: new Float32Array(MAX_M_T),
                nextBdy: new Float32Array(MAX_M_T),
                ringHead: 0,
                ringX: new Float64Array(RING_SIZE),
                ringY: new Float64Array(RING_SIZE),
                ringBx: new Float64Array(RING_SIZE),
                ringBy: new Float64Array(RING_SIZE),
                ringBdx: new Float64Array(RING_SIZE),
                ringBdy: new Float64Array(RING_SIZE)
            };
            seedExactRetardedHistory(src, idx, totalParts, t0);
            return src;
        }

        function packVelocityForStorage(bx, by, velMode) {
            if (velMode === 2) {
                const bMag = Math.min(0.9999995, Math.hypot(bx, by));
                const gamma = 1.0 / Math.sqrt(Math.max(1e-14, (1.0 - bMag) * (1.0 + bMag)));
                return { ux: f32(gamma * bx), uy: f32(gamma * by), ux64: gamma * bx, uy64: gamma * by };
            }
            return { ux: f32(bx), uy: f32(by), ux64: bx, uy64: by };
        }

        // Fixed CPU Ring Buffer Sampler:
        // Uses 2-point Kinematical Cubic Hermite interpolation on [k1, k2] for position (with h11 * ty2 fixed!)
        // + 4-tap Catmull-Rom with Arc-Length Speed Preservation for velocity.
        function sampleFromRingBuffer(src, stepsBackFloat) {
            const sClamped = Math.max(0.0, Math.min(RING_SIZE - 3.001, stepsBackFloat));
            const k1 = Math.floor(sClamped);
            const k0 = Math.max(0, k1 - 1);
            const k2 = Math.min(RING_SIZE - 1, k1 + 1);
            const k3 = Math.min(RING_SIZE - 1, k1 + 2);

            const t = sClamped - k1;
            const t2 = t * t;
            const t3 = t2 * t;

            const i0 = (src.ringHead - k0 + RING_SIZE) % RING_SIZE;
            const i1 = (src.ringHead - k1 + RING_SIZE) % RING_SIZE;
            const i2 = (src.ringHead - k2 + RING_SIZE) % RING_SIZE;
            const i3 = (src.ringHead - k3 + RING_SIZE) % RING_SIZE;

            const stepScale = C_LIGHT * DT_STEP;
            const tx1 = -src.ringBx[i1] * stepScale;
            const ty1 = -src.ringBy[i1] * stepScale;
            const tx2 = -src.ringBx[i2] * stepScale;
            const ty2 = -src.ringBy[i2] * stepScale;

            const h00 =  2.0 * t3 - 3.0 * t2 + 1.0;
            const h10 =        t3 - 2.0 * t2 + t;
            const h01 = -2.0 * t3 + 3.0 * t2;
            const h11 =        t3 -       t2;

            const x = h00 * src.ringX[i1] + h10 * tx1 + h01 * src.ringX[i2] + h11 * tx2;
            const y = h00 * src.ringY[i1] + h10 * ty1 + h01 * src.ringY[i2] + h11 * ty2;

            const w0 = -0.5 * t3 + t2 - 0.5 * t;
            const w1 =  1.5 * t3 - 2.5 * t2 + 1.0;
            const w2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
            const w3 =  0.5 * t3 - 0.5 * t2;

            let rawBx, rawBy, bdx, bdy;
            if (k1 === 0) {
                const bdx1 = -src.ringBdx[i1] * stepScale;
                const bdy1 = -src.ringBdy[i1] * stepScale;
                const bdx2 = -src.ringBdx[i2] * stepScale;
                const bdy2 = -src.ringBdy[i2] * stepScale;
                rawBx = h00 * src.ringBx[i1] + h10 * bdx1 + h01 * src.ringBx[i2] + h11 * bdx2;
                rawBy = src.ringBy[i1] * h00 + h10 * bdy1 + h01 * src.ringBy[i2] + h11 * bdy2;
                bdx = src.ringBdx[i1] + t * (src.ringBdx[i2] - src.ringBdx[i1]);
                bdy = src.ringBdy[i1] + t * (src.ringBdy[i2] - src.ringBdy[i1]);
            } else {
                rawBx = src.ringBx[i0] * w0 + src.ringBx[i1] * w1 + src.ringBx[i2] * w2 + src.ringBx[i3] * w3;
                rawBy = src.ringBy[i0] * w0 + src.ringBy[i1] * w1 + src.ringBy[i2] * w2 + src.ringBy[i3] * w3;
                bdx = src.ringBdx[i0] * w0 + src.ringBdx[i1] * w1 + src.ringBdx[i2] * w2 + src.ringBdx[i3] * w3;
                bdy = src.ringBdy[i0] * w0 + src.ringBdy[i1] * w1 + src.ringBdy[i2] * w2 + src.ringBdy[i3] * w3;
            }

            const bMag1 = Math.hypot(src.ringBx[i1], src.ringBy[i1]);
            const bMag2 = Math.hypot(src.ringBx[i2], src.ringBy[i2]);
            const bTarget = bMag1 + t * (bMag2 - bMag1);
            const rawBMag = Math.hypot(rawBx, rawBy);
            const bScale = rawBMag > 1e-12 ? (bTarget / rawBMag) : 1.0;

            return {
                x,
                y,
                bx: rawBx * bScale,
                by: rawBy * bScale,
                bdx,
                bdy
            };
        }

        // Populates the compactified shells directly from the evolved CPU Ring Buffer
        // gridMode: 'compactified' (default), 'uniform', 'log' (exponential adaptive), or 'power2' (quadratic adaptive)
        function populateShellsFromRingBuffer(src, M, velMode, dt, uniformMaxR, gridMode) {
            const dTheta = Math.PI / M;
            const orbitLambda = (2.0 * Math.PI * C_LIGHT) / Math.max(0.5, state.omega);
            const gMode = gridMode || (uniformMaxR !== undefined && uniformMaxR > 0 ? 'uniform' : 'compactified');
            const rMaxTarget = uniformMaxR || 220.0;

            for (let m = 0; m < M; m++) {
                const u = (m + 0.5) / M;
                let rShell64, rShellF32, dR;
                if (gMode === 'uniform') {
                    dR = rMaxTarget / M;
                    rShell64 = (m + 0.5) * dR;
                    rShellF32 = f32(rShell64);
                } else if (gMode === 'log') {
                    // Exponential / Logarithmic Adaptive Grid: R(u) = r0 * (exp(alpha * u) - 1) spanning [0, rMaxTarget]
                    const r0 = 0.05;
                    const alpha = Math.log(1.0 + rMaxTarget / r0);
                    rShell64 = r0 * (Math.exp(alpha * u) - 1.0);
                    rShellF32 = f32(rShell64);
                    dR = (r0 * alpha * Math.exp(alpha * u)) / M;
                } else if (gMode === 'power2') {
                    // Quadratic / Power-Law Adaptive Grid: R(u) = rMaxTarget * u^2 spanning [0, rMaxTarget]
                    rShell64 = rMaxTarget * u * u;
                    rShellF32 = f32(rShell64);
                    dR = (2.0 * rMaxTarget * u) / M;
                } else {
                    const th = (m + 0.5) * dTheta;
                    rShell64 = state.scaleL * Math.tan(0.5 * th);
                    rShellF32 = f32(f32(state.scaleL) * f32(Math.tan(f32(0.5 * th))));
                    dR = (state.scaleL * dTheta) / (1.0 + Math.cos(th));
                }

                const stepsBackCenter = (rShellF32 / C_LIGHT) / dt;
                const stepsBackCenter64 = (rShell64 / C_LIGHT) / dt;
                const filterRatio = dR / (0.5 * orbitLambda);

                let sample, sample64;
                if (filterRatio > 0.25 && gMode === 'compactified') {
                    const dSteps = Math.min((dR / C_LIGHT) / dt, (2.0 * Math.PI / state.omega) / dt);
                    const s0 = sampleFromRingBuffer(src, stepsBackCenter - 0.35 * dSteps);
                    const s1 = sampleFromRingBuffer(src, stepsBackCenter);
                    const s2 = sampleFromRingBuffer(src, stepsBackCenter + 0.35 * dSteps);
                    const wNyq = Math.exp(-Math.pow(filterRatio, 4.0));

                    const avgBx = 0.25 * s0.bx + 0.5 * s1.bx + 0.25 * s2.bx;
                    const avgBy = 0.25 * s0.by + 0.5 * s1.by + 0.25 * s2.by;
                    const bTarget = 0.25 * Math.hypot(s0.bx, s0.by) + 0.5 * Math.hypot(s1.bx, s1.by) + 0.25 * Math.hypot(s2.bx, s2.by);
                    const avgBMag = Math.hypot(avgBx, avgBy);
                    const scaleB = avgBMag > 1e-12 ? (bTarget / avgBMag) : 1.0;

                    sample = {
                        x: s1.x,
                        y: s1.y,
                        bx: avgBx * scaleB,
                        by: avgBy * scaleB,
                        bdx: (0.25 * s0.bdx + 0.5 * s1.bdx + 0.25 * s2.bdx) * wNyq,
                        bdy: (0.25 * s0.bdy + 0.5 * s1.bdy + 0.25 * s2.bdy) * wNyq
                    };
                    sample64 = sampleFromRingBuffer(src, stepsBackCenter64);
                } else {
                    sample = sampleFromRingBuffer(src, stepsBackCenter);
                    sample64 = sampleFromRingBuffer(src, stepsBackCenter64);
                }

                const packedV = packVelocityForStorage(sample.bx, sample.by, velMode);
                const packedV64 = packVelocityForStorage(sample64.bx, sample64.by, velMode);

                src.emitX[m] = f32(sample.x);
                src.emitY[m] = f32(sample.y);
                src.emitUx[m] = packedV.ux;
                src.emitUy[m] = packedV.uy;
                src.emitBdx[m] = f32(sample.bdx);
                src.emitBdy[m] = f32(sample.bdy);
                src.emitR[m] = rShellF32;

                src.emitX64[m] = sample64.x;
                src.emitY64[m] = sample64.y;
                src.emitUx64[m] = packedV64.ux64;
                src.emitUy64[m] = packedV64.uy64;
                src.emitBdx64[m] = sample64.bdx;
                src.emitBdy64[m] = sample64.bdy;
                src.emitR64[m] = rShell64;
            }

            const maxSub = f32(0.99995);
            for (let m = 0; m < M - 1; m++) {
                const r1 = src.emitR[m];
                const r2 = src.emitR[m + 1];
                const dR = f32(Math.max(f32(1e-6), f32(r2 - r1)));
                const dx = f32(src.emitX[m + 1] - src.emitX[m]);
                const dy = f32(src.emitY[m + 1] - src.emitY[m]);
                const chordSpeed = f32(f32Len(dx, dy) / dR);
                if (chordSpeed > maxSub) {
                    const shrink = f32(maxSub / chordSpeed);
                    src.emitX[m + 1] = f32(src.emitX[m] + f32(dx * shrink));
                    src.emitY[m + 1] = f32(src.emitY[m] + f32(dy * shrink));
                }
            }
        }

        function seedExactRetardedHistory(src, idx, totalParts, t0, betaOverride, motionOverride, mTOverride, velModeOverride, uniformMaxR, gridMode) {
            const M = mTOverride || state.mT;
            const vMode = velModeOverride !== undefined ? velModeOverride : state.velMode;
            const nowKin = getExactTrajectoryKinematics(idx, totalParts, t0, betaOverride, motionOverride);
            src.x = nowKin.x; src.y = nowKin.y;
            src.bx = nowKin.bx; src.by = nowKin.by;
            src.bdx = nowKin.bdx; src.bdy = nowKin.bdy;
            src.charge = nowKin.charge;

            src.ringHead = 0;
            for (let k = 0; k < RING_SIZE; k++) {
                const tPast = t0 - k * DT_STEP;
                const kin = getExactTrajectoryKinematics(idx, totalParts, tPast, betaOverride, motionOverride);
                const rIdx = (RING_SIZE - k) % RING_SIZE;
                src.ringX[rIdx] = kin.x;
                src.ringY[rIdx] = kin.y;
                src.ringBx[rIdx] = kin.bx;
                src.ringBy[rIdx] = kin.by;
                src.ringBdx[rIdx] = kin.bdx;
                src.ringBdy[rIdx] = kin.bdy;
            }

            populateShellsFromRingBuffer(src, M, vMode, DT_STEP, uniformMaxR, gridMode);
        }

        function initSources() {
            state.sources = [];
            for (let i = 0; i < MAX_PARTICLES; i++) {
                state.sources.push(createCompactifiedSource(i, state.numParticles, state.time));
            }
            reseedAllSources();
        }

        function reseedAllSources() {
            for (let i = 0; i < state.numParticles; i++) {
                seedExactRetardedHistory(state.sources[i], i, state.numParticles, state.time);
            }
        }

        // Converged Float64 Analytical Reference (28-step bracketed bisection + 10-step Newton-Raphson)
        function evaluateExactAnalyticalLW(px, py, t, betaOverride, motionOverride) {
            let totEx = 0, totEy = 0;
            let totVelMag = 0, totRadMag = 0;
            let primaryR = 0, primaryPosX = 0, primaryPosY = 0;

            const N = state.numParticles;
            for (let i = 0; i < N; i++) {
                let tLow = t - 260.0 / C_LIGHT;
                let tHigh = t;
                for (let b = 0; b < 32; b++) {
                    const tMid = 0.5 * (tLow + tHigh);
                    const kin = getExactTrajectoryKinematics(i, N, tMid, betaOverride, motionOverride);
                    const fMid = C_LIGHT * (t - tMid) - Math.hypot(px - kin.x, py - kin.y);
                    if (fMid > 0) tLow = tMid; else tHigh = tMid;
                }
                let tRet = 0.5 * (tLow + tHigh);
                for (let iter = 0; iter < 12; iter++) {
                    const kin = getExactTrajectoryKinematics(i, N, tRet, betaOverride, motionOverride);
                    const dx = px - kin.x;
                    const dy = py - kin.y;
                    const dist = Math.hypot(dx, dy);
                    const f = C_LIGHT * (t - tRet) - dist;
                    if (Math.abs(f) < 1e-13) break;
                    const nx = dist > 1e-12 ? dx / dist : 1.0;
                    const ny = dist > 1e-12 ? dy / dist : 0.0;
                    const df = -C_LIGHT * Math.max(1e-11, 1.0 - (nx * kin.bx + ny * kin.by));
                    const cand = tRet - f / df;
                    tRet = (cand >= tLow && cand <= tHigh) ? cand : 0.5 * (tLow + tHigh);
                    if (f > 0) tLow = tRet; else tHigh = tRet;
                }

                const retKin = getExactTrajectoryKinematics(i, N, tRet, betaOverride, motionOverride);
                const dx = px - retKin.x, dy = py - retKin.y;
                const r2 = dx * dx + dy * dy;
                const r = Math.sqrt(r2);
                const rSoft2 = r2 + state.softEps * state.softEps;
                const rSoft = Math.sqrt(rSoft2);
                const nx = r > 1e-12 ? dx / r : 1.0;
                const ny = r > 1e-12 ? dy / r : 0.0;

                const bMag = Math.hypot(retKin.bx, retKin.by);
                const oneMinusBeta2 = (1.0 - bMag) * (1.0 + bMag);
                const kappa = Math.max(1e-11, 1.0 - (nx * retKin.bx + ny * retKin.by));
                const kappa3 = kappa * kappa * kappa;

                const velPref = (retKin.charge * oneMinusBeta2) / (kappa3 * rSoft2);
                const eVelX = velPref * (nx - retKin.bx);
                const eVelY = velPref * (ny - retKin.by);

                const ux = nx - retKin.bx, uy = ny - retKin.by;
                const crossZ = ux * retKin.bdy - uy * retKin.bdx;
                const radPref = retKin.charge / (kappa3 * rSoft);
                const eRadX = radPref * (ny * crossZ);
                const eRadY = radPref * (-nx * crossZ);

                totEx += eVelX + eRadX;
                totEy += eVelY + eRadY;
                totVelMag += Math.hypot(eVelX, eVelY);
                totRadMag += Math.hypot(eRadX, eRadY);
                if (i === 0) {
                    primaryR = r;
                    primaryPosX = retKin.x;
                    primaryPosY = retKin.y;
                }
            }

            return {
                ex: totEx, ey: totEy,
                mag: Math.hypot(totEx, totEy),
                velMag: totVelMag, radMag: totRadMag,
                rRet: primaryR, emitX: primaryPosX, emitY: primaryPosY
            };
        }

        function getKnotBetaFloat32(src, mIdx, velMode) {
            const ux = src.emitUx[mIdx];
            const uy = src.emitUy[mIdx];
            if (velMode === 2) {
                const uSq = f32(f32(ux * ux) + f32(uy * uy));
                const invGamma = f32(1.0 / f32(Math.sqrt(f32(1.0 + uSq))));
                return { bx: f32(ux * invGamma), by: f32(uy * invGamma) };
            }
            return { bx: ux, by: uy };
        }

        function computeBernsteinAlphaFloat32(vSecX, vSecY, v1RawX, v1RawY, maxSub) {
            const v1Sq = f32(f32(v1RawX * v1RawX) + f32(v1RawY * v1RawY));
            const maxSq = f32(maxSub * maxSub);
            if (v1Sq <= maxSq) return f32(1.0);

            const dx = f32(v1RawX - vSecX);
            const dy = f32(v1RawY - vSecY);
            const a = f32(f32(dx * dx) + f32(dy * dy));
            if (a <= f32(1e-12)) return f32(0.0);

            const bHalf = f32(f32(vSecX * dx) + f32(vSecY * dy));
            const c = f32(f32(f32(vSecX * vSecX) + f32(vSecY * vSecY)) - maxSq);
            if (c >= f32(0.0)) return f32(0.0);

            const disc = f32(Math.max(f32(0.0), f32(f32(bHalf * bHalf) - f32(a * c))));
            const alpha = f32(f32(-bHalf + f32(Math.sqrt(disc))) / a);
            return f32(Math.max(f32(0.0), Math.min(f32(1.0), alpha)));
        }

        // Strict IEEE-754 Float32 Kinematical Hermite Interpolator (uses f32Len instead of Float64-backed Math.hypot!)
        function sampleSourceHistoryFloat32(src, mFloat, M, velMode) {
            const mClamped = f32(Math.max(0.0, Math.min(f32(M - 1.0001), f32(mFloat))));
            const m1 = Math.floor(mClamped);
            const m0 = Math.max(0, m1 - 1);
            const m2 = Math.min(M - 1, m1 + 1);
            const m3 = Math.min(M - 1, m1 + 2);

            const t = f32(mClamped - m1);
            const t2 = f32(t * t);
            const t3 = f32(t2 * t);

            const r1 = src.emitR[m1];
            const r2 = src.emitR[m2];
            const dR = m2 > m1 ? f32(Math.max(f32(1e-6), f32(r2 - r1))) : f32(1e-6);
            const s = m2 > m1 ? t : f32(0.0);
            const dRSub = f32(s * dR);

            const x1 = src.emitX[m1], y1 = src.emitY[m1];
            const x2 = src.emitX[m2], y2 = src.emitY[m2];
            const dxChord = f32(x2 - x1), dyChord = f32(y2 - y1);
            const invDR = f32(1.0 / dR);
            const maxSubluminal = f32(0.99995);

            let vSecX = f32(dxChord * invDR);
            let vSecY = f32(dyChord * invDR);
            const vSecMag = f32Len(vSecX, vSecY);
            if (vSecMag > maxSubluminal) {
                const sc = f32(maxSubluminal / vSecMag);
                vSecX = f32(vSecX * sc);
                vSecY = f32(vSecY * sc);
            }

            const kb1 = getKnotBetaFloat32(src, m1, velMode);
            const kb2 = getKnotBetaFloat32(src, m2, velMode);
            let b1x = kb1.bx, b1y = kb1.by;
            let b2x = kb2.bx, b2y = kb2.by;
            const b1Mag = f32Len(b1x, b1y);
            if (b1Mag > maxSubluminal) { b1x = f32(b1x * f32(maxSubluminal / b1Mag)); b1y = f32(b1y * f32(maxSubluminal / b1Mag)); }
            const b2Mag = f32Len(b2x, b2y);
            if (b2Mag > maxSubluminal) { b2x = f32(b2x * f32(maxSubluminal / b2Mag)); b2y = f32(b2y * f32(maxSubluminal / b2Mag)); }

            const v1RawX = f32(f32(3.0 * vSecX) + f32(b1x + b2x));
            const v1RawY = f32(f32(3.0 * vSecY) + f32(b1y + b2y));

            const alpha = computeBernsteinAlphaFloat32(vSecX, vSecY, v1RawX, v1RawY, maxSubluminal);
            const oma = f32(1.0 - alpha);

            const v0x = f32(f32(oma * vSecX) - f32(alpha * b1x));
            const v0y = f32(f32(oma * vSecY) - f32(alpha * b1y));
            const v1x = f32(vSecX + f32(alpha * f32(v1RawX - vSecX)));
            const v1y = f32(vSecY + f32(alpha * f32(v1RawY - vSecY)));
            const v2x = f32(f32(oma * vSecX) - f32(alpha * b2x));
            const v2y = f32(f32(oma * vSecY) - f32(alpha * b2y));

            const tx1 = f32(v0x * dR);
            const ty1 = f32(v0y * dR);
            const tx2 = f32(v2x * dR);
            const ty2 = f32(v2y * dR);

            const s2 = f32(s * s);
            const s3 = f32(s2 * s);
            const h10 = f32(s3 - f32(2.0 * s2) + s);
            const h01 = f32(f32(-2.0 * s3) + f32(3.0 * s2));
            const h11 = f32(s3 - s2);

            const dxSub = f32(f32(h01 * dxChord) + f32(h10 * tx1) + f32(h11 * tx2));
            const dySub = f32(f32(h01 * dyChord) + f32(h10 * ty1) + f32(h11 * ty2));
            const x = f32(x1 + dxSub);
            const y = f32(y1 + dySub);

            const oms = f32(1.0 - s);
            const b0 = f32(oms * oms);
            const b1 = f32(2.0 * f32(s * oms));
            const b2 = s2;
            const dzdRx = f32(f32(b0 * v0x) + f32(b1 * v1x) + f32(b2 * v2x));
            const dzdRy = f32(f32(b0 * v0y) + f32(b1 * v1y) + f32(b2 * v2y));
            const splineVelX = f32(-dzdRx);
            const splineVelY = f32(-dzdRy);
            const splineVelMag = f32Len(splineVelX, splineVelY);

            const w0 = f32(f32(-0.5 * t3) + f32(t2 - f32(0.5 * t)));
            const w1 = f32(f32(1.5 * t3) - f32(2.5 * t2) + 1.0);
            const w2 = f32(f32(-1.5 * t3) + f32(2.0 * t2) + f32(0.5 * t));
            const w3 = f32(f32(0.5 * t3) - f32(0.5 * t2));

            let bx, by, bHatX, bHatY, bMag, oneMinusBeta2;
            if (velMode === 2) {
                const uMag1 = f32Len(src.emitUx[m1], src.emitUy[m1]);
                const uMag2 = f32Len(src.emitUx[m2], src.emitUy[m2]);
                const uTarget = f32(uMag1 + f32(t * f32(uMag2 - uMag1)));
                bHatX = splineVelMag > 1e-7 ? f32(splineVelX / splineVelMag) : f32(1.0);
                bHatY = splineVelMag > 1e-7 ? f32(splineVelY / splineVelMag) : f32(0.0);
                const uSq = f32(uTarget * uTarget);
                const gamma = f32(Math.sqrt(f32(1.0 + uSq)));
                bMag = f32(uTarget / gamma);
                bx = f32(bHatX * bMag);
                by = f32(bHatY * bMag);
                oneMinusBeta2 = f32(1.0 / f32(1.0 + uSq));
            } else if (velMode === 1) {
                const bMag1 = f32Len(src.emitUx[m1], src.emitUy[m1]);
                const bMag2 = f32Len(src.emitUx[m2], src.emitUy[m2]);
                const bTarget = f32(bMag1 + f32(t * f32(bMag2 - bMag1)));
                bHatX = splineVelMag > 1e-7 ? f32(splineVelX / splineVelMag) : f32(1.0);
                bHatY = splineVelMag > 1e-7 ? f32(splineVelY / splineVelMag) : f32(0.0);
                bMag = f32(Math.min(0.9999995, bTarget));
                bx = f32(bHatX * bMag);
                by = f32(bHatY * bMag);
                oneMinusBeta2 = f32(f32(1.0 - bMag) * f32(1.0 + bMag));
            } else {
                const rawUx = f32(f32(src.emitUx[m0] * w0) + f32(src.emitUx[m1] * w1) + f32(src.emitUx[m2] * w2) + f32(src.emitUx[m3] * w3));
                const rawUy = f32(f32(src.emitUy[m0] * w0) + f32(src.emitUy[m1] * w1) + f32(src.emitUy[m2] * w2) + f32(src.emitUy[m3] * w3));
                bx = rawUx;
                by = rawUy;
                bMag = f32(Math.min(0.9999995, f32Len(bx, by)));
                bHatX = bMag > 1e-7 ? f32(bx / bMag) : f32(1.0);
                bHatY = bMag > 1e-7 ? f32(by / bMag) : f32(0.0);
                oneMinusBeta2 = f32(f32(1.0 - bMag) * f32(1.0 + bMag));
            }

            const bdx = f32(f32(src.emitBdx[m0] * w0) + f32(src.emitBdx[m1] * w1) + f32(src.emitBdx[m2] * w2) + f32(src.emitBdx[m3] * w3));
            const bdy = f32(f32(src.emitBdy[m0] * w0) + f32(src.emitBdy[m1] * w1) + f32(src.emitBdy[m2] * w2) + f32(src.emitBdy[m3] * w3));

            return { x, y, x1, y1, dxSub, dySub, r1, dRSub, dRdm: dR, dzdRx, dzdRy, bx, by, bHatX, bHatY, bMag, oneMinusBeta2, bdx, bdy };
        }

        // Pure Float64 Compactified Evaluator (Separates Spatial Spline Convergence from Float32 Arithmetic Floor)
        function evaluateCompactifiedFloat64(px, py, M, velMode) {
            const src = state.sources[0];
            let mLow = 0.0, mHigh = M - 1.0001;

            function sample64(mFloat) {
                const mClamped = Math.max(0.0, Math.min(M - 1.0001, mFloat));
                const m1 = Math.floor(mClamped);
                const m0 = Math.max(0, m1 - 1);
                const m2 = Math.min(M - 1, m1 + 1);
                const m3 = Math.min(M - 1, m1 + 2);
                const s = m2 > m1 ? (mClamped - m1) : 0.0;
                const s2 = s * s, s3 = s2 * s;
                const r1 = src.emitR64[m1], r2 = src.emitR64[m2];
                const dR = m2 > m1 ? Math.max(1e-12, r2 - r1) : 1e-12;
                const dRSub = s * dR;

                const uSq1 = src.emitUx64[m1] * src.emitUx64[m1] + src.emitUy64[m1] * src.emitUy64[m1];
                const uSq2 = src.emitUx64[m2] * src.emitUx64[m2] + src.emitUy64[m2] * src.emitUy64[m2];
                const b1x = velMode === 2 ? src.emitUx64[m1] / Math.sqrt(1.0 + uSq1) : src.emitUx64[m1];
                const b1y = velMode === 2 ? src.emitUy64[m1] / Math.sqrt(1.0 + uSq1) : src.emitUy64[m1];
                const b2x = velMode === 2 ? src.emitUx64[m2] / Math.sqrt(1.0 + uSq2) : src.emitUx64[m2];
                const b2y = velMode === 2 ? src.emitUy64[m2] / Math.sqrt(1.0 + uSq2) : src.emitUy64[m2];

                const dxChord = src.emitX64[m2] - src.emitX64[m1];
                const dyChord = src.emitY64[m2] - src.emitY64[m1];
                const vSecX = dxChord / dR, vSecY = dyChord / dR;
                const v0x = -b1x, v0y = -b1y;
                const v1x = 3.0 * vSecX + b1x + b2x, v1y = 3.0 * vSecY + b1y + b2y;
                const v2x = -b2x, v2y = -b2y;

                const h10 = s3 - 2.0 * s2 + s;
                const h01 = -2.0 * s3 + 3.0 * s2;
                const h11 = s3 - s2;
                const dxSub = h01 * dxChord + h10 * (v0x * dR) + h11 * (v2x * dR);
                const dySub = h01 * dyChord + h10 * (v0y * dR) + h11 * (v2y * dR);

                const oms = 1.0 - s;
                const dzdRx = oms * oms * v0x + 2.0 * s * oms * v1x + s2 * v2x;
                const dzdRy = oms * oms * v0y + 2.0 * s * oms * v1y + s2 * v2y;
                const spMag = Math.hypot(dzdRx, dzdRy);
                const bHatX = spMag > 1e-12 ? -dzdRx / spMag : 1.0;
                const bHatY = spMag > 1e-12 ? -dzdRy / spMag : 0.0;

                const uMag1 = Math.hypot(src.emitUx64[m1], src.emitUy64[m1]);
                const uMag2 = Math.hypot(src.emitUx64[m2], src.emitUy64[m2]);
                const uTarget = uMag1 + s * (uMag2 - uMag1);
                const uSq = uTarget * uTarget;
                const bMag = velMode === 2 ? uTarget / Math.sqrt(1.0 + uSq) : uTarget;
                const oneMinusBeta2 = velMode === 2 ? 1.0 / (1.0 + uSq) : (1.0 - bMag) * (1.0 + bMag);

                const w0 = -0.5 * s3 + s2 - 0.5 * s;
                const w1 =  1.5 * s3 - 2.5 * s2 + 1.0;
                const w2 = -1.5 * s3 + 2.0 * s2 + 0.5 * s;
                const w3 =  0.5 * s3 - 0.5 * s2;
                const bdx = src.emitBdx64[m0] * w0 + src.emitBdx64[m1] * w1 + src.emitBdx64[m2] * w2 + src.emitBdx64[m3] * w3;
                const bdy = src.emitBdy64[m0] * w0 + src.emitBdy64[m1] * w1 + src.emitBdy64[m2] * w2 + src.emitBdy64[m3] * w3;

                return { x1: src.emitX64[m1], y1: src.emitY64[m1], dxSub, dySub, r1, dRSub, dRdm: dR, dzdRx, dzdRy, bHatX, bHatY, bMag, oneMinusBeta2, bdx, bdy };
            }

            for (let iter = 0; iter < 24; iter++) {
                const mMid = 0.5 * (mLow + mHigh);
                const h = sample64(mMid);
                const dist = Math.hypot((px - h.x1) - h.dxSub, (py - h.y1) - h.dySub);
                if (h.r1 + h.dRSub < dist) mLow = mMid; else mHigh = mMid;
            }
            let mRet = 0.5 * (mLow + mHigh);
            for (let nIter = 0; nIter < 6; nIter++) {
                const h = sample64(mRet);
                const dx = (px - h.x1) - h.dxSub, dy = (py - h.y1) - h.dySub;
                const dist = Math.hypot(dx, dy);
                const nx = dx / dist, ny = dy / dist;
                const fVal = (h.r1 - dist) + h.dRSub;
                const dfdm = h.dRdm * Math.max(1e-11, 1.0 + nx * h.dzdRx + ny * h.dzdRy);
                mRet = Math.max(mLow, Math.min(mHigh, mRet - fVal / dfdm));
            }
            const ret = sample64(mRet);
            const dx = (px - ret.x1) - ret.dxSub, dy = (py - ret.y1) - ret.dySub;
            const r2 = dx * dx + dy * dy, r = Math.sqrt(r2);
            const rSoft2 = r2 + state.softEps * state.softEps, rSoft = Math.sqrt(rSoft2);
            const nx = dx / r, ny = dy / r;
            const oneMinusBeta = ret.oneMinusBeta2 / (1.0 + ret.bMag);
            const diffX = nx - ret.bHatX, diffY = ny - ret.bHatY;
            const kappa = oneMinusBeta + 0.5 * ret.bMag * (diffX * diffX + diffY * diffY);
            const kappa3 = kappa * kappa * kappa;
            const ux = oneMinusBeta * nx + ret.bMag * diffX;
            const uy = oneMinusBeta * ny + ret.bMag * diffY;
            const velPref = ret.oneMinusBeta2 / (kappa3 * rSoft2);
            const crossZ = ux * ret.bdy - uy * ret.bdx;
            const radPref = 1.0 / (kappa3 * rSoft);
            const ex = velPref * ux + radPref * (ny * crossZ);
            const ey = velPref * uy + radPref * (-nx * crossZ);
            return { ex, ey, mag: Math.hypot(ex, ey) };
        }

        function evalCancellationFreeKappaAndUFloat32(nx, ny, hist) {
            const oneMinusBeta = f32(hist.oneMinusBeta2 / f32(1.0 + hist.bMag));
            const diffX = f32(nx - hist.bHatX);
            const diffY = f32(ny - hist.bHatY);
            const diffSq = f32(f32(diffX * diffX) + f32(diffY * diffY));
            const kappa = f32(Math.max(f32(1e-7), f32(oneMinusBeta + f32(0.5 * f32(hist.bMag * diffSq)))));
            const ux = f32(f32(oneMinusBeta * nx) + f32(hist.bMag * diffX));
            const uy = f32(f32(oneMinusBeta * ny) + f32(hist.bMag * diffY));
            return { kappa, ux, uy };
        }

        // Strict IEEE-754 Float32 Shader-Equivalent Evaluator (Uses f32Len to match GLSL length() bit-for-bit!)
        function evaluateShaderEquivalentFloat32(pxIn, pyIn, mTOverride, kBinOverride, velModeOverride) {
            const px = f32(pxIn), py = f32(pyIn);
            const M = mTOverride || state.mT;
            const K_BIN = kBinOverride || state.kBin;
            const vMode = velModeOverride !== undefined ? velModeOverride : state.velMode;
            const softEps = f32(state.softEps);
            const dTheta = f32(Math.PI / M);

            let totEx = f32(0.0), totEy = f32(0.0);
            let totVelMag = f32(0.0), totRadMag = f32(0.0);
            let primaryR = f32(0.0), primaryTheta = f32(0.0);
            let primaryPosX = f32(0.0), primaryPosY = f32(0.0);

            const N = state.numParticles;
            for (let s = 0; s < N; s++) {
                const src = state.sources[s];
                let mLow = f32(0.0);
                let mHigh = f32(M - 1.001);

                for (let iter = 0; iter < K_BIN; iter++) {
                    const mMid = f32(0.5 * f32(mLow + mHigh));
                    const hist = sampleSourceHistoryFloat32(src, mMid, M, vMode);
                    const dx = f32(f32(px - hist.x1) - hist.dxSub);
                    const dy = f32(f32(py - hist.y1) - hist.dySub);
                    const dist = f32Len(dx, dy);
                    const rShell = f32(hist.r1 + hist.dRSub);
                    if (rShell < dist) mLow = mMid; else mHigh = mMid;
                }

                let mRet = f32(0.5 * f32(mLow + mHigh));
                for (let nIter = 0; nIter < 2; nIter++) {
                    const hist = sampleSourceHistoryFloat32(src, mRet, M, vMode);
                    const dx = f32(f32(px - hist.x1) - hist.dxSub);
                    const dy = f32(f32(py - hist.y1) - hist.dySub);
                    const dist = f32Len(dx, dy);
                    const nx = dist > 1e-6 ? f32(dx / dist) : f32(1.0);
                    const ny = dist > 1e-6 ? f32(dy / dist) : f32(0.0);
                    const fVal = f32(f32(hist.r1 - dist) + hist.dRSub);
                    const nDotDzdR = f32(f32(nx * hist.dzdRx) + f32(ny * hist.dzdRy));
                    const kappaSpline = f32(Math.max(f32(5e-5), f32(1.0 + nDotDzdR)));
                    const dfdm = f32(Math.max(f32(1e-7), f32(hist.dRdm * kappaSpline)));
                    const cand = f32(mRet - f32(fVal / dfdm));
                    mRet = f32(Math.max(mLow, Math.min(mHigh, cand)));
                }

                const ret = sampleSourceHistoryFloat32(src, mRet, M, vMode);
                const dx = f32(f32(px - ret.x1) - ret.dxSub);
                const dy = f32(f32(py - ret.y1) - ret.dySub);
                const r2 = f32(f32(dx * dx) + f32(dy * dy));
                const r = f32(Math.sqrt(r2));
                const rSoft2 = f32(r2 + f32(softEps * softEps));
                const rSoft = f32(Math.sqrt(rSoft2));
                const nx = r > 1e-6 ? f32(dx / r) : f32(1.0);
                const ny = r > 1e-6 ? f32(dy / r) : f32(0.0);

                const ku = evalCancellationFreeKappaAndUFloat32(nx, ny, ret);
                const kappa = ku.kappa;
                const kappa3 = f32(kappa * f32(kappa * kappa));

                const chg = f32(src.charge);
                const velPref = f32(f32(chg * ret.oneMinusBeta2) / f32(kappa3 * rSoft2));
                const eVelX = f32(velPref * ku.ux);
                const eVelY = f32(velPref * ku.uy);

                const crossZ = f32(f32(ku.ux * ret.bdy) - f32(ku.uy * ret.bdx));
                const radPref = f32(chg / f32(kappa3 * rSoft));
                const eRadX = f32(radPref * f32(ny * crossZ));
                const eRadY = f32(radPref * f32(-nx * crossZ));

                totEx = f32(totEx + f32(eVelX + eRadX));
                totEy = f32(totEy + f32(eVelY + eRadY));
                totVelMag = f32(totVelMag + f32Len(eVelX, eVelY));
                totRadMag = f32(totRadMag + f32Len(eRadX, eRadY));

                if (s === 0) {
                    primaryR = r;
                    primaryTheta = f32(f32(mRet + 0.5) * dTheta);
                    primaryPosX = ret.x;
                    primaryPosY = ret.y;
                }
            }

            return {
                ex: totEx, ey: totEy,
                mag: f32Len(totEx, totEy),
                velMag: totVelMag, radMag: totRadMag,
                rRet: primaryR, thetaRet: primaryTheta,
                emitX: primaryPosX, emitY: primaryPosY
            };
        }

        function stepSimulation(dt, betaOverride, motionOverride, mTOverride, velModeOverride, uniformMaxR, gridMode, skipShellPopulate) {
            state.time += dt;
            const N = state.numParticles;
            const M = mTOverride || state.mT;
            const vMode = velModeOverride !== undefined ? velModeOverride : state.velMode;
            const dTheta = Math.PI / M;

            for (let i = 0; i < N; i++) {
                const s = state.sources[i];

                // Allow smooth, C^2 spring-damper manual dragging of Source #0 AND subluminal return to orbit upon release!
                const kin = getExactTrajectoryKinematics(i, N, state.time, betaOverride, motionOverride);
                s.charge = kin.charge;

                if (i === 0 && betaOverride === undefined && (state.interactionTarget === 'source0' || s.returningToOrbit)) {
                    const goalX = state.interactionTarget === 'source0' ? state.draggedSourceTarget.x : kin.x;
                    const goalY = state.interactionTarget === 'source0' ? state.draggedSourceTarget.y : kin.y;
                    const goalBx = state.interactionTarget === 'source0' ? 0.0 : kin.bx;
                    const goalBy = state.interactionTarget === 'source0' ? 0.0 : kin.by;

                    const dx = goalX - s.x;
                    const dy = goalY - s.y;
                    const distToGoal = Math.hypot(dx, dy);
                    const dvGoal = Math.hypot(s.bx - goalBx, s.by - goalBy);

                    if (state.interactionTarget !== 'source0' && distToGoal < 0.008 && dvGoal < 0.03) {
                        s.returningToOrbit = false;
                        s.x = kin.x; s.y = kin.y;
                        s.bx = kin.bx; s.by = kin.by;
                        s.bdx = kin.bdx; s.bdy = kin.bdy;
                    } else {
                        const springK = 8.5;
                        let targetBx = goalBx + (dx * springK) / C_LIGHT;
                        let targetBy = goalBy + (dy * springK) / C_LIGHT;
                        const targetBMag = Math.hypot(targetBx, targetBy);
                        const maxDragBeta = 0.90;
                        if (targetBMag > maxDragBeta) {
                            targetBx *= (maxDragBeta / targetBMag);
                            targetBy *= (maxDragBeta / targetBMag);
                        }
                        const blend = 0.085;
                        const newBx = (1.0 - blend) * s.bx + blend * targetBx;
                        const newBy = (1.0 - blend) * s.by + blend * targetBy;
                        let rawBdx = ((newBx - s.bx) / Math.max(1e-5, dt)) / C_LIGHT;
                        let rawBdy = ((newBy - s.by) / Math.max(1e-5, dt)) / C_LIGHT;
                        const accMag = Math.hypot(rawBdx, rawBdy);
                        const maxDragAcc = 1.15;
                        if (accMag > maxDragAcc) {
                            rawBdx *= (maxDragAcc / accMag);
                            rawBdy *= (maxDragAcc / accMag);
                        }
                        s.bdx = 0.88 * s.bdx + 0.12 * rawBdx;
                        s.bdy = 0.88 * s.bdy + 0.12 * rawBdy;
                        s.bx = newBx;
                        s.by = newBy;
                        s.x += s.bx * C_LIGHT * dt;
                        s.y += s.by * C_LIGHT * dt;
                    }
                } else {
                    s.x = kin.x; s.y = kin.y;
                    s.bx = kin.bx; s.by = kin.by;
                    s.bdx = kin.bdx; s.bdy = kin.bdy;
                }

                s.ringHead = (s.ringHead + 1) % RING_SIZE;
                s.ringX[s.ringHead] = s.x;
                s.ringY[s.ringHead] = s.y;
                s.ringBx[s.ringHead] = s.bx;
                s.ringBy[s.ringHead] = s.by;
                s.ringBdx[s.ringHead] = s.bdx;
                s.ringBdy[s.ringHead] = s.bdy;

                if (skipShellPopulate) continue;

                if (!state.useDiffusiveEulerian) {
                    populateShellsFromRingBuffer(s, M, vMode, dt, uniformMaxR, gridMode);
                } else {
                    const cDt = C_LIGHT * dt;
                    for (let m = 0; m < M; m++) {
                        const th = (m + 0.5) * dTheta;
                        const rNow = state.scaleL * Math.tan(0.5 * th);
                        const rPrev = rNow - cDt;
                        if (rPrev <= 0) {
                            const packedV = packVelocityForStorage(s.bx, s.by, vMode);
                            s.nextX[m] = f32(s.x); s.nextY[m] = f32(s.y);
                            s.nextUx[m] = packedV.ux; s.nextUy[m] = packedV.uy;
                            s.nextBdx[m] = f32(s.bdx); s.nextBdy[m] = f32(s.bdy);
                        } else {
                            const thetaPrev = 2.0 * Math.atan(rPrev / state.scaleL);
                            const mPrev = (thetaPrev / Math.PI) * M - 0.5;
                            const mClamped = Math.max(0, Math.min(M - 1.001, mPrev));
                            const m0 = Math.floor(mClamped);
                            const m1 = Math.min(M - 1, m0 + 1);
                            const f = mClamped - m0, omf = 1.0 - f;
                            s.nextX[m] = f32(s.emitX[m0] * omf + s.emitX[m1] * f);
                            s.nextY[m] = f32(s.emitY[m0] * omf + s.emitY[m1] * f);
                            s.nextUx[m] = f32(s.emitUx[m0] * omf + s.emitUx[m1] * f);
                            s.nextUy[m] = f32(s.emitUy[m0] * omf + s.emitUy[m1] * f);
                            s.nextBdx[m] = f32(s.emitBdx[m0] * omf + s.emitBdx[m1] * f);
                            s.nextBdy[m] = f32(s.emitBdy[m0] * omf + s.emitBdy[m1] * f);
                        }
                    }
                    s.emitX.set(s.nextX); s.emitY.set(s.nextY);
                    s.emitUx.set(s.nextUx); s.emitUy.set(s.nextUy);
                    s.emitBdx.set(s.nextBdx); s.emitBdy.set(s.nextBdy);
                }
            }
        }

module.exports = { state, f32, seedExactRetardedHistory, stepSimulation, evaluateExactAnalyticalLW, evaluateShaderEquivalentFloat32, evaluateCompactifiedFloat64, getPulseCrossingSweepPoints, createCompactifiedSource, sampleSourceHistoryFloat32, computeBernsteinAlphaFloat32, DT_STEP, C_LIGHT, getKnotBetaFloat32 };
