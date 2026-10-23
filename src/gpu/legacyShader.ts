// Legacy shaders, extracted verbatim from the original solver (reference/original_solver.html).
// Kept for regression testing; the production shader lives in shader.ts.

export const LEGACY_VERTEX_SHADER = `#version 300 es
            in vec2 a_pos;
            out vec2 v_uv;
            void main() {
                v_uv = a_pos * 0.5 + 0.5;
                gl_Position = vec4(a_pos, 0.0, 1.0);
            }`;

export const LEGACY_FRAGMENT_SHADER = `#version 300 es
            precision highp float;
            in vec2 v_uv;
            out vec4 outColor;

            uniform sampler2D u_texPosVel;
            uniform sampler2D u_texAccChg;
            uniform sampler2D u_texQueryCoords;
            uniform int u_activeCount;
            uniform int u_mT;
            uniform int u_kBin;
            uniform int u_velMode;
            uniform int u_readbackMode;
            uniform float u_aspect;
            uniform float u_scaleL;
            uniform float u_softEps;
            uniform int u_channel;
            uniform int u_motionMode;
            uniform int u_stressMode;
            uniform int u_showWavePool;
            uniform int u_showImpostors;
            uniform float u_beta;
            uniform float u_omega;
            uniform float u_time;

            const float PI = 3.14159265359;
            const float C_LIGHT = 2.35;

            vec2 unpackKnotBeta(vec2 rawVel) {
                if (u_velMode == 2) {
                    float uSq = dot(rawVel, rawVel);
                    return rawVel / sqrt(1.0 + uSq);
                }
                return rawVel;
            }

            float computeBernsteinAlpha(vec2 vSec, vec2 v1Raw, float maxSub) {
                float v1Sq = dot(v1Raw, v1Raw);
                float maxSq = maxSub * maxSub;
                if (v1Sq <= maxSq) return 1.0;
                vec2 d = v1Raw - vSec;
                float a = dot(d, d);
                if (a <= 1e-12) return 0.0;
                float bHalf = dot(vSec, d);
                float c = dot(vSec, vSec) - maxSq;
                if (c >= 0.0) return 0.0;
                float disc = max(0.0, bHalf * bHalf - a * c);
                return clamp((-bHalf + sqrt(disc)) / a, 0.0, 1.0);
            }

            void sampleCatmullRom(int s, float mFloat, out vec2 pos1, out vec2 dPosSub, out float r1, out float dRSub, out float dRdm, out vec2 dzdR, out vec2 bHat, out float bMag, out float oneMinusBeta2, out vec2 acc, out float chg) {
                float mClamped = clamp(mFloat, 0.0, float(u_mT) - 1.0001);
                float mFloor = floor(mClamped);
                int m1 = int(mFloor);
                int m0 = max(0, m1 - 1);
                int m2 = min(u_mT - 1, m1 + 1);
                int m3 = min(u_mT - 1, m1 + 2);
                float t = mClamped - mFloor;
                float t2 = t * t;
                float t3 = t2 * t;

                vec4 pv0 = texelFetch(u_texPosVel, ivec2(m0, s), 0);
                vec4 pv1 = texelFetch(u_texPosVel, ivec2(m1, s), 0);
                vec4 pv2 = texelFetch(u_texPosVel, ivec2(m2, s), 0);
                vec4 pv3 = texelFetch(u_texPosVel, ivec2(m3, s), 0);

                vec4 ac0 = texelFetch(u_texAccChg, ivec2(m0, s), 0);
                vec4 ac1 = texelFetch(u_texAccChg, ivec2(m1, s), 0);
                vec4 ac2 = texelFetch(u_texAccChg, ivec2(m2, s), 0);
                vec4 ac3 = texelFetch(u_texAccChg, ivec2(m3, s), 0);

                r1 = ac1.w;
                float r2 = ac2.w;
                float dR = m2 > m1 ? max(1e-6, r2 - r1) : 1e-6;
                float sRad = m2 > m1 ? t : 0.0;
                dRSub = sRad * dR;
                dRdm = dR;

                float maxSubluminal = 0.99995;
                vec2 dChord = pv2.xy - pv1.xy;
                vec2 vSec = dChord / dR;
                float vSecMag = length(vSec);
                if (vSecMag > maxSubluminal) vSec *= (maxSubluminal / vSecMag);

                vec2 b1 = unpackKnotBeta(pv1.zw);
                vec2 b2 = unpackKnotBeta(pv2.zw);
                float b1Mag = length(b1);
                if (b1Mag > maxSubluminal) b1 *= (maxSubluminal / b1Mag);
                float b2Mag = length(b2);
                if (b2Mag > maxSubluminal) b2 *= (maxSubluminal / b2Mag);

                vec2 v1Raw = 3.0 * vSec + b1 + b2;
                float alpha = computeBernsteinAlpha(vSec, v1Raw, maxSubluminal);
                float oma = 1.0 - alpha;

                vec2 v0 = oma * vSec - alpha * b1;
                vec2 v1 = vSec + alpha * (v1Raw - vSec);
                vec2 v2 = oma * vSec - alpha * b2;
                vec2 tang1 = v0 * dR;
                vec2 tang2 = v2 * dR;

                float s2 = sRad * sRad;
                float s3 = s2 * sRad;
                float h10 =        s3 - 2.0 * s2 + sRad;
                float h01 = -2.0 * s3 + 3.0 * s2;
                float h11 =        s3 -       s2;

                pos1 = pv1.xy;
                dPosSub = h01 * dChord + h10 * tang1 + h11 * tang2;

                float oms = 1.0 - sRad;
                dzdR = (oms * oms) * v0 + (2.0 * sRad * oms) * v1 + s2 * v2;
                vec2 splineVel = -dzdR;
                float splineVelMag = length(splineVel);

                float w0 = -0.5 * t3 + t2 - 0.5 * t;
                float w1 =  1.5 * t3 - 2.5 * t2 + 1.0;
                float w2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
                float w3 =  0.5 * t3 - 0.5 * t2;

                vec4 ac = ac0 * w0 + ac1 * w1 + ac2 * w2 + ac3 * w3;
                acc = ac.xy;
                chg = ac.z;

                if (u_velMode == 2) {
                    float u1 = length(pv1.zw);
                    float u2 = length(pv2.zw);
                    float uTarget = mix(u1, u2, t);
                    bHat = splineVelMag > 1e-7 ? (splineVel / splineVelMag) : vec2(1.0, 0.0);
                    float uSq = uTarget * uTarget;
                    float gamma = sqrt(1.0 + uSq);
                    bMag = uTarget / gamma;
                    oneMinusBeta2 = 1.0 / (1.0 + uSq);
                } else if (u_velMode == 1) {
                    float b1m = length(pv1.zw);
                    float b2m = length(pv2.zw);
                    float bTarget = mix(b1m, b2m, t);
                    bHat = splineVelMag > 1e-7 ? (splineVel / splineVelMag) : vec2(1.0, 0.0);
                    bMag = min(0.9999995, bTarget);
                    oneMinusBeta2 = (1.0 - bMag) * (1.0 + bMag);
                } else {
                    vec2 rawVel = pv0.zw * w0 + pv1.zw * w1 + pv2.zw * w2 + pv3.zw * w3;
                    bMag = min(0.9999995, length(rawVel));
                    bHat = bMag > 1e-7 ? (rawVel / bMag) : vec2(1.0, 0.0);
                    oneMinusBeta2 = (1.0 - bMag) * (1.0 + bMag);
                }
            }

            void evalCancellationFreeKappaAndU(vec2 n, vec2 bHat, float bMag, float oneMinusBeta2, out float kappa, out vec2 uVec) {
                float oneMinusBeta = oneMinusBeta2 / (1.0 + bMag);
                vec2 diff = n - bHat;
                float diffSq = dot(diff, diff);
                kappa = max(1e-7, oneMinusBeta + 0.5 * bMag * diffSq);
                uVec = oneMinusBeta * n + bMag * diff;
            }

            void evalLWKernel(vec2 d, vec2 bHat, float bMag, float oneMinusBeta2, vec2 betaDot, float charge, out vec2 eVel, out vec2 eRad) {
                float r2 = dot(d, d);
                float r = sqrt(r2);
                float rSoft2 = r2 + u_softEps * u_softEps;
                float rSoft = sqrt(rSoft2);
                vec2 n = r > 1e-6 ? d / r : vec2(1.0, 0.0);

                float kappa;
                vec2 u;
                evalCancellationFreeKappaAndU(n, bHat, bMag, oneMinusBeta2, kappa, u);
                float kappa3 = kappa * kappa * kappa;

                eVel = (charge * oneMinusBeta2 / (kappa3 * rSoft2)) * u;
                float crossZ = u.x * betaDot.y - u.y * betaDot.x;
                eRad = (charge / (kappa3 * rSoft)) * vec2(n.y * crossZ, -n.x * crossZ);
            }

            vec3 evalConformalSphereImpostors(vec2 wPos) {
                vec3 impCol = vec3(0.0);
                float shellSpace = 0.52;
                float phaseOff = fract((u_time * C_LIGHT) / shellSpace);
                vec3 lightDir = normalize(vec3(-0.35, 0.45, 0.82));

                for (int k = 0; k < 5; k++) {
                    float rShell = (float(k) + phaseOff) * shellSpace;
                    if (rShell < 0.06 || rShell > 2.65) continue;
                    float th = 2.0 * atan(rShell / u_scaleL);
                    float mFl = (th / PI) * float(u_mT) - 0.5;
                    vec2 pos1, dPosSub, dzdR, bHat, acc;
                    float r1, dRSub, dRdm, bMag, omb2, chg;
                    sampleCatmullRom(0, mFl, pos1, dPosSub, r1, dRSub, dRdm, dzdR, bHat, bMag, omb2, acc, chg);

                    vec2 d = (wPos - pos1) - dPosSub;
                    float d2 = dot(d, d);
                    float r2 = rShell * rShell;
                    if (d2 < r2) {
                        float z = sqrt(r2 - d2);
                        vec3 normal = vec3(d.x, d.y, z) / rShell;
                        float nv = clamp(normal.z, 0.0, 1.0);
                        float fresnel = pow(1.0 - nv, 4.0);
                        float rim = exp(-abs(sqrt(d2) - rShell) * 48.0);
                        vec2 n2 = d / max(1e-5, sqrt(d2));
                        float kappa;
                        vec2 uTmp;
                        evalCancellationFreeKappaAndU(n2, bHat, bMag, omb2, kappa, uTmp);
                        float doppler = 1.0 / max(0.18, kappa);
                        float spec = pow(max(0.0, dot(reflect(-lightDir, normal), vec3(0.0, 0.0, 1.0))), 28.0);
                        float fade = clamp(1.0 - rShell / 2.65, 0.0, 1.0);
                        vec3 shellTint = mix(vec3(0.15, 0.68, 0.98), vec3(0.98, 0.75, 0.25), clamp((doppler - 1.0) * 0.35, 0.0, 1.0));
                        impCol += fade * (shellTint * (0.28 * fresnel + 0.65 * rim) + vec3(0.85, 0.95, 1.0) * spec * 0.45);
                    }
                }
                return impCol;
            }

            vec3 hsv2rgb(vec3 c) {
                vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
            }

            void main() {
                vec2 wPos;
                if (u_readbackMode == 1) {
                    int qIdx = int(gl_FragCoord.x);
                    wPos = texelFetch(u_texQueryCoords, ivec2(qIdx, 0), 0).xy;
                } else {
                    vec2 pNorm = vec2((v_uv.x - 0.5) * 2.0 * u_aspect, (v_uv.y - 0.5) * 2.0);
                    wPos = pNorm * 1.45;
                }

                vec2 totCompactE = vec2(0.0);
                vec2 totRadE = vec2(0.0);
                vec2 primaryEmitPos = vec2(0.0);

                // In visual multi-source mode (N_p > 4 and u_readbackMode == 0), use 7 binary + 1 Newton step for 60 FPS performance;
                // In readback mode or single-source benchmark mode, use exact u_kBin + 2 Newton steps!
                int effKBin = (u_readbackMode == 0 && u_activeCount > 4) ? min(u_kBin, 7) : u_kBin;
                int effNewton = (u_readbackMode == 0 && u_activeCount > 4) ? 1 : 2;

                for (int s = 0; s < 256; s++) {
                    if (s >= u_activeCount) break;

                    float mLow = 0.0;
                    float mHigh = float(u_mT) - 1.001;
                    for (int iter = 0; iter < 16; iter++) {
                        if (iter >= effKBin) break;
                        float mMid = 0.5 * (mLow + mHigh);
                        vec2 pos1, dPosSub, dzdR, bHat, acc;
                        float r1, dRSub, dRdm, bMag, omb2, chg;
                        sampleCatmullRom(s, mMid, pos1, dPosSub, r1, dRSub, dRdm, dzdR, bHat, bMag, omb2, acc, chg);
                        vec2 dVec = (wPos - pos1) - dPosSub;
                        if ((r1 + dRSub) < length(dVec)) mLow = mMid; else mHigh = mMid;
                    }

                    float mRet = 0.5 * (mLow + mHigh);
                    for (int nIter = 0; nIter < 2; nIter++) {
                        if (nIter >= effNewton) break;
                        vec2 pos1, dPosSub, dzdR, bHat, acc;
                        float r1, dRSub, dRdm, bMag, omb2, chg;
                        sampleCatmullRom(s, mRet, pos1, dPosSub, r1, dRSub, dRdm, dzdR, bHat, bMag, omb2, acc, chg);
                        vec2 dVec = (wPos - pos1) - dPosSub;
                        float dist = length(dVec);
                        vec2 n = dist > 1e-6 ? (dVec / dist) : vec2(1.0, 0.0);
                        float fVal = (r1 - dist) + dRSub;
                        float kappaSpline = max(5e-5, 1.0 + dot(n, dzdR));
                        float dfdm = max(1e-7, dRdm * kappaSpline);
                        float cand = mRet - (fVal / dfdm);
                        mRet = clamp(cand, mLow, mHigh);
                    }

                    vec2 pos1Ret, dPosSubRet, dzdRRet, bHatRet, accRet;
                    float r1Ret, dRSubRet, dRdmRet, bMagRet, omb2Ret, chgRet;
                    sampleCatmullRom(s, mRet, pos1Ret, dPosSubRet, r1Ret, dRSubRet, dRdmRet, dzdRRet, bHatRet, bMagRet, omb2Ret, accRet, chgRet);

                    vec2 dRet = (wPos - pos1Ret) - dPosSubRet;
                    vec2 eVel, eRad;
                    evalLWKernel(dRet, bHatRet, bMagRet, omb2Ret, accRet, chgRet, eVel, eRad);
                    totCompactE += (eVel + eRad);
                    totRadE += eRad;
                    if (s == 0) primaryEmitPos = pos1Ret + dPosSubRet;
                }

                if (u_readbackMode == 1) {
                    outColor = vec4(totCompactE.x, totCompactE.y, primaryEmitPos.x, primaryEmitPos.y);
                    return;
                }

                if (u_showWavePool == 0) {
                    vec3 darkBg = vec3(0.027, 0.039, 0.071);
                    if (u_showImpostors == 1) darkBg += evalConformalSphereImpostors(wPos);
                    outColor = vec4(clamp(darkBg, 0.0, 1.0), 1.0);
                    return;
                }

                vec2 displayVec = u_channel == 3 ? totRadE : totCompactE;
                // Balanced filmic auto-gain across N_p: zero white clipping, smooth water-caustic dynamic range
                float multiScaleGain = 1.0 / pow(float(max(1, u_activeCount)), 0.42);
                float mag = length(displayVec) * multiScaleGain;
                // Smooth filmic shoulder (never hard-clips to flat white even under strong acceleration)
                float logMag = log(1.0 + 1.85 * mag);
                float compressed = logMag / (1.35 + logMag);
                float val = clamp(0.14 + 0.82 * pow(compressed, 0.75), 0.0, 0.96);
                float ang = atan(displayVec.y, displayVec.x);
                float hue = fract((ang + PI) / (2.0 * PI));
                // Desaturate only slightly at the very highest intensity peak so colors stay rich and legible
                float sat = mix(0.78, 0.42, pow(compressed, 3.5));
                vec3 fieldRgb = hsv2rgb(vec3(hue, sat, val));

                vec3 poolBase = vec3(0.03, 0.11, 0.22) * (0.4 + 0.6 * compressed);
                float caustic = pow(clamp(compressed, 0.0, 1.0), 3.2) * 0.24;
                vec3 col = clamp(mix(poolBase, fieldRgb, clamp(compressed * 1.15 + 0.18, 0.0, 1.0)) + vec3(caustic * 0.6, caustic * 0.85, caustic), 0.0, 1.0);

                if (u_showImpostors == 1) {
                    col = clamp(col + evalConformalSphereImpostors(wPos) * 0.60, 0.0, 1.0);
                }

                outColor = vec4(col, 1.0);
            }`;
