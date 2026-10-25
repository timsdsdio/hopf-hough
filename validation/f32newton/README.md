# v6f kernel studies (preprint v6.2, §2.4 and §2.6)

Scripts and recorded results behind the v6f numbers in preprint v6.2.

- `f32newton_2d/`: attribution of v6's float32 error floor (the in-segment chord deficit); v6f accuracy, op counts and relative timing; v5/v6 bit-identity and drag hand-off checks.
- `f32newton_floor/`: attribution and fix of v6f's remaining floor (float32 direction rounding near the crest); numpy float32 emulation (`emu_field.py`, `emu_fix.py`).

The studies were run against a development copy of `web/index.html` that is identical to the v6.2.0 app. Paths now point at `web/index.html`, and the app of main at commit 904bcce (the v6 kernel with the κ-form and Newton fixes, before this release) is at `reference/web_v6_1.html`. The counter and diagnostic builds (`make_diag.py`, used by `diag_test.mjs`) patch the app as it was before the field-floor fix and no longer apply to the current `web/index.html`; their recorded outputs are `results_counts.json` and `results_attrib.json`. Intermediate snapshots used only for the before/after timing in `f32newton_floor/make_timing.py` (`app2d_before.html`, `app2d_fixA.html`) are not included; the recorded results are in the `results_*.json` and `*.log` files.

GPU timings are from headless Chromium with SwiftShader and are relative only. Run with `node <script>.mjs` from the repository root after `npm ci`.

Cost after the v6.2 review: `f32newton_2d/opcount_2d.py` now also counts the field-floor fix (readback and display paths, γ gate false and true; `results_opcount.json`), and `f32newton_floor/frame_ratio_v62.mjs` re-measures the SwiftShader frame-time ratios of v6, v6f and v6f with `F32N_NOFIX`, ring and known path, below and above the gate, over several interleaved rounds (`results_frame_ratio_v62.json`).
