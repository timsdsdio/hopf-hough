# v6f kernel studies (preprint v6.2, §2.4 and §2.6)

Scripts and recorded results behind the v6f numbers in preprint v6.2.

- `f32newton_2d/`: attribution of v6's float32 error floor (the in-segment chord deficit); v6f accuracy, op counts and relative timing; v5/v6 bit-identity and drag hand-off checks.
- `f32newton_floor/`: attribution and fix of v6f's remaining floor (float32 direction rounding near the crest); numpy float32 emulation (`emu_field.py`, `emu_fix.py`).

The studies were run against a development copy of `web/index.html` that is identical to the v6.2.0 app. Paths now point at `web/index.html`, and v6.1's app is at `reference/web_v6_1.html`. Intermediate snapshots used only for the before/after timing in `f32newton_floor/make_timing.py` (`app2d_before.html`, `app2d_fixA.html`) are not included; the recorded results are in the `results_*.json` and `*.log` files.

GPU timings are from headless Chromium with SwiftShader and are relative only. Run with `node <script>.mjs` from the repository root after `npm ci`.
