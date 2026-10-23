# PyCharge cross-validation

Independent check of hopf-hough Liénard–Wiechert fields against
[PyCharge](https://github.com/MatthewFilipovich/pycharge) v2 (JAX), plus a
Float64 port of the original solver's analytic reference. Results: `REPORT.md`.

## Install

```
pip install pycharge --break-system-packages   # tested: pycharge 2.0.1, jax 0.12.0.1, Python 3.13
```

## Files

| file | purpose |
|---|---|
| `pycharge_fields.py` | CLI/library: JSON in, PyCharge E and B_z out (code units) |
| `reference_port.py` | Port of `getExactTrajectoryKinematics`, `evaluateExactAnalyticalLW`, `getPulseCrossingSweepPoints` |
| `trajectories.py` | Preset → orbit parameters (same clamps as the original) and JAX position functions |
| `compare.py` | Runs every case, prints the table, writes `results.json` |
| `example_input.json` | Minimal input |

## `pycharge_fields.py` JSON schema

```
python3 pycharge_fields.py input.json [output.json]     # no output path -> stdout
```

### Input

```jsonc
{
  "trajectory": { ... },     // one trajectory object, OR a list of them (fields are summed);
                             // "trajectories" works as the key too
  "t": 2.508,                // observation time, code units
  "points": [[x, y], ...],   // observers in the z = 0 plane, code units
  "c": 2.35,                 // optional, speed of light in code units (default 2.35)
  "L0_m": 1.0                // optional, metres per code length unit (results do not depend on it)
}
```

Trajectory objects (every one also takes optional `"q"` (default 1) and `"phase"` (rad, default 0;
not used by `linear`)):

| preset | keys | motion |
|---|---|---|
| `synchrotron` | `beta`, `omega` (default 1.6) | circle at the origin, `r = min(0.85, βc/max(0.5, ω))`, angular rate `βc/r` (the original's clamps, so for β > 0.579 the rate is **not** `omega`) |
| `oscillator` | `beta`, `omega` (default 1.6), `cap_amplitude` (default `true`) | `x = A sin(ωt + phase)`, `y = 0`. With the cap (as the original does it) `A = min(0.75, βc/max(0.4, ω))`; `false` uses `A = βc/ω`, so `beta` really is β_max |
| `linear` | `beta`, `angle` (rad, default 0), `x0`, `y0`, `t0` (default 0) | straight line: `r(t) = (x0, y0) + βc(cos angle, sin angle)(t − t0)`. This is **not** the original's sawtooth `uniform` preset |

`swarm` and `uniform` are not supported on purpose (see REPORT.md).

### Output

```jsonc
{
  "E":   [[Ex, Ey], ...],    // same order as "points"
  "B_z": [...],              // B_z in Gaussian units, B = n × E, so it has the same units as E
  "meta": {"pycharge_version": "2.0.1", "c": 2.35, "units": "..."}
}
```

Units: Gaussian-like code units. A static unit charge gives `|E| = 1/R²`;
`E = q(1−β²)(n−β)/(κ³R²) + q n×((n−β)×β̇)/(cκ³R)` with β̇ = dβ/dt. **No
Plummer softening** (ε = 0). To compare against a softened GPU output, either run
the GPU with ε = 0 or expect differences of about ε²/R² (see REPORT.md).

If PyCharge's Newton retarded-time solve does not converge, or an observer sits on the
trajectory, the script raises an error and writes nothing.

## Rerun the validation

```
cd validation/pycharge && python3 compare.py     # about 20 s on CPU
```
