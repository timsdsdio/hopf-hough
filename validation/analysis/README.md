# Analyses behind the version-6 corrections

These scripts were written during the review of preprint version 5. They run against
the original single-file solver (`reference/original_solver.html`), extracted into
standalone Node/Python code, and their saved outputs (`*.out`) are the numbers quoted
in the preprint and in `docs/ACCURACY.md`.

| folder | question | report |
|---|---|---|
| `precision/` | Where does the float32 accuracy floor at γ ≈ 70 come from, and does a double-float root function remove it? | [REPORT.md](precision/REPORT.md) |
| `limiter/` | How conservative is the Theorem 1 limiter, is there a sharper certificate, and when is one needed? | [REPORT.md](limiter/REPORT.md) |
| `farfield/` | Could a far-field (τ, S²) store replace the worldline history, and is a Hopf-bundle representation useful? | [REPORT.md](farfield/REPORT.md) |

The reports are working notes from that review; the repository's tests and `results/` supersede
their numbers where they differ.
