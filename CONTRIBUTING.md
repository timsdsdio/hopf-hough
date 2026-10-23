# Contributing

Thanks for your interest. Bug reports, accuracy measurements on your hardware, and pull requests are all welcome.

## Reporting problems

Open an issue with:
- what you did (scenario, settings, or the URL from the app's address bar),
- what you expected and what happened,
- your browser and GPU (the verification report in the app prints the renderer).

Accuracy reports are especially useful: run **Run verification suite** in the app and attach the downloaded JSON.

## Development

```bash
npm install
npm run build        # library, single-file app, browser bundle
npm test             # unit tests (Node)
npm run test:gpu     # headless Chromium with software WebGL2
npm run tables       # regenerate results/tables.json (slow in software rendering)
```

Ground rules for changes to the solver:
- The legacy path must keep reproducing the original solver (`tests/gpu/legacy_regression.mjs`, 154/154 values).
- Accuracy claims must come from the test or table scripts, measured against the exact reference with softening off.
- New trajectory types must return kinematically consistent position, β and a/c², and refuse superluminal input.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
