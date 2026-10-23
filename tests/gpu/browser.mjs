// Shared headless-Chromium launcher (software WebGL2 via SwiftShader).
// Uses CHROME_PATH or a pre-installed Playwright Chromium if present; otherwise the
// browser installed by `npx playwright-core install chromium`.
import { chromium } from 'playwright-core';
import { existsSync } from 'node:fs';

const CANDIDATES = [process.env.CHROME_PATH, '/opt/pw-browsers/chromium-1243/chrome-linux64/chrome'].filter(Boolean);

export async function launch() {
  const executablePath = CANDIDATES.find((p) => existsSync(p));
  return chromium.launch({
    ...(executablePath ? { executablePath } : {}),
    args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
  });
}
