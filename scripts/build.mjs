// Builds the library (dist/lib) and the zero-install single-file app (dist/hopf-hough.html).
import { build } from 'esbuild';
import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';

const root = new URL('..', import.meta.url).pathname;
mkdirSync(root + 'dist', { recursive: true });

execFileSync(root + 'node_modules/.bin/tsc', ['-p', root], { stdio: 'inherit' });

const css = execFileSync(root + 'node_modules/.bin/tailwindcss', ['-c', root + 'tailwind.config.cjs', '-i', root + 'app/styles.css', '--minify'], { cwd: root, stdio: ['ignore', 'pipe', 'ignore'] }).toString();

const js = await build({
  entryPoints: [root + 'app/app.ts'],
  bundle: true, format: 'iife', minify: true, write: false, target: 'es2020',
  legalComments: 'none',
  banner: { js: '/* hopf-hough · MIT License · https://github.com/timsdsdio/hopf-hough */' },
});
const code = js.outputFiles[0].text.replace(/<\/script/gi, '<\\/script');
const html = readFileSync(root + 'app/index.html', 'utf8')
  .replace('/*__CSS__*/', () => css)
  .replace('/*__JS__*/', () => code);
writeFileSync(root + 'dist/hopf-hough.html', html);

// Browser bundle of the library for scripting (window.HopfHough).
await build({
  entryPoints: [root + 'src/index.ts'], bundle: true, format: 'iife', globalName: 'HopfHough',
  outfile: root + 'dist/hopf-hough.iife.js', minify: false, target: 'es2020',
});
console.log(`dist/hopf-hough.html  ${(html.length / 1024).toFixed(0)} KiB (no external requests)`);
