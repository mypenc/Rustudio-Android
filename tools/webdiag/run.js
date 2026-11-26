// Loads the page in headless Chromium (software GL), records console/page
// errors, and screenshots the canvas over time.
const { chromium } = require('playwright');
const fs = require('fs');
const URL = process.env.URL || 'https://mypenc.github.io/Rustudio-Android/';
const SECS = parseInt(process.env.SECS || '150', 10);
(async () => {
  fs.mkdirSync('out', { recursive: true });
  const b = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--enable-webgl', '--enable-unsafe-webgpu'] });
  const ctx = await b.newContext({ viewport: { width: 1280, height: 800 } });
  const p = await ctx.newPage();
  const log = [];
  const t0 = Date.now();
  const stamp = () => ((Date.now() - t0) / 1000).toFixed(1).padStart(6);
  p.on('console', m => log.push(`${stamp()} [${m.type()}] ${m.text()}`));
  p.on('pageerror', e => log.push(`${stamp()} [pageerror] ${e.stack || e}`));
  p.on('requestfailed', r => log.push(`${stamp()} [reqfail] ${r.url()} ${r.failure()?.errorText}`));
  p.on('crash', () => log.push(`${stamp()} [CRASH] page crashed`));
  log.push(`url ${URL}`);
  log.push('webgl2/webgpu: ' + JSON.stringify(await p.evaluate(() => ({ webgpu: !!navigator.gpu, sab: typeof SharedArrayBuffer, coi: self.crossOriginIsolated, cores: navigator.hardwareConcurrency }))).slice(0, 200));
  await p.goto(URL, { waitUntil: 'domcontentloaded' }).catch(e => log.push('goto: ' + e));
  for (let t = 15; t <= SECS; t += 15) {
    await p.waitForTimeout(15000);
    const status = await p.evaluate(() => document.getElementById('status')?.textContent || '').catch(() => '?');
    log.push(`${stamp()} [status] ${status.replace(/\n/g, ' | ').slice(0, 300)}`);
    if (t % 45 === 0 || t === SECS) await p.screenshot({ path: `out/shot-${String(t).padStart(3, '0')}.png` }).catch(() => {});
    fs.writeFileSync('out/console.log', log.join('\n'));
  }
  fs.writeFileSync('out/console.log', log.join('\n'));
  await b.close();
})();
