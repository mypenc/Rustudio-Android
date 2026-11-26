// Loads the page in headless Chromium (software GL), records console/page
// errors, and screenshots over time. Every wait is bounded: a page whose main
// thread is hung must still produce a report.
const { chromium } = require('playwright');
const fs = require('fs');
const URL = process.env.URL || 'https://mypenc.github.io/Rustudio-Android/';
const SECS = parseInt(process.env.SECS || '150', 10);
const race = (p, ms, tag) => Promise.race([p, new Promise(r => setTimeout(() => r(`<${tag} timed out after ${ms}ms: main thread busy?>`), ms))]);
(async () => {
  fs.mkdirSync('out', { recursive: true });
  const log = [];
  const t0 = Date.now();
  const stamp = () => ((Date.now() - t0) / 1000).toFixed(1).padStart(6);
  const flush = () => fs.writeFileSync('out/console.log', log.join('\n'));
  const add = l => { log.push(`${stamp()} ${l}`); flush(); };
  setTimeout(() => { add('[hard stop]'); process.exit(0); }, (SECS + 60) * 1000);
  const b = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--enable-webgl', '--enable-unsafe-webgpu'] });
  const ctx = await b.newContext({ viewport: { width: 1280, height: 800 } });
  const p = await ctx.newPage();
  p.on('console', m => add(`[${m.type()}] ${m.text().slice(0, 6000)}`));
  p.on('pageerror', e => add(`[pageerror] ${String(e.stack || e).slice(0, 1200)}`));
  p.on('requestfailed', r => add(`[reqfail] ${r.url()} ${r.failure()?.errorText}`));
  p.on('response', r => { if (r.status() >= 400) add(`[http ${r.status()}] ${r.url()}`); });
  p.on('crash', () => add('[CRASH] page crashed'));
  const cdp = await ctx.newCDPSession(p);
  await cdp.send('Debugger.enable');
  let samples = 0;
  cdp.on('Debugger.paused', ev => {
    samples++;
    const fr = ev.callFrames.slice(0, 40).map(f => `${f.functionName || '?'} @${(f.url || '').split('/').pop() || f.location.scriptId}:${f.location.lineNumber}:${f.location.columnNumber}`);
    add(`[PAUSED sample ${samples}] reason=${ev.reason}\n   ` + fr.join('\n   '));
    cdp.send('Debugger.resume').catch(() => {});
  });
  add(`url ${URL}`);
  await p.goto(URL, { waitUntil: 'commit', timeout: 60000 }).catch(e => add('goto: ' + e));
  for (let t = 10; t <= SECS; t += 10) {
    await new Promise(r => setTimeout(r, 10000));
    if (t >= 20 && t <= 80 && t % 10 === 0) cdp.send('Debugger.pause').catch(() => {});
    const info = await race(p.evaluate(() => ({
      status: (document.getElementById('status') || {}).textContent || '',
      canvas: (() => { const c = document.querySelector('canvas'); return c ? `${c.width}x${c.height}` : 'none'; })(),
      sab: typeof SharedArrayBuffer, coi: self.crossOriginIsolated, gpu: !!navigator.gpu,
    })), 5000, 'evaluate');
    add(`[probe] ${typeof info === 'string' ? info : JSON.stringify(info).replace(/\\n/g, ' | ').slice(0, 400)}`);
    if (t % 30 === 0) await race(p.screenshot({ path: `out/shot-${String(t).padStart(3, '0')}.png`, timeout: 8000 }).catch(() => {}), 10000, 'screenshot');
  }
  add('[done]');
  process.exit(0);
})();
