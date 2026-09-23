// Minimal fake DOM that runs the dashboard's real chart script (server.js_source()) so zoom/hover logic can be tested without a browser.
const fs = require('fs');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));

class El {
  constructor(tag, attrs, text) { this.tagName = tag; this.attrs = Object.assign({}, attrs || {}); this.children = []; this.parentNode = null; this.style = {}; this.handlers = {}; this._text = text || ''; }
  get checked() { return this._checked !== undefined ? this._checked : ('checked' in this.attrs); }
  set checked(v) { this._checked = !!v; }
  get value() { return this.attrs.value; }
  get id() { return this.attrs.id; }
  set id(v) { this.attrs.id = v; }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  get classList() { const el = this; const set = () => new Set((el.attrs.class || '').split(/\s+/).filter(Boolean));
    return { contains: c => set().has(c), add: c => { const s = set(); s.add(c); el.attrs.class = [...s].join(' ') }, remove: c => { const s = set(); s.delete(c); el.attrs.class = [...s].join(' ') },
             toggle: (c, f) => { const s = set(); const on = f === undefined ? !s.has(c) : !!f; on ? s.add(c) : s.delete(c); el.attrs.class = [...s].join(' '); return on } }; }
  get textContent() { return this._text + this.children.map(c => c.textContent).join(''); }
  set textContent(v) { this._text = String(v); this.children = []; }
  get viewBox() { const p = (this.attrs.viewBox || '0 0 0 0').split(' ').map(Number); return { baseVal: { width: p[2], height: p[3] } }; }
  getBoundingClientRect() { const p = (this.attrs.viewBox || '0 0 640 220').split(' ').map(Number); return { left: 0, top: 0, width: p[2], height: p[3] }; }
  appendChild(c) { c.parentNode = this; this.children.push(c); return c; }
  insertBefore(c, ref) { if (c.parentNode) c.parentNode.removeChild(c); c.parentNode = this; const i = this.children.indexOf(ref); this.children.splice(i < 0 ? this.children.length : i, 0, c); return c; }
  removeChild(c) { this.children = this.children.filter(x => x !== c); c.parentNode = null; return c; }
  addEventListener(t, f) { (this.handlers[t] = this.handlers[t] || []).push(f); }
  setPointerCapture() {}
  click() { fire(this, 'click', {}); }
  all(pred, out = []) { for (const c of this.children) { if (pred(c)) out.push(c); c.all(pred, out); } return out; }
  querySelectorAll(sel) { return this.all(matcher(sel)); }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
}
function matcher(sel) {
  const m = sel.match(/^([a-z]*)((?:\.[\w-]+)*)(?:\[([\w-]+)\])?$/); if (!m) throw new Error('selector ' + sel);
  const classes = m[2].split('.').filter(Boolean);
  return el => (!m[1] || el.tagName === m[1]) && classes.every(c => el.classList.contains(c)) && (!m[3] || m[3] in el.attrs);
}
function fire(el, type, props) { const ev = Object.assign({ type, prevented: false, preventDefault() { this.prevented = true } }, props); (el.handlers[type] || []).forEach(f => f(ev)); return ev; }
function build(n) { const el = new El(n.tag, n.attrs, n.text); (n.children || []).forEach(c => el.appendChild(build(c))); return el; }

const body = build(input.tree);
const document = { body, createElement: t => new El(t), createElementNS: (ns, t) => new El(t),
  getElementById: id => body.all(e => e.attrs.id === id)[0] || null, querySelectorAll: s => body.querySelectorAll(s) };
document.body.appendChild = El.prototype.appendChild;
const window = { innerWidth: 1000 };
new Function('document', 'window', input.js)(document, window);
const out = {};
const svgs = body.querySelectorAll('svg.chart');
const btn = (svg, y) => body.querySelectorAll('.zctl').find(c => c.attrs['data-for'] === svg.attrs['data-uid']).querySelectorAll('.zbtn').find(b => b.attrs['data-y'] === String(y));
const tf = svg => { const g = svg.querySelectorAll('.zoom').pop(); const m = (g.attrs.transform || '').match(/translate\(([-\d.e]+),0\) scale\(([-\d.e]+),1\)/); return m ? { tx: +m[1], s: +m[2] } : { tx: 0, s: 1 }; };
const tip = () => document.getElementById('tip').textContent;
const hover = (svg, x) => { fire(svg, 'pointermove', { pointerId: 9, pointerType: 'mouse', buttons: 0, clientX: x, clientY: 50 }); };
const vline = svg => +svg.querySelector('.vline').attrs.x1;
const vertexX = (svg, month) => { const data = JSON.parse(svg.attrs['data-h']); const i = data.findIndex(p => p[0].slice(0, 7) === month);
  const g = svg.querySelectorAll('.zoom').pop(); const pl = g.querySelectorAll('polyline').pop(); const x = +pl.attrs.points.split(' ')[i].split(',')[0]; const t = tf(svg); return t.tx + t.s * x; };

const svg = svgs[1];                               // the chart with the odd (partial-month) last date
out.initialTransform = tf(svg);
btn(svg, 2).click(); out.zoom2 = tf(svg);
out.buttonOn = body.querySelectorAll('.zctl').find(c => c.attrs['data-for'] === svg.attrs['data-uid']).querySelectorAll('.zbtn').filter(b => b.classList.contains('on')).map(b => b.attrs['data-y']);
out.hoverErr = [];
for (const x of [70, 200, 333, 450, 560, 575]) { hover(svg, x); const m = tip().slice(0, 7); out.hoverErr.push(Math.abs(vline(svg) - vertexX(svg, m))); }
// right-click must not start a drag
const before = JSON.stringify(tf(svg));
fire(svg, 'pointerdown', { pointerId: 1, pointerType: 'mouse', button: 2, buttons: 2, clientX: 300, clientY: 50 });
fire(svg, 'pointermove', { pointerId: 1, pointerType: 'mouse', button: 0, buttons: 0, clientX: 380, clientY: 50 });
out.rightClickMoved = JSON.stringify(tf(svg)) !== before;
// left drag pans
fire(svg, 'pointerdown', { pointerId: 2, pointerType: 'mouse', button: 0, buttons: 1, clientX: 300, clientY: 50 });
fire(svg, 'pointermove', { pointerId: 2, pointerType: 'mouse', button: 0, buttons: 1, clientX: 340, clientY: 50 });
fire(svg, 'pointerup', { pointerId: 2, pointerType: 'mouse', clientX: 340, clientY: 50 });
out.dragMoved = JSON.stringify(tf(svg)) !== before;
// ctrl+wheel zooms and prevents the page's own zoom; plain wheel is left alone
const w1 = fire(svg, 'wheel', { ctrlKey: true, deltaY: -100, deltaMode: 0, deltaX: 0, clientX: 300 }); const w2 = fire(svg, 'wheel', { deltaY: 100, deltaX: 0, clientX: 300 });
out.wheel = [w1.prevented, w2.prevented];
fire(svg, 'dblclick', {}); out.reset = tf(svg);
// forward view: known and unknown next year
body.querySelectorAll('.mview-box').filter(r => r.attrs.value === 'fwd')[0].checked = true;
fire(body.querySelectorAll('.mview-box').filter(r => r.attrs.value === 'fwd')[0], 'change', {});
out.view = body.attrs['data-mview'];
hover(svg, 300); out.fwdMid = tip();
hover(svg, 575); out.fwdEnd = tip();
const xAt = d => { const dat = JSON.parse(svg.attrs['data-h']); const t0 = Date.parse(dat[0][0]), t1 = Date.parse(dat[dat.length - 1][0]); return 60 + (Date.parse(d) - t0) / (t1 - t0) * 516; };
hover(svg, xAt('2025-09-30')); out.fwdJustAfter = tip();      // 30 days after the last known forward value: must not reuse it
hover(svg, xAt('2025-08-31')); out.fwdLastKnown = tip();
// a broken chart must not stop the others: the first svg has garbage data
out.brokenFirst = svgs[0].attrs['data-h'] === 'not json';
btn(svg, 5).click(); out.zoomAfterBroken = tf(svg).s > 1;
out.hiddenControl = svgs.map(s => body.querySelectorAll('.zctl').find(c => c.attrs['data-for'] === s.attrs['data-uid']).style.display || '');
console.log(JSON.stringify(out));
