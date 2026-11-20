// Origin: course-hydrodynamics homework02/marks.js. Changes for this report (eval-01):
//   file name evaluation-01-markers.json, storage keys "dte-eval01-task01-marks*", BLOCK_SEL with <pre>.
// ---- Study marks: flag a block or a selected text as "Struggle / Unsure / Got it", with a note.
//
// Data flow:
//   evaluation-01-markers.json (in git)  --build.py-->  <script id="mk-committed">  (the committed marks)
//   your clicks          -->  browser localStorage                        (the working copy)
//   "Export" button      -->  write evaluation-01-markers.json (linked file), or save, download or copy it
//   Ctrl+S (Mac: Cmd+S)  -->  the same as "Export", also when the panel is closed
//   "Link evaluation-01-markers.json"    -->  Chrome/Edge: the page reads this file at each load (File System Access API)
//
// Anchors: a block mark stores the key "<section id>:<hash of the block text>".
// A text mark stores the same key plus the selected text ("quote") and its occurrence ("occ").
// Math counts as its TeX source, for example "$\Wo$", so a formula can be part of a quote.
(function () {
  'use strict';

  // ------------------------------------------------------------------ settings
  const LS_KEY = 'dte-eval01-task01-marks-v1';
  const LEVELS = [
    { id: 'struggle', label: 'Struggle' },
    { id: 'unsure',   label: 'Unsure' },
    { id: 'ok',       label: 'Got it' },
  ];
  // Elements that get a flag. Nested ones get their own flag too.
  const BLOCK_SEL = 'p, li, pre, figure, div.box, tr';  // eval-01: + pre, - display math
  // Nested blocks that do not count in the key of their parent (so a change inside them keeps the parent key).
  const SKIP_IN_KEY = new Set(['DETAILS', 'OL', 'UL', 'TABLE', 'FIGURE']);

  const main = document.querySelector('main');
  if (!main) return;

  // ------------------------------------------------------------------ helpers
  function h(tag, props, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (k === 'class') el.className = v;
      else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
      else if (v !== false && v != null) el.setAttribute(k, v === true ? '' : v);
    }
    for (const kid of kids.flat(Infinity)) if (kid != null) el.append(kid);
    return el;
  }
  function fnv(s) {  // FNV-1a, 32 bit, as 8 hex digits
    let x = 0x811c9dc5;
    for (let i = 0; i < s.length; i++) { x ^= s.charCodeAt(i); x = Math.imul(x, 0x01000193); }
    return (x >>> 0).toString(16).padStart(8, '0');
  }
  const today = () => new Date().toISOString().slice(0, 10);
  const newId = () => 'm' + Date.now().toString(36) + Math.random().toString(36).slice(2, 5);
  const short = (s, n) => (s.length > n ? s.slice(0, n - 1) + '…' : s);
  function nthIndexOf(text, q, n) {
    let i = -1;
    for (let k = 0; k <= n; k++) { i = text.indexOf(q, i + 1); if (i < 0) return -1; }
    return i;
  }
  function countBefore(text, q, pos) {
    let n = 0, i = text.indexOf(q);
    while (i >= 0 && i < pos) { n++; i = text.indexOf(q, i + 1); }
    return n;
  }

  // ------------------------------------------------------------------ text of a block
  // Returns the normalised text (one space for each run of white space) and the segments that make it.
  // A text segment keeps "map": normalised index -> raw index in its text node.
  const texOf = new WeakMap();  // mjx-container -> TeX source
  function serialize(root, forKey) {
    let text = '', space = true;
    const segs = [];
    function addMath(el) {
      const t = texOf.get(el) || el.textContent;
      const s = el.getAttribute('display') === 'true' ? t : '$' + t + '$';
      segs.push({ node: el, math: true, start: text.length, end: text.length + s.length });
      text += s; space = false;
    }
    function walk(node) {
      for (const ch of node.childNodes) {
        if (ch.nodeType === 3) {
          const raw = ch.data, map = [];
          let norm = '';
          for (let i = 0; i < raw.length; i++) {
            if (/\s/.test(raw[i])) { if (space) continue; norm += ' '; space = true; }
            else { norm += raw[i]; space = false; }
            map.push(i);
          }
          if (norm) { segs.push({ node: ch, start: text.length, end: text.length + norm.length, map }); text += norm; }
        } else if (ch.nodeType === 1) {
          const tag = ch.tagName.toUpperCase();
          if (tag === 'MJX-CONTAINER') addMath(ch);
          else if (ch.classList.contains('mk-ui') || tag === 'SCRIPT' || tag === 'STYLE' || tag === 'SVG') continue;
          else if (forKey && SKIP_IN_KEY.has(tag)) continue;
          else walk(ch);
        }
      }
    }
    if (root.tagName.toUpperCase() === 'MJX-CONTAINER') addMath(root); else walk(root);
    return { text, segs };
  }

  // ------------------------------------------------------------------ blocks
  const blocks = [];              // { el, host, flag, idx, secId, where, key, keyText }
  const byKey = new Map();
  const byEl = new WeakMap();

  function sectionOf(el) {
    const chain = [];
    for (let d = el.closest('details.sec'); d; d = d.parentElement && d.parentElement.closest('details.sec')) {
      const hd = d.querySelector(':scope > summary [id]');
      if (hd) chain.unshift({ id: hd.id, title: serialize(hd, true).text.trim() });
    }
    if (!chain.length) return { id: 'top', where: 'Introduction and references' };
    return { id: chain[chain.length - 1].id, where: chain.map(c => c.title).join(' › ') };
  }

  function discover() {
    const seen = new Map();
    for (const el of main.querySelectorAll(BLOCK_SEL)) {
      if (el.closest('nav, .widget, .mk-ui')) continue;
      if (el.tagName === 'TR' && !el.querySelector('td')) continue;
      const keyText = serialize(el, true).text.trim();
      if (!keyText) continue;
      const sec = sectionOf(el);
      let key = sec.id + ':' + fnv(keyText);
      const n = seen.get(key) || 0;
      seen.set(key, n + 1);
      if (n) key += '~' + n;  // the same text twice in one section

      // The host shows the flag and the colour. A display equation gets a wrapper,
      // because the equation itself scrolls and would hide the flag.
      let host = el;
      if (el.tagName.toUpperCase() === 'MJX-CONTAINER') {
        host = h('span', { class: 'mk-eqwrap' });
        el.before(host);
        host.append(el);
      }
      const flag = h('button', { class: 'mk-ui mk-flag', type: 'button', title: 'Mark this block' }, '⚑');
      if (el.tagName === 'TR') el.lastElementChild.append(flag); else host.append(flag);
      host.classList.add('mk-host');
      el.classList.add('mk-block');

      const b = { el, host, flag, idx: blocks.length, secId: sec.id, where: sec.where, key, keyText };
      flag.addEventListener('click', e => { e.stopPropagation(); onFlag(b); });
      blocks.push(b); byKey.set(key, b); byEl.set(el, b);
    }
  }

  // ------------------------------------------------------------------ marks: load and save
  let committed = [];  // the baseline: evaluation-01-markers.json in the build, or the linked file
  try { committed = JSON.parse(document.getElementById('mk-committed').textContent).marks || []; } catch (e) { /* no committed marks */ }
  const FIELDS = ['id', 'kind', 'level', 'where', 'key', 'quote', 'occ', 'note', 'date'];
  const clean = m => Object.fromEntries(FIELDS.filter(f => m[f] !== undefined).map(f => [f, m[f]]));
  let base = fnv(JSON.stringify(committed.map(clean)));

  let marks = null, localBase = null, hadLocal = false;
  try {
    const s = JSON.parse(localStorage.getItem(LS_KEY));
    if (s && Array.isArray(s.marks)) { marks = s.marks; localBase = s.base; hadLocal = true; }
  } catch (e) { /* storage missing or blocked */ }
  if (!marks) { marks = committed.map(clean); localBase = base; }

  function save() {
    try { localStorage.setItem(LS_KEY, JSON.stringify({ base: localBase, marks })); } catch (e) { /* not kept */ }
  }

  // One mark per line, in document order: easy to read and to diff in git.
  function exportJSON(list) {
    const pos = m => { const r = resolve(m); return r ? r.b.idx * 1e6 + (r.s || 0) : 1e12; };
    const sorted = list.map(m => [pos(m), m]).sort((a, b) => a[0] - b[0]).map(x => x[1]);
    return '{"version": 1, "marks": [\n' + sorted.map(m => '  ' + JSON.stringify(clean(m))).join(',\n') +
      (sorted.length ? '\n' : '') + ']}\n';
  }
  const dirty = () => exportJSON(marks) !== exportJSON(committed);
  // A new build with new evaluation-01-markers.json: adopt it silently when it equals the local marks.
  let conflict = localBase !== base;

  // ------------------------------------------------------------------ marks: find their place
  function resolve(m) {
    let b = byKey.get(m.key);
    if (m.kind === 'text') {
      const inB = bb => bb && serialize(bb.el, false).text.includes(m.quote);
      if (!inB(b)) b = blocks.find(bb => bb.secId === m.key.split(':')[0] && inB(bb)) || blocks.find(inB);
      if (!b) return null;
      const text = serialize(b.el, false).text;
      let s = nthIndexOf(text, m.quote, m.occ || 0);
      if (s < 0) s = text.indexOf(m.quote);
      return { b, s, e: s + m.quote.length };
    }
    if (!b) {  // the block text changed: look for the same start in the same section
      const head = (m.quote || '').slice(0, 40);
      b = head && blocks.find(bb => bb.secId === m.key.split(':')[0] && bb.keyText.startsWith(head));
    }
    return b ? { b } : null;
  }

  // ------------------------------------------------------------------ show the marks in the page
  function clearHighlights() {
    for (const mk of main.querySelectorAll('mark.mk-hl')) { const p = mk.parentNode; mk.replaceWith(...mk.childNodes); p.normalize(); }
    for (const el of main.querySelectorAll('.mk-hlm')) { el.classList.remove('mk-hlm', ...LEVELS.map(l => 'mk-L-' + l.id)); delete el.dataset.mk; }
    for (const b of blocks) {
      b.host.classList.remove(...LEVELS.map(l => 'mk-L-' + l.id), 'mk-note');
      if (b.host !== b.el) b.el.classList.remove(...LEVELS.map(l => 'mk-L-' + l.id));
      b.flag.title = 'Mark this block';
    }
  }

  function highlight(m, r) {
    const { segs } = serialize(r.b.el, false);
    for (const g of segs) {
      if (g.end <= r.s || g.start >= r.e) continue;
      if (g.math) { g.node.classList.add('mk-hlm', 'mk-L-' + m.level); g.node.dataset.mk = m.id; continue; }
      const a = Math.max(r.s, g.start) - g.start, z = Math.min(r.e, g.end) - g.start;
      const rawA = g.map[a], rawZ = z < g.map.length ? g.map[z] : g.node.data.length;
      let node = g.node;
      if (rawA > 0) node = node.splitText(rawA);
      if (rawZ - rawA < node.data.length) node.splitText(rawZ - rawA);
      const mk = h('mark', { class: 'mk-hl mk-L-' + m.level, 'data-mk': m.id, title: m.note || LEVELS.find(l => l.id === m.level).label });
      node.before(mk);
      mk.append(node);
    }
  }

  function render() {
    clearHighlights();
    for (const m of marks) {
      const r = resolve(m);
      if (!r) continue;
      if (m.kind === 'text') { if (r.s >= 0) highlight(m, r); continue; }
      r.b.host.classList.add('mk-L-' + m.level);
      if (m.note) r.b.host.classList.add('mk-note');
      r.b.flag.title = LEVELS.find(l => l.id === m.level).label + (m.note ? ': ' + m.note : '');
    }
    if (focus) applyFocus(true);
    renderPanel();
  }

  // ------------------------------------------------------------------ popover: level and note of one mark
  const pop = h('div', { class: 'mk-ui mk-pop', hidden: true });
  let popMark = null;
  function openPop(m, rect) {
    popMark = m;
    pop.replaceChildren(
      h('div', { class: 'mk-pop-head' }, m.kind === 'text' ? '“' + short(m.quote, 70) + '”' : short(m.quote || '', 70)),
      h('div', { class: 'mk-levels' }, LEVELS.map(l => h('button', {
        type: 'button', class: 'mk-lv mk-L-' + l.id + (m.level === l.id ? ' mk-on' : ''),
        onclick: () => { m.level = l.id; save(); render(); openPop(m, rect); },
      }, l.label))),
      h('textarea', { rows: 3, placeholder: 'Note: what is unclear?', oninput: e => { m.note = e.target.value; save(); renderPanel(); } }),
      h('div', { class: 'mk-pop-foot' },
        h('button', { type: 'button', onclick: () => { marks = marks.filter(x => x !== m); closePop(); save(); render(); } }, 'Remove'),
        h('button', { type: 'button', onclick: closePop }, 'Close')),
    );
    pop.querySelector('textarea').value = m.note || '';
    pop.hidden = false;
    const left = Math.min(Math.max(rect.right + scrollX - 300, 8 + scrollX), scrollX + document.documentElement.clientWidth - 316);
    pop.style.left = left + 'px';
    pop.style.top = (rect.bottom + scrollY + 6) + 'px';
  }
  function closePop() { pop.hidden = true; popMark = null; }

  function onFlag(b) {
    let m = marks.find(x => x.kind === 'block' && resolve(x) && resolve(x).b === b);
    if (!m) {
      m = { id: newId(), kind: 'block', level: 'struggle', where: b.where, key: b.key, quote: short(b.keyText, 160), note: '', date: today() };
      marks.push(m); save(); render();
    }
    openPop(m, b.flag.getBoundingClientRect());
  }

  // ------------------------------------------------------------------ text selection
  const selBtn = h('button', { class: 'mk-ui mk-selbtn', type: 'button', hidden: true }, '⚑ Mark selection');
  let pending = null;
  function readSelection() {
    const sel = getSelection();
    if (!sel.rangeCount || sel.isCollapsed) return null;
    const r = sel.getRangeAt(0);
    let n = r.commonAncestorContainer;
    if (n.nodeType === 3) n = n.parentNode;
    if (!main.contains(n) || n.closest('.mk-ui')) return null;
    const el = n.closest('.mk-block');
    const b = el && byEl.get(el);
    if (!b) return { error: 'Select text inside one paragraph, step, equation or table row.', r };
    const { text, segs } = serialize(b.el, false);
    let s = Infinity, e = -1;
    for (const g of segs) {
      if (!r.intersectsNode(g.node)) continue;
      if (g.math) { s = Math.min(s, g.start); e = Math.max(e, g.end); continue; }
      const ra = g.node === r.startContainer ? r.startOffset : 0;
      const rz = g.node === r.endContainer ? r.endOffset : g.node.data.length;
      if (rz <= ra) continue;
      const idx = raw => { const i = g.map.findIndex(x => x >= raw); return i < 0 ? g.map.length : i; };
      s = Math.min(s, g.start + idx(ra)); e = Math.max(e, g.start + idx(rz));
    }
    while (s < e && text[s] === ' ') s++;
    while (e > s && text[e - 1] === ' ') e--;
    if (!(e > s)) return null;
    const quote = text.slice(s, e);
    return { b, quote, occ: countBefore(text, quote, s), r };
  }
  function onSelect() {
    pending = readSelection();
    if (!pending) { selBtn.hidden = true; return; }
    const rect = pending.r.getBoundingClientRect();
    selBtn.textContent = pending.error ? pending.error : '⚑ Mark selection';
    selBtn.classList.toggle('mk-err', !!pending.error);
    selBtn.hidden = false;
    selBtn.style.left = (rect.right + scrollX - 10) + 'px';
    selBtn.style.top = (rect.bottom + scrollY + 4) + 'px';
  }
  selBtn.addEventListener('mousedown', e => e.preventDefault());
  selBtn.addEventListener('click', () => {
    selBtn.hidden = true;
    if (!pending || pending.error) return;
    const rect = pending.r.getBoundingClientRect();
    const m = { id: newId(), kind: 'text', level: 'struggle', where: pending.b.where, key: pending.b.key,
                quote: pending.quote, occ: pending.occ, note: '', date: today() };
    marks.push(m);
    getSelection().removeAllRanges();
    save(); render();
    openPop(m, rect);
  });

  // ------------------------------------------------------------------ focus filter
  let focus = false;
  const openedByFocus = new Set();
  function markedEls() {
    const out = [];
    for (const m of marks) { const r = resolve(m); if (r) out.push(r.b.el); }
    return out;
  }
  function applyFocus(on) {
    document.body.classList.toggle('mk-focus', on);
    const lit = markedEls();
    for (const b of blocks) {
      const keep = lit.some(x => x.contains(b.el) || b.el.contains(x));
      b.host.classList.toggle('mk-dim', on && !keep);
    }
    if (on) {
      for (const x of lit) for (let d = x.closest('details'); d; d = d.parentElement && d.parentElement.closest('details'))
        if (!d.open) { d.open = true; openedByFocus.add(d); }
    } else {
      for (const d of openedByFocus) d.open = false;
      openedByFocus.clear();
    }
  }

  // ------------------------------------------------------------------ go to a mark
  function goTo(m) {
    const r = resolve(m);
    if (!r) return;
    for (let d = r.b.el.closest('details'); d; d = d.parentElement && d.parentElement.closest('details')) d.open = true;
    const target = (m.kind === 'text' && main.querySelector('[data-mk="' + m.id + '"]')) || r.b.host;
    target.scrollIntoView({ block: 'center', behavior: 'smooth' });
    target.classList.add('mk-flash');
    setTimeout(() => target.classList.remove('mk-flash'), 1600);
  }

  // ------------------------------------------------------------------ linked evaluation-01-markers.json (Chrome/Edge)
  // The file handle stays in IndexedDB. After a browser restart, Chrome asks again for permission.
  const canLink = 'showOpenFilePicker' in window;
  let fileHandle = null, fileState = 'none';  // none | ok | ask
  function idb(mode, fn) {
    return new Promise((res, rej) => {
      const rq = indexedDB.open('dte-eval01-task01-marks', 1);
      rq.onupgradeneeded = () => rq.result.createObjectStore('kv');
      rq.onerror = () => rej(rq.error);
      rq.onsuccess = () => {
        const tx = rq.result.transaction('kv', mode), r = fn(tx.objectStore('kv'));
        tx.oncomplete = () => res(r.result);
        tx.onerror = () => rej(tx.error);
      };
    });
  }
  const loadHandle = () => idb('readonly', st => st.get('file')).catch(() => null);
  const keepHandle = hd => idb('readwrite', st => st.put(hd, 'file')).catch(() => {});

  // Make the file the new baseline. Keep local changes only if they differ from the file.
  async function readLinked() {
    const list = JSON.parse(await (await fileHandle.getFile()).text()).marks || [];
    committed = list.map(clean);
    base = fnv(JSON.stringify(committed));
    if (!hadLocal || !dirty()) { marks = committed.map(clean); localBase = base; conflict = false; }
    else conflict = localBase !== base;
    fileState = 'ok';
    save(); render();
  }
  async function linkFile() {
    try {
      [fileHandle] = await showOpenFilePicker({ types: [{ description: 'evaluation-01-markers.json', accept: { 'application/json': ['.json'] } }] });
      await keepHandle(fileHandle);
      hadLocal = true;  // the marks in the page stay; the panel shows if they differ from the file
      await reconnect();
    } catch (e) { if (e.name !== 'AbortError') flashStatus('Link failed: ' + e.message); }
  }
  async function reconnect() {
    try {
      if (await fileHandle.requestPermission({ mode: 'readwrite' }) !== 'granted') return;
      await readLinked();
    } catch (e) { flashStatus('Read failed: ' + e.message); }
  }
  async function autoLoad() {
    if (!canLink) return;
    fileHandle = await loadHandle();
    if (!fileHandle) return;
    try {
      if (await fileHandle.queryPermission({ mode: 'readwrite' }) === 'granted') await readLinked();
      else { fileState = 'ask'; renderPanel(); }  // a click is necessary: see the bar button
    } catch (e) { fileState = 'ask'; renderPanel(); }
  }

  // ------------------------------------------------------------------ toolbar and panel
  const barCount = h('span', {});
  const focusBtn = h('button', { type: 'button', title: 'Dim all unmarked content and open the boxes with marks',
    onclick: () => { focus = !focus; focusBtn.classList.toggle('mk-on', focus); applyFocus(focus); } }, 'Focus');
  const bar = h('div', { class: 'mk-ui mk-bar' },
    h('button', { type: 'button', onclick: () => { panel.hidden = !panel.hidden; renderPanel(); } }, '⚑ Marks ', barCount),
    focusBtn);
  const askBtn = h('button', { type: 'button', hidden: true, title: 'Allow the page to read and write the linked evaluation-01-markers.json',
    onclick: () => reconnect() }, '↻ Load evaluation-01-markers.json');
  bar.append(askBtn);
  const panel = h('aside', { class: 'mk-ui mk-panel', hidden: true });
  const io = h('textarea', { class: 'mk-io', rows: 6, hidden: true, spellcheck: 'false' });

  function renderPanel() {
    const n = marks.length;
    barCount.textContent = '(' + n + ')' + (dirty() ? ' •' : '');
    askBtn.hidden = fileState !== 'ask';
    if (panel.hidden) return;
    const count = id => marks.filter(m => m.level === id).length;
    const groups = new Map(), lost = [];
    const rows = marks.map(m => [resolve(m), m]).sort((a, b) =>
      (a[0] ? a[0].b.idx * 1e6 + (a[0].s || 0) : 1e12) - (b[0] ? b[0].b.idx * 1e6 + (b[0].s || 0) : 1e12));
    for (const [r, m] of rows) {
      if (!r) { lost.push(m); continue; }
      if (!groups.has(r.b.where)) groups.set(r.b.where, []);
      groups.get(r.b.where).push(m);
    }
    const item = m => h('li', { class: 'mk-item', onclick: () => goTo(m) },
      h('span', { class: 'mk-dot mk-L-' + m.level }),
      h('span', { class: 'mk-kind' }, m.kind === 'text' ? '“…”' : '¶'),
      h('span', { class: 'mk-quote' }, short(m.quote || '', 110)),
      m.note ? h('div', { class: 'mk-item-note' }, m.note) : null);

    panel.replaceChildren(...[
      h('div', { class: 'mk-panel-head' }, h('b', {}, 'Study marks'),
        h('button', { type: 'button', onclick: () => { panel.hidden = true; } }, '×')),
      h('p', { class: 'mk-help' },
        'Click ⚑ at the right of a paragraph, step, equation or table row. ',
        'Or select words or a formula, then click “Mark selection”. ',
        'Click a mark to change its level or to write a note. ',
        'Press Ctrl+S (Mac: ⌘S) to export.'),
      h('div', { class: 'mk-counts' }, LEVELS.map(l => h('span', {}, h('span', { class: 'mk-dot mk-L-' + l.id }), count(l.id) + ' ' + l.label))),
      conflict ? h('div', { class: 'mk-warn' },
        'evaluation-01-markers.json changed after your last local edit. ',
        h('button', { type: 'button', onclick: () => { marks = committed.map(clean); localBase = base; conflict = false; save(); render(); } }, 'Use evaluation-01-markers.json'),
        h('button', { type: 'button', onclick: () => { localBase = base; conflict = false; save(); renderPanel(); } }, 'Keep my local marks')) : null,
      h('div', { class: 'mk-status' }, dirty()
        ? 'Some marks are not in evaluation-01-markers.json yet. Click “Export”.'
        : 'All marks are in evaluation-01-markers.json.'),
      h('div', { class: 'mk-file' },
        fileState === 'ok' ? 'Linked file: ' + fileHandle.name + ' (the page reads it at each load).'
        : fileState === 'ask' ? h('span', {}, 'Linked file: ' + fileHandle.name + '. ', h('button', { type: 'button', onclick: () => reconnect() }, 'Load it'))
        : canLink ? h('span', {}, 'No linked file. ', h('button', { type: 'button', onclick: linkFile, title: 'Pick context/evaluation-01/task01/evaluation-01-markers.json once. Then the page reads it at each load, and “Export” writes to it.' }, 'Link evaluation-01-markers.json…'))
        : 'This browser cannot link a file. The page uses the marks in the build and in the browser.'),
      h('div', { class: 'mk-actions' },
        h('button', { type: 'button', onclick: exportMarks, title: 'Write evaluation-01-markers.json. If that is not possible, save, download or copy it. Shortcut: Ctrl+S (Mac: ⌘S).' }, 'Export (Ctrl+S)'),
        h('button', { type: 'button', onclick: () => { io.hidden = false; io.value = ''; io.placeholder = 'Paste the text of a evaluation-01-markers.json file here, then click “Import”.'; io.focus(); } }, 'Paste…'),
        h('button', { type: 'button', onclick: importMarks }, 'Import'),
        h('button', { type: 'button', title: 'Drop the local changes', onclick: () => { if (dirty()) { marks = committed.map(clean); localBase = base; conflict = false; save(); render(); } } }, 'Revert to evaluation-01-markers.json')),
      io,
      [...groups].map(([where, list]) => h('div', { class: 'mk-group' }, h('div', { class: 'mk-where' }, where), h('ul', {}, list.map(item)))),
      lost.length ? h('div', { class: 'mk-group' },
        h('div', { class: 'mk-where' }, 'Not found: the text changed'),
        h('ul', {}, lost.map(m => h('li', { class: 'mk-item mk-lost' }, h('span', { class: 'mk-dot mk-L-' + m.level }),
          short(m.quote || '', 110), ' (' + m.where + ')', m.note ? h('div', { class: 'mk-item-note' }, m.note) : null)))) : null,
    ].flat(Infinity).filter(x => x != null));
    // Show the TeX in the list as formulas.
    if (window.MathJax && window.MathJax.typesetPromise) window.MathJax.typesetPromise([panel]).catch(() => {});
  }

  // Export, in this order: 1. the linked file, 2. a save dialog, 3. a download (local copy only), 4. the clipboard.
  async function exportMarks() {
    const text = exportJSON(marks);
    const written = () => { committed = marks.map(clean); base = fnv(JSON.stringify(committed)); localBase = base; conflict = false; save(); };
    try {
      if (fileHandle && await fileHandle.requestPermission({ mode: 'readwrite' }) === 'granted') {
        const w = await fileHandle.createWritable(); await w.write(text); await w.close();
        fileState = 'ok'; written(); renderPanel(); flashStatus('Saved to ' + fileHandle.name + '. Commit it with git.');
        return;
      }
      if ('showSaveFilePicker' in window) {
        const hd = await showSaveFilePicker({ suggestedName: 'evaluation-01-markers.json', types: [{ description: 'evaluation-01-markers.json', accept: { 'application/json': ['.json'] } }] });
        const w = await hd.createWritable(); await w.write(text); await w.close();
        fileHandle = hd; await keepHandle(hd);
        fileState = 'ok'; written(); renderPanel(); flashStatus('Saved to ' + hd.name + '. The page now reads this file at each load.');
        return;
      }
    } catch (e) {
      if (e.name === 'AbortError') return;  // the dialog was closed
      /* blocked, for example in a sandboxed frame: use the next way */
    }
    if (location.protocol === 'file:') {
      h('a', { href: URL.createObjectURL(new Blob([text], { type: 'application/json' })), download: 'evaluation-01-markers.json' }).click();
      flashStatus('Downloaded. Move it to context/evaluation-01/task01/evaluation-01-markers.json.');
      return;
    }
    const fallback = () => { io.hidden = false; io.value = text; io.focus(); io.select(); flashStatus('Copy the text below, then save it as context/evaluation-01/task01/evaluation-01-markers.json.'); };
    try { navigator.clipboard.writeText(text).then(() => flashStatus('Copied. Save it as context/evaluation-01/task01/evaluation-01-markers.json, or paste it to Claude.'), fallback); }
    catch (e) { fallback(); }
  }
  function importMarks() {
    try {
      const list = JSON.parse(io.value).marks;
      if (!Array.isArray(list)) throw new Error('no "marks" list');
      marks = list.map(clean); save(); io.hidden = true; render();
    } catch (e) { flashStatus('Import failed: ' + e.message); }
  }
  function flashStatus(msg) { const s = panel.querySelector('.mk-status'); if (s) s.textContent = msg; toast(msg); }
  // A short message at the bottom of the page, so that it shows also when the panel is closed.
  function toast(msg) {
    const t = h('div', { class: 'mk-ui mk-toast', role: 'status' }, msg);
    document.body.append(t);
    setTimeout(() => t.remove(), 4000);
  }

  // ------------------------------------------------------------------ events and start
  function start() {
    discover();
    document.body.append(pop, selBtn, bar, panel);
    pop.addEventListener('click', e => e.stopPropagation());  // its buttons rebuild the popover
    let hot = null;
    main.addEventListener('mouseover', e => {
      const x = e.target.closest('.mk-host');
      if (x !== hot) { if (hot) hot.classList.remove('mk-hot'); hot = x; if (x) x.classList.add('mk-hot'); }
    });
    document.addEventListener('mouseup', e => { if (!(e.target.closest && e.target.closest('.mk-ui'))) setTimeout(onSelect, 0); });
    document.addEventListener('click', e => {
      const hit = e.target.closest('[data-mk]');
      if (hit && !e.target.closest('.mk-ui') && getSelection().isCollapsed) {
        const m = marks.find(x => x.id === hit.dataset.mk);
        if (m) { openPop(m, hit.getBoundingClientRect()); return; }
      }
      if (!pop.hidden && !e.target.closest('.mk-pop, .mk-flag, .mk-selbtn')) closePop();
    });
    document.addEventListener('keydown', e => { if (e.key === 'Escape') { closePop(); selBtn.hidden = true; } });
    // Shortcut: Ctrl+S (Mac: Cmd+S) exports the marks, in the place of the browser "Save page".
    document.addEventListener('keydown', e => {
      if ((e.ctrlKey || e.metaKey) && !e.shiftKey && !e.altKey && e.key.toLowerCase() === 's') { e.preventDefault(); exportMarks(); }
    });
    if (conflict && !dirty()) { localBase = base; conflict = false; save(); }  // new evaluation-01-markers.json equals the local marks
    render();
    autoLoad();
  }

  window.addEventListener('load', () => {
    const mj = window.MathJax && window.MathJax.startup && window.MathJax.startup.promise;
    (mj || Promise.resolve()).then(() => {
      try { for (const item of window.MathJax.startup.document.math) texOf.set(item.typesetRoot, item.math); } catch (e) { /* no MathJax */ }
      start();
    });
  });
})();
