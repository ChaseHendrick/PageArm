// A residual family ranker for the shelf scripts.
// Same shape as a small cipher-family router: tanh layers, a residual hop,
// label smoothing, and a promotion gate. It ranks five PageArm scripts.
// It does not decipher anything, and it does not load anyone else's weights.

const FAMILIES = ["fill", "table", "form", "required", "glance"];
const WIDTH = 8;
const HIDDEN = 12;
const SEED = 20261004;

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randn(rng) {
  const u = Math.max(rng(), 1e-9);
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function zeros(n) {
  return Array.from({ length: n }, () => 0);
}

function matrix(rows, cols, rng, scale) {
  const out = [];
  for (let r = 0; r < rows; r++) {
    const row = [];
    for (let c = 0; c < cols; c++) row.push(randn(rng) * scale);
    out.push(row);
  }
  return out;
}

function tanh(x) {
  const y = Math.tanh(x);
  return Number.isFinite(y) ? y : (x > 0 ? 1 : -1);
}

export function features(glance) {
  const g = glance && typeof glance === "object" ? glance : {};
  const fields = Array.isArray(g.fields) ? g.fields.length : 0;
  const buttons = Array.isArray(g.buttons) ? g.buttons : [];
  const headings = Array.isArray(g.headings) ? g.headings.length : 0;
  const hasTable = g.table && Array.isArray(g.table.rows) && g.table.rows.length ? 1 : 0;
  const hasSave = buttons.some((b) => /save|submit|claim/i.test(String(b || ""))) ? 1 : 0;
  return [
    Math.min(fields, 12) / 12,
    Math.min(buttons.length, 12) / 12,
    hasTable,
    Math.min(headings, 8) / 8,
    hasSave,
    fields > 0 && buttons.length === 0 ? 1 : 0,
    hasTable && fields < 2 ? 1 : 0,
    Math.min(String(g.title || "").length, 120) / 120,
  ];
}

function forward(p, x) {
  const h0 = zeros(HIDDEN);
  for (let j = 0; j < HIDDEN; j++) {
    let s = p.b0[j];
    for (let i = 0; i < WIDTH; i++) s += x[i] * p.w0[i][j];
    h0[j] = tanh(s);
  }
  const h1 = zeros(HIDDEN);
  for (let j = 0; j < HIDDEN; j++) {
    let s = p.b1[j];
    for (let i = 0; i < HIDDEN; i++) s += h0[i] * p.w1[i][j];
    h1[j] = tanh(s) + h0[j];
  }
  const logits = zeros(FAMILIES.length);
  for (let k = 0; k < FAMILIES.length; k++) {
    let s = p.b2[k];
    for (let j = 0; j < HIDDEN; j++) s += h1[j] * p.w2[j][k];
    logits[k] = s;
  }
  return logits;
}

function softmax(logits) {
  const m = Math.max(...logits);
  const ex = logits.map((v) => Math.exp(v - m));
  const z = ex.reduce((a, b) => a + b, 0) || 1;
  return ex.map((v) => v / z);
}

function sampleGlance(family, rng) {
  const g = { title: "Page", headings: [], fields: [], buttons: [], table: null };
  const n = (lo, span) => lo + Math.floor(rng() * span);
  if (family === "table") {
    g.table = { rows: [["lot", "vessel"], ["117", "Red Dory"]] };
    g.title = "Lots";
    if (rng() < 0.4) g.buttons = ["Copy"];
  } else if (family === "fill") {
    for (let i = 0; i < n(3, 5); i++) g.fields.push({ name: "f" + i, type: "text", tag: "input" });
    g.buttons = [rng() < 0.5 ? "Save" : "Save claim"];
    g.headings = ["Ticket"];
  } else if (family === "form") {
    for (let i = 0; i < n(2, 4); i++) g.fields.push({ name: "f" + i, type: "text", tag: "input" });
    g.title = "Record";
  } else if (family === "required") {
    for (let i = 0; i < n(1, 4); i++) g.fields.push({ name: "f" + i, type: "text", tag: "input" });
    g.headings = ["Needed"];
  } else {
    g.headings = ["Notes", "Later"].slice(0, n(1, 2));
    g.title = "Reading";
  }
  return g;
}

function init(rng) {
  return {
    w0: matrix(WIDTH, HIDDEN, rng, Math.sqrt(2 / (WIDTH + HIDDEN))),
    b0: zeros(HIDDEN),
    w1: matrix(HIDDEN, HIDDEN, rng, Math.sqrt(1 / HIDDEN)),
    b1: zeros(HIDDEN),
    w2: matrix(HIDDEN, FAMILIES.length, rng, Math.sqrt(2 / (HIDDEN + FAMILIES.length))),
    b2: zeros(FAMILIES.length),
  };
}

function train() {
  const rng = mulberry32(SEED);
  const p = init(rng);
  const rows = [];
  for (let n = 0; n < 40; n++) {
    FAMILIES.forEach((family, label) => {
      rows.push({ x: features(sampleGlance(family, rng)), label });
    });
  }
  const rate = 0.08;
  for (let epoch = 0; epoch < 160; epoch++) {
    for (const row of rows) {
      const logits = forward(p, row.x);
      const prob = softmax(logits);
      const target = FAMILIES.map((_, k) => (k === row.label ? 1 - 0.03 : 0.03 / (FAMILIES.length - 1)));
      const h0 = zeros(HIDDEN);
      for (let j = 0; j < HIDDEN; j++) {
        let s = p.b0[j];
        for (let i = 0; i < WIDTH; i++) s += row.x[i] * p.w0[i][j];
        h0[j] = tanh(s);
      }
      const pre1 = zeros(HIDDEN);
      const h1 = zeros(HIDDEN);
      for (let j = 0; j < HIDDEN; j++) {
        let s = p.b1[j];
        for (let i = 0; i < HIDDEN; i++) s += h0[i] * p.w1[i][j];
        pre1[j] = s;
        h1[j] = tanh(s) + h0[j];
      }
      const dLog = prob.map((v, k) => v - target[k]);
      const dH1 = zeros(HIDDEN);
      for (let j = 0; j < HIDDEN; j++) {
        for (let k = 0; k < FAMILIES.length; k++) dH1[j] += dLog[k] * p.w2[j][k];
      }
      const dPre = pre1.map((s, j) => dH1[j] * (1 - Math.tanh(s) * Math.tanh(s)));
      const dH0 = zeros(HIDDEN);
      for (let i = 0; i < HIDDEN; i++) {
        let s = dH1[i];
        for (let j = 0; j < HIDDEN; j++) s += dPre[j] * p.w1[i][j];
        dH0[i] = s;
      }
      const dPre0 = h0.map((h, j) => {
        const raw = Math.atanh(Math.max(-0.999, Math.min(0.999, h)));
        return dH0[j] * (1 - Math.tanh(raw) * Math.tanh(raw));
      });
      for (let k = 0; k < FAMILIES.length; k++) {
        p.b2[k] -= rate * (dLog[k] + 0.0001 * p.b2[k]);
        for (let j = 0; j < HIDDEN; j++) p.w2[j][k] -= rate * (dLog[k] * h1[j] + 0.0001 * p.w2[j][k]);
      }
      for (let j = 0; j < HIDDEN; j++) {
        p.b1[j] -= rate * (dPre[j] + 0.0001 * p.b1[j]);
        for (let i = 0; i < HIDDEN; i++) p.w1[i][j] -= rate * (dPre[j] * h0[i] + 0.0001 * p.w1[i][j]);
      }
      for (let j = 0; j < HIDDEN; j++) {
        p.b0[j] -= rate * (dPre0[j] + 0.0001 * p.b0[j]);
        for (let i = 0; i < WIDTH; i++) p.w0[i][j] -= rate * (dPre0[j] * row.x[i] + 0.0001 * p.w0[i][j]);
      }
    }
  }
  return p;
}

const NET = train();

export function scoreGlance(glance) {
  const logits = forward(NET, features(glance));
  const probs = softmax(logits);
  const scores = {};
  let best = 0;
  for (let i = 0; i < FAMILIES.length; i++) {
    scores[FAMILIES[i]] = Math.round(probs[i] * 1000) / 1000;
    if (probs[i] > probs[best]) best = i;
  }
  return { family: FAMILIES[best], confidence: scores[FAMILIES[best]], scores };
}

// A worse guess does not replace the one you are already holding.
export function promote(held, next) {
  if (!next || !next.family) return { family: "", confidence: 0, scores: {}, promoted: false, reason: "nothing to rank" };
  if (!held || !held.family) return { ...next, promoted: true, reason: "first look" };
  if (held.family === next.family) return { ...next, promoted: false, reason: "same family" };
  if (Number(next.confidence) + 1e-9 < Number(held.confidence)) {
    return { ...held, promoted: false, reason: "worse candidate kept" };
  }
  return { ...next, promoted: true, reason: "stronger family" };
}

export function heldOutAccuracy() {
  const rng = mulberry32(SEED + 1);
  let hit = 0;
  let n = 0;
  for (let i = 0; i < 20; i++) {
    for (const family of FAMILIES) {
      n++;
      if (scoreGlance(sampleGlance(family, rng)).family === family) hit++;
    }
  }
  return { hit, n };
}

export { FAMILIES };
