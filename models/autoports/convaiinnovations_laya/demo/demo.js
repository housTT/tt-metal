(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const BATCH_STATES = 8;
  const HISTORY = 60;
  const MAX_ROWS = 300;
  const app = {
    health: null,
    online: false,
    presets: [],
    feed: null,
    feedSource: "server",
    gold: null,
    history: { client: [], server: [], device: [], input_tokens: [], state_tokens: [] },
    last: null,
    run: null,
    lastStats: null,
  };

  function esc(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
  function fmt(x, d = 1) {
    if (x === null || x === undefined || Number.isNaN(x)) return "n/a";
    return Number(x).toFixed(d);
  }
  function pct(p) {
    return (100 * p).toFixed(1) + "%";
  }
  function percentile(arr, p) {
    if (!arr.length) return null;
    const s = arr.slice().sort((a, b) => a - b);
    const i = Math.min(s.length - 1, Math.max(0, Math.ceil((p / 100) * s.length) - 1));
    return s[i];
  }
  function summary(arr) {
    if (!arr.length) return { n: 0, p50: null, p95: null, mean: null, max: null };
    return {
      n: arr.length,
      p50: percentile(arr, 50),
      p95: percentile(arr, 95),
      mean: arr.reduce((a, b) => a + b, 0) / arr.length,
      max: Math.max(...arr),
    };
  }
  function headerNum(headers, name) {
    const v = headers.get(name);
    if (v === null || v === "") return null;
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  }

  async function post(path, body) {
    const t0 = performance.now();
    const res = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const clientMs = performance.now() - t0;
    let json = null;
    try {
      json = await res.json();
    } catch (e) {
      json = { detail: "response is not JSON" };
    }
    return {
      ok: res.ok,
      status: res.status,
      json,
      clientMs,
      serverMs: headerNum(res.headers, "X-Inference-Time-Ms"),
      deviceMs: headerNum(res.headers, "X-Laya-Device-Ms"),
      batch: res.headers.get("X-Laya-Batch"),
    };
  }

  async function pollHealth() {
    const pill = $("health");
    try {
      const res = await fetch("/v1/health", { cache: "no-store" });
      const h = await res.json();
      app.health = h;
      app.online = true;
      $("offline-banner").hidden = true;
      if (res.status === 503 || !h.ready) {
        pill.className = "pill pill-wait";
        $("health-text").textContent = "loading";
        $("health-detail").textContent = "";
        return;
      }
      pill.className = "pill pill-ok";
      $("health-text").textContent = (h.backend || "?") + " ready";
      const parts = [];
      if (h.precision) parts.push(h.precision);
      if (h.mesh_shape) parts.push("mesh " + h.mesh_shape);
      if (h.seq_buckets) parts.push("seq " + h.seq_buckets.join("/"));
      if (h.row_buckets) parts.push("rows " + h.row_buckets[0] + ".." + h.row_buckets[h.row_buckets.length - 1]);
      if (h.warm_shapes && h.warm_shapes.length) parts.push(h.warm_shapes.length + " traces");
      $("health-detail").textContent = parts.join(" | ");
      $("cpu-banner").hidden = h.backend !== "cpu";
    } catch (e) {
      app.online = false;
      pill.className = "pill pill-bad";
      $("health-text").textContent = "offline";
      $("health-detail").textContent = "";
      $("offline-banner").hidden = false;
    }
  }

  async function loadPresets() {
    const res = await fetch("/demo/presets.json", { cache: "no-store" });
    app.presets = await res.json();
    const box = $("presets");
    box.innerHTML = "";
    for (const p of app.presets) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "chip";
      b.textContent = p.title;
      b.dataset.id = p.id;
      b.addEventListener("click", () => applyPreset(p.id));
      box.appendChild(b);
    }
  }

  function applyPreset(id) {
    const p = app.presets.find((x) => x.id === id);
    if (!p) return;
    for (const c of $("presets").children) c.classList.toggle("active", c.dataset.id === id);
    $("state").value = typeof p.state === "string" ? p.state : JSON.stringify(p.state, null, 1);
    $("questions").value = JSON.stringify(p.questions, null, 1);
    $("preset-blurb").textContent = (p.blurb || "") + (p.source ? " Source: " + p.source : "");
    app.gold = p.gold || null;
    $("decide-error").hidden = true;
  }

  function parseState(text) {
    const t = text.trim();
    if (!t) throw new Error("state is empty");
    try {
      return JSON.parse(t);
    } catch (e) {
      return text;
    }
  }
  function parseQuestions(text) {
    let q;
    try {
      q = JSON.parse(text);
    } catch (e) {
      throw new Error("questions is not valid JSON: " + e.message);
    }
    if (!q || typeof q !== "object" || Array.isArray(q) || !Object.keys(q).length) throw new Error("questions must be a JSON object with at least one question");
    return q;
  }
  function optionalNumber(id) {
    const v = $(id).value.trim();
    if (v === "") return null;
    const n = Number(v);
    if (!Number.isFinite(n)) throw new Error(id + " must be a number");
    return n;
  }

  async function decide() {
    const err = $("decide-error");
    err.hidden = true;
    let body;
    try {
      body = { state: parseState($("state").value), questions: parseQuestions($("questions").value) };
      const ml = optionalNumber("max_len");
      const hml = optionalNumber("head_max_len");
      const mc = optionalNumber("min_confidence");
      if (ml !== null) body.max_len = Math.trunc(ml);
      if (hml !== null) body.head_max_len = Math.trunc(hml);
      if (mc !== null) body.min_confidence = mc;
    } catch (e) {
      err.textContent = e.message;
      err.hidden = false;
      return;
    }
    $("decide").disabled = true;
    try {
      const r = await post("/v1/systemone", body);
      if (!r.ok) {
        err.textContent = "HTTP " + r.status + ": " + (typeof r.json.detail === "string" ? r.json.detail : JSON.stringify(r.json.detail || r.json));
        err.hidden = false;
        return;
      }
      app.last = r;
      renderCards(r.json.answers, body.questions, app.gold, r.json.usage);
      $("results-meta").textContent = "client " + fmt(r.clientMs) + " ms | server " + fmt(r.serverMs) + " ms | device " + fmt(r.deviceMs) + " ms | batch " + (r.batch || "n/a");
      pushTile(r);
    } catch (e) {
      err.textContent = "request failed: " + e.message;
      err.hidden = false;
    } finally {
      $("decide").disabled = false;
    }
  }

  function barRow(label, p, top, goldP) {
    const tick = goldP === null || goldP === undefined ? "" : `<span class="tick" style="left:${(100 * goldP).toFixed(2)}%" title="gold ${pct(goldP)}"></span>`;
    return `<div class="bar${top ? " top" : ""}"><span class="lbl" title="${esc(label)}">${esc(label)}</span><span class="track"><span class="fill" style="width:${(100 * p).toFixed(2)}%"></span>${tick}</span><span class="pct">${pct(p)}</span></div>`;
  }

  function cardHtml(qid, q, a, gold) {
    const type = a.type || q.type;
    let body = "";
    const goldProbs = gold && gold.probabilities ? gold.probabilities : null;
    if (type === "choice") {
      const probs = a.probabilities || {};
      body = '<div class="bars">' + Object.keys(probs).map((k) => barRow(k, probs[k], k === a.choice, goldProbs ? goldProbs[k] : null)).join("") + "</div>";
    } else if (type === "score") {
      const probs = a.probabilities || {};
      const keys = Object.keys(probs);
      const k = keys.length;
      const top = keys.reduce((b, x) => (probs[x] > probs[b] ? x : b), keys[0]);
      body = '<div class="bars">' + keys.map((i) => barRow(i + ": " + (a.legend ? a.legend[i] : ""), probs[i], i === top, goldProbs ? goldProbs[i] : null)).join("") + "</div>";
      const pos = k > 1 ? (100 * a.score) / (k - 1) : 0;
      const gmark = gold && typeof gold.score === "number" && k > 1 ? `<span class="gmark" style="left:${((100 * gold.score) / (k - 1)).toFixed(2)}%" title="gold expected score ${fmt(gold.score, 3)}"></span>` : "";
      body += `<div class="ruler"><span class="mark" style="left:${pos.toFixed(2)}%" title="expected score ${fmt(a.score, 3)}"></span>${gmark}<div class="ends"><span>0</span><span>expected score ${fmt(a.score, 3)}</span><span>${k - 1}</span></div></div>`;
    } else {
      const pt = a.noul;
      const gtick = gold && typeof gold.noul === "number" ? `<span class="tick" style="left:${(100 * gold.noul).toFixed(2)}%" title="gold P(true) ${pct(gold.noul)}"></span>` : "";
      body = `<div class="noul"><span class="f" style="width:${(100 * (1 - pt)).toFixed(2)}%"></span><span class="t" style="width:${(100 * pt).toFixed(2)}%"></span><span class="lab l">false ${pct(1 - pt)}</span><span class="lab r">true ${pct(pt)}</span>${gtick}</div>`;
    }
    const act = a.action && typeof a.action.act_probability === "number" ? `<span class="act" title="action.act_probability as returned by the API; see the caveat below">act ${fmt(a.action.act_probability, 3)} / escalate ${fmt(1 - a.action.act_probability, 3)}</span>` : "";
    const abst = a.abstention ? `<span class="abst">${esc(a.abstention)} at ${fmt(a.abstention_threshold, 2)}</span>` : "";
    const goldLine = gold && gold.label !== undefined ? `<span>gold <b>${esc(gold.label)}</b></span>` : "";
    const headline = type === "choice" ? `<b>${esc(a.choice)}</b>` : type === "score" ? `<b>score ${fmt(a.score, 3)}</b>` : `<b>${pt(a) }</b>`;
    return `<div class="acard"><div class="acard-head"><span class="qid">${esc(qid)}</span><span class="badge badge-${esc(type)}">${esc(type)}</span>${headline}${goldLine}</div><div class="ins">${esc(q.instructions)}</div>${body}<div class="meta"><span>confidence <b>${fmt(a.confidence, 3)}</b></span><span>answer_confidence <b>${fmt(a.answer_confidence, 3)}</b></span>${act}${abst}</div></div>`;
    function pt(x) {
      return x.noul >= 0.5 ? "true " + pct(x.noul) : "false " + pct(1 - x.noul);
    }
  }

  function renderCards(answers, questions, gold, usage) {
    const box = $("cards");
    const parts = [];
    for (const qid of Object.keys(questions)) {
      const a = answers[qid];
      if (!a) continue;
      parts.push(cardHtml(qid, questions[qid], a, gold ? gold[qid] : null));
    }
    if (usage) {
      const u = usage;
      const trunc = u.truncated ? `truncated: ${u.state_tokens_dropped} state tokens dropped` : "state fits";
      parts.push(`<div class="meta"><span>input tokens <b>${u.input_tokens}</b></span><span>state tokens <b>${u.state_tokens}</b></span><span>${trunc}</span>${u.truncated_questions && u.truncated_questions.length ? "<span>questions truncated: " + esc(u.truncated_questions.join(", ")) + "</span>" : ""}${u.options ? "<span>collapsed options: " + esc(JSON.stringify(u.options)) + "</span>" : ""}</div>`);
    }
    box.innerHTML = parts.join("") || '<p class="muted">No answers.</p>';
  }

  const TILES = [
    { key: "client", label: "client ms", get: (r) => r.clientMs },
    { key: "server", label: "server ms", get: (r) => r.serverMs },
    { key: "device", label: "device ms", get: (r) => r.deviceMs },
    { key: "batch", label: "batch", text: (r) => r.batch || "n/a" },
    { key: "input_tokens", label: "input tokens", get: (r) => (r.json.usage ? r.json.usage.input_tokens : null) },
    { key: "state_tokens", label: "state tokens", get: (r) => (r.json.usage ? r.json.usage.state_tokens : null) },
    { key: "truncated", label: "truncated", text: (r) => (r.json.usage && r.json.usage.truncated ? "yes" : "no") },
  ];

  function sparkline(values) {
    const v = values.slice(-HISTORY).filter((x) => typeof x === "number" && Number.isFinite(x));
    if (v.length < 2) return "<svg viewBox=\"0 0 100 28\"></svg>";
    const lo = Math.min(...v);
    const hi = Math.max(...v);
    const span = hi - lo || 1;
    const pts = v.map((x, i) => `${((100 * i) / (v.length - 1)).toFixed(2)},${(26 - (24 * (x - lo)) / span).toFixed(2)}`).join(" ");
    return `<svg viewBox="0 0 100 28" preserveAspectRatio="none"><polyline points="${pts}"></polyline></svg>`;
  }

  function pushTile(r) {
    for (const t of TILES) {
      if (t.get && app.history[t.key]) {
        const v = t.get(r);
        if (typeof v === "number") {
          app.history[t.key].push(v);
          if (app.history[t.key].length > HISTORY) app.history[t.key].shift();
        }
      }
    }
    app.last = r;
    renderTiles();
  }

  function renderTiles() {
    const r = app.last;
    $("tiles").innerHTML = TILES.map((t) => {
      let v = "n/a";
      if (r) v = t.text ? t.text(r) : t.get(r) === null || t.get(r) === undefined ? "n/a" : typeof t.get(r) === "number" && !Number.isInteger(t.get(r)) ? fmt(t.get(r)) : String(t.get(r));
      const spark = app.history[t.key] ? sparkline(app.history[t.key]) : "";
      return `<div class="tile"><div class="k">${esc(t.label)}</div><div class="v">${esc(v)}</div>${spark}</div>`;
    }).join("");
  }

  async function loadFeed() {
    const res = await fetch("/demo/feed.json", { cache: "no-store" });
    const feed = await res.json();
    setFeed(feed, "server");
  }

  function setFeed(feed, source) {
    app.feed = feed;
    app.feedSource = source;
    const a = feed.attribution || {};
    $("feed-source").textContent = `${feed.cases.length} cases, ${feed.cases.reduce((n, c) => n + Object.keys(c.questions).length, 0)} decisions (${source})`;
    $("feed-attribution").textContent = a.dataset ? `Feed: ${a.dataset} (${a.config || ""} ${a.split || ""}, revision ${a.revision || "?"}, ${a.license || "licence unknown"}). ${a.selection || ""}` : "Feed: uploaded JSONL.";
  }

  function argmaxOf(a) {
    if (a.type === "choice") return String(a.choice);
    if (a.type === "score") {
      const p = a.probabilities || {};
      return Object.keys(p).reduce((b, k) => (p[k] > p[b] ? k : b), Object.keys(p)[0]);
    }
    return a.noul >= 0.5 ? "true" : "false";
  }
  function goldLabelOf(g, type) {
    if (!g || g.label === undefined || g.label === null) return null;
    return type === "noul" ? String(g.label).toLowerCase() : String(g.label);
  }

  class FeedRun {
    constructor(cfg) {
      this.cfg = cfg;
      this.cases = app.feed.cases;
      this.stopping = false;
      this.inFlight = 0;
      this.index = 0;
      this.startedAt = Date.now();
      this.endedAt = null;
      this.timer = null;
      this.waiters = [];
      this.stats = { requests: 0, cases: 0, decisions: 0, errors: 0, agree: 0, byType: {}, client: [], server: [], device: [], batchHist: {}, rows: [] };
      this.units = this.cfg.batch ? this.groupBatches() : this.cases.map((c) => [c]);
    }
    groupBatches() {
      const units = [];
      let cur = [];
      for (const c of this.cases) {
        const key = JSON.stringify(c.questions);
        if (cur.length && (cur.length >= BATCH_STATES || cur[0].key !== key)) {
          units.push(cur.map((x) => x.c));
          cur = [];
        }
        cur.push({ c, key });
      }
      if (cur.length) units.push(cur.map((x) => x.c));
      return units;
    }
    start() {
      const perUnit = this.cfg.batch ? BATCH_STATES : 1;
      const interval = (1000 * perUnit) / this.cfg.rate;
      let nextAt = performance.now();
      const tick = async () => {
        if (this.stopping) return;
        if (this.index >= this.units.length) {
          if (this.cfg.loop) this.index = 0;
          else return this.finish();
        }
        while (this.inFlight >= this.cfg.concurrency && !this.stopping) await new Promise((r) => this.waiters.push(r));
        if (this.stopping) return;
        const unit = this.units[this.index++];
        this.dispatch(unit);
        nextAt += interval;
        const delay = Math.max(0, nextAt - performance.now());
        this.timer = setTimeout(tick, delay);
      };
      tick();
      if (this.cfg.seconds > 0) this.deadline = setTimeout(() => this.stop(), this.cfg.seconds * 1000);
      renderStats(this);
      this.statsTimer = setInterval(() => renderStats(this), 500);
    }
    release() {
      this.inFlight--;
      const w = this.waiters.shift();
      if (w) w();
      if (this.stopping && this.inFlight <= 0) this.finish();
      else if (!this.cfg.loop && this.index >= this.units.length && this.inFlight <= 0) this.finish();
    }
    async dispatch(unit) {
      this.inFlight++;
      try {
        if (unit.length === 1 && !this.cfg.batch) {
          const c = unit[0];
          const r = await post("/v1/systemone", { state: c.state, questions: c.questions });
          this.record(r, [c], r.ok ? [r.json] : null);
        } else {
          const r = await post("/v1/systemone/batch", { states: unit.map((c) => c.state), questions: unit[0].questions });
          this.record(r, unit, r.ok ? r.json.results : null);
        }
      } catch (e) {
        this.record({ ok: false, status: 0, json: { detail: e.message }, clientMs: null, serverMs: null, deviceMs: null, batch: null }, unit, null);
      } finally {
        this.release();
      }
    }
    record(r, cases, results) {
      const s = this.stats;
      s.requests++;
      if (typeof r.clientMs === "number") s.client.push(r.clientMs);
      if (typeof r.serverMs === "number") s.server.push(r.serverMs);
      if (typeof r.deviceMs === "number") s.device.push(r.deviceMs);
      if (r.batch) for (const b of r.batch.split(",")) s.batchHist[b] = (s.batchHist[b] || 0) + 1;
      if (r.ok) pushTile(Object.assign({}, r, { json: results[0] }));
      cases.forEach((c, i) => {
        const row = { id: c.id, workflow: c.workflow, answers: {}, gold: {}, agree: 0, n: 0, client_ms: r.clientMs, server_ms: r.serverMs, device_ms: r.deviceMs, batch: r.batch, error: null };
        if (!r.ok || !results || !results[i]) {
          s.errors++;
          row.error = "HTTP " + r.status + " " + (typeof r.json.detail === "string" ? r.json.detail : JSON.stringify(r.json.detail || r.json)).slice(0, 160);
        } else {
          s.cases++;
          const answers = results[i].answers || {};
          for (const qid of Object.keys(c.questions)) {
            const a = answers[qid];
            if (!a) continue;
            const type = a.type || c.questions[qid].type;
            const got = argmaxOf(a);
            const gold = goldLabelOf(c.gold ? c.gold[qid] : null, type);
            row.answers[qid] = got;
            row.gold[qid] = gold;
            row.n++;
            s.decisions++;
            const t = (s.byType[type] = s.byType[type] || { n: 0, agree: 0 });
            if (gold !== null) {
              t.n++;
              if (got === gold) {
                t.agree++;
                s.agree++;
                row.agree++;
              }
            }
          }
        }
        s.rows.push(row);
        if (s.rows.length > MAX_ROWS) s.rows.shift();
        appendRow(row, c);
      });
    }
    stop() {
      if (this.stopping) return;
      this.stopping = true;
      clearTimeout(this.timer);
      clearTimeout(this.deadline);
      for (const w of this.waiters.splice(0)) w();
      if (this.inFlight <= 0) this.finish();
    }
    finish() {
      if (this.endedAt) return;
      this.endedAt = Date.now();
      clearTimeout(this.timer);
      clearTimeout(this.deadline);
      clearInterval(this.statsTimer);
      renderStats(this);
      app.lastStats = statsJson(this);
      $("feed-start").disabled = false;
      $("feed-stop").disabled = true;
      app.run = null;
    }
  }

  function appendRow(row, c) {
    const tbody = $("feed-table").querySelector("tbody");
    const tr = document.createElement("tr");
    const qids = Object.keys(c.questions);
    let cells = `<td>${esc(row.id)}</td><td>${esc(row.workflow || "")}</td>`;
    if (row.error) {
      cells += `<td class="err" colspan="5">${esc(row.error)}</td>`;
    } else {
      for (let i = 0; i < 5; i++) {
        const qid = qids[i];
        if (!qid) {
          cells += "<td></td>";
          continue;
        }
        const got = row.answers[qid];
        const gold = row.gold[qid];
        const cls = gold === null || gold === undefined ? "na" : got === gold ? "ok" : "bad";
        cells += `<td class="ans ${cls}" title="${esc(qid)}: got ${esc(got)}${gold !== null && gold !== undefined ? ", gold " + esc(gold) : ""}">${esc(qid)}=${esc(got === undefined ? "?" : got)}</td>`;
      }
    }
    cells += `<td>${fmt(row.device_ms)}</td><td>${esc(row.batch || "")}</td>`;
    tr.innerHTML = cells;
    tbody.appendChild(tr);
    while (tbody.children.length > MAX_ROWS) tbody.removeChild(tbody.firstChild);
    const wrap = $("feed-table").parentElement;
    wrap.scrollTop = wrap.scrollHeight;
  }

  function statsJson(run) {
    const s = run.stats;
    const wall = ((run.endedAt || Date.now()) - run.startedAt) / 1000;
    const h = app.health || {};
    const byType = {};
    for (const [k, v] of Object.entries(s.byType)) byType[k] = { n: v.n, agree: v.agree, rate: v.n ? v.agree / v.n : null };
    const scored = Object.values(s.byType).reduce((n, v) => n + v.n, 0);
    return {
      schema: "laya-demo-feed-stats/1",
      source: "demo-page",
      page_url: location.href,
      server: { backend: h.backend || null, precision: h.precision || null, mesh_shape: h.mesh_shape || null, seq_buckets: h.seq_buckets || null, row_buckets: h.row_buckets || null, model: h.model || null, revision: h.revision || null },
      config: { rate: run.cfg.rate, concurrency: run.cfg.concurrency, loop: run.cfg.loop, batch: run.cfg.batch, batch_states: BATCH_STATES, seconds: run.cfg.seconds },
      feed: { source: app.feedSource, dataset: app.feed.attribution ? app.feed.attribution.dataset : null, revision: app.feed.attribution ? app.feed.attribution.revision : null, cases_available: app.feed.cases.length },
      started_at: new Date(run.startedAt).toISOString(),
      ended_at: run.endedAt ? new Date(run.endedAt).toISOString() : null,
      wall_s: Math.round(wall * 100) / 100,
      requests: s.requests,
      cases: s.cases,
      decisions: s.decisions,
      errors: s.errors,
      requests_per_s: wall > 0 ? s.requests / wall : null,
      cases_per_s: wall > 0 ? s.cases / wall : null,
      decisions_per_s: wall > 0 ? s.decisions / wall : null,
      latency_unit: "per request (a batch request covers up to " + BATCH_STATES + " cases)",
      client_ms: summary(s.client),
      server_ms: summary(s.server),
      device_ms: summary(s.device),
      agreement: { decisions: scored, agree: s.agree, rate: scored ? s.agree / scored : null, by_type: byType, rule: "choice: choice == gold.label; score: argmax(probabilities) == gold.label; noul: (noul >= 0.5) == gold.label" },
      batch_shapes: Object.assign({}, s.batchHist),
      rows: s.rows.slice(),
    };
  }

  function renderStats(run) {
    const j = statsJson(run);
    const items = [
      ["decisions/s", fmt(j.decisions_per_s, 2)],
      ["cases/s", fmt(j.cases_per_s, 2)],
      ["cases", String(j.cases)],
      ["decisions", String(j.decisions)],
      ["client p50 / p95 ms", fmt(j.client_ms.p50) + " / " + fmt(j.client_ms.p95)],
      ["server p50 / p95 ms", fmt(j.server_ms.p50) + " / " + fmt(j.server_ms.p95)],
      ["device p50 ms", fmt(j.device_ms.p50)],
      ["agreement with gold", j.agreement.rate === null ? "n/a" : pct(j.agreement.rate) + " (" + j.agreement.agree + "/" + j.agreement.decisions + ")"],
      ["errors", String(j.errors)],
      ["in flight", String(run.inFlight)],
      ["batch shapes", Object.entries(j.batch_shapes).map(([k, v]) => k + " x" + v).join(", ") || "n/a"],
      ["wall s", fmt(j.wall_s)],
    ];
    $("feed-stats").innerHTML = items.map(([k, v]) => `<div class="stat"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div></div>`).join("");
  }

  function readFeedConfig() {
    const rate = Math.min(20, Math.max(0.5, Number($("rate").value) || 4));
    const concurrency = Math.min(4, Math.max(1, Math.trunc(Number($("concurrency").value) || 1)));
    const seconds = Math.max(0, Number($("seconds").value) || 0);
    $("rate").value = rate;
    $("concurrency").value = concurrency;
    return { rate, concurrency, loop: $("loop").checked, batch: $("batch").checked, seconds };
  }

  function startFeed() {
    const err = $("feed-error");
    err.hidden = true;
    if (!app.feed || !app.feed.cases.length) {
      err.textContent = "no feed cases loaded";
      err.hidden = false;
      return;
    }
    if (app.run) return;
    $("feed-table").querySelector("tbody").innerHTML = "";
    app.run = new FeedRun(readFeedConfig());
    $("feed-start").disabled = true;
    $("feed-stop").disabled = false;
    app.run.start();
  }

  function stopFeed() {
    if (app.run) app.run.stop();
  }

  async function copyStats() {
    const j = app.run ? statsJson(app.run) : app.lastStats;
    const err = $("feed-error");
    if (!j) {
      err.textContent = "no feed run yet";
      err.hidden = false;
      return;
    }
    const text = JSON.stringify(j, null, 1);
    try {
      await navigator.clipboard.writeText(text);
      $("copy-stats").textContent = "Copied";
    } catch (e) {
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      try {
        document.execCommand("copy");
        $("copy-stats").textContent = "Copied";
      } catch (e2) {
        err.textContent = "clipboard unavailable; stats JSON printed to the console";
        err.hidden = false;
        console.log(text);
      }
      document.body.removeChild(ta);
    }
    setTimeout(() => ($("copy-stats").textContent = "Copy stats JSON"), 1500);
  }

  function loadJsonl(file) {
    const err = $("feed-error");
    err.hidden = true;
    const reader = new FileReader();
    reader.onload = () => {
      const cases = [];
      const lines = String(reader.result).split(/\r?\n/);
      try {
        for (let i = 0; i < lines.length; i++) {
          const line = lines[i].trim();
          if (!line) continue;
          const c = JSON.parse(line);
          if (!("state" in c) || !c.questions || typeof c.questions !== "object") throw new Error("line " + (i + 1) + " needs state and questions");
          cases.push({ id: c.id || "line" + (i + 1), workflow: c.workflow || "", state: c.state, questions: c.questions, gold: c.gold || null });
        }
      } catch (e) {
        err.textContent = "JSONL rejected: " + e.message;
        err.hidden = false;
        return;
      }
      if (!cases.length) {
        err.textContent = "JSONL has no cases";
        err.hidden = false;
        return;
      }
      setFeed({ attribution: { dataset: null }, cases }, "upload " + file.name);
    };
    reader.readAsText(file);
  }

  async function autorun() {
    const qp = new URLSearchParams(location.search);
    const modes = (qp.get("autorun") || "").split(",").map((s) => s.trim()).filter(Boolean);
    if (qp.get("rate")) $("rate").value = qp.get("rate");
    if (qp.get("seconds")) $("seconds").value = qp.get("seconds");
    if (qp.get("concurrency")) $("concurrency").value = qp.get("concurrency");
    if (qp.get("batch") === "1") $("batch").checked = true;
    if (qp.get("loop") === "1") $("loop").checked = true;
    if (qp.get("preset")) applyPreset(qp.get("preset"));
    if (modes.includes("decide")) await decide();
    if (modes.includes("feed")) startFeed();
  }

  async function init() {
    $("decide").addEventListener("click", decide);
    $("feed-start").addEventListener("click", startFeed);
    $("feed-stop").addEventListener("click", stopFeed);
    $("copy-stats").addEventListener("click", copyStats);
    $("feed-file").addEventListener("change", (e) => e.target.files[0] && loadJsonl(e.target.files[0]));
    renderTiles();
    await pollHealth();
    setInterval(pollHealth, 5000);
    try {
      await loadPresets();
      applyPreset(app.presets[0].id);
    } catch (e) {
      $("decide-error").textContent = "presets failed to load: " + e.message;
      $("decide-error").hidden = false;
    }
    try {
      await loadFeed();
    } catch (e) {
      $("feed-error").textContent = "feed failed to load: " + e.message;
      $("feed-error").hidden = false;
    }
    await autorun();
  }

  window.layaDemo = { app, statsJson: () => (app.run ? statsJson(app.run) : app.lastStats) };
  document.addEventListener("DOMContentLoaded", init);
})();
