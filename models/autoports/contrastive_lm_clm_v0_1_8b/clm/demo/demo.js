(() => {
  'use strict';
  const $ = (s) => document.querySelector(s);
  const W = 600, H = 150, HORIZON_Y = 127;
  const DAY = { fg: [83, 83, 83], bg: [247, 247, 247] };
  const NIGHT = { fg: [172, 172, 172], bg: [0, 0, 0] };
  const CACTUS = {
    0: { w: 17, h: 35, boxes: [[0, 7, 5, 27], [4, 0, 6, 34], [10, 4, 7, 14]] },
    1: { w: 25, h: 50, boxes: [[0, 12, 7, 38], [8, 0, 7, 49], [13, 10, 10, 38]] },
  };
  const BIRD = { w: 46, h: 40, boxes: [[15, 15, 16, 5], [18, 21, 24, 6], [2, 14, 4, 3], [6, 10, 4, 7], [10, 8, 6, 9]] };
  const TREX_BOXES = [[22, 0, 17, 16], [1, 18, 30, 9], [10, 35, 14, 8], [1, 24, 29, 5], [5, 30, 21, 4], [9, 34, 15, 4]];
  const DUCK_BOX = [1, 18, 55, 25];
  const MOON_OFFSET = [-16, -9, -4, 0, 4, 9, 16];
  const ACTIONS = ['jump', 'duck', 'run'];
  const LS = { key: 'clm.playground.apiKey', theme: 'clm.playground.theme' };
  const SUN = '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>';
  const MOON = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>';
  const S = {
    frame: null, decision: null, stats: null, course: null, rows: [], reference: null, hello: null,
    running: false, phase: 'idle', events: [], series: { lat: [], inf: [], srv: [] }, overlay: false,
    lastSeq: 0, dropFlashUntil: 0, flashText: '',
  };
  let ws = null, backoff = 500, scale = 2;

  function setStatus(cls, text) {
    const b = $('#status');
    b.className = 'status ' + cls;
    $('#status-text').textContent = text;
  }

  function applyTheme(t) {
    document.documentElement.dataset.theme = t;
    const b = $('#btn-theme');
    b.querySelector('svg').innerHTML = t === 'dark' ? SUN : MOON;
    b.title = t === 'dark' ? 'Switch to light' : 'Switch to dark';
  }

  function wsUrl() {
    const u = new URL('ws', location.href);
    u.protocol = u.protocol === 'https:' ? 'wss:' : 'ws:';
    return u.href;
  }

  function connect() {
    setStatus('busy', 'connecting');
    ws = new WebSocket(wsUrl());
    ws.onopen = () => { backoff = 500; setStatus('ok', 'stream connected'); };
    ws.onmessage = (e) => { try { handle(JSON.parse(e.data)); } catch (err) { console.error(err); } };
    ws.onclose = () => {
      setStatus('bad', 'stream lost, reconnecting');
      setTimeout(connect, backoff);
      backoff = Math.min(8000, backoff * 2);
    };
    ws.onerror = () => { try { ws.close(); } catch (e) { } };
  }

  function send(obj) {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
  }

  function setPhase(text, cls) {
    const el = $('#phase');
    el.textContent = text;
    el.className = 'phase' + (cls ? ' ' + cls : '');
  }

  function setRunning(on) {
    S.running = on;
    for (const el of $('#controls').querySelectorAll('input')) el.disabled = on;
    $('#btn-start').disabled = on;
    $('#btn-stop').disabled = !on;
  }

  function handle(m) {
    switch (m.type) {
      case 'hello':
        S.hello = m;
        setRunning(m.running);
        if (m.running) {
          $('#ctl-note').textContent = 'A run started at ' + (m.started_at || '').replace('T', ' ').replace('+00:00', ' UTC') + ' is in progress; you are watching it and may stop it.';
          setPhase('joined a running game', 'live');
        } else if (!S.rows.length) {
          setPhase('idle');
        }
        if (m.api_key_required) {
          $('#f-key-wrap').hidden = false;
          try { $('#f-key').value = localStorage.getItem(LS.key) || ''; } catch (e) { }
        }
        break;
      case 'status':
        S.phase = m.phase;
        setPhase(phaseText(m), m.phase === 'error' ? 'err' : (m.phase === 'playing' ? 'live' : ''));
        if (m.phase === 'finished' || m.phase === 'stopped' || m.phase === 'error') setRunning(false);
        break;
      case 'course':
        S.course = m;
        $('#course-line').textContent = 'course ' + (m.index + 1) + ', seed ' + m.seed + ', ' + m.duration + ' s, ' + m.endpoint.replace(/ · /g, ', ') + ', warm answer ' + Math.min(...m.warm_ms).toFixed(0) + ' ms';
        setRunning(true);
        break;
      case 'frame':
        S.frame = m;
        for (const d of m.d) onDecision(d);
        for (const e of m.e) pushEvent(e.frame, e.event);
        break;
      case 'stats':
        S.stats = m;
        renderStats(m);
        break;
      case 'course_end':
        S.rows.push(m.row);
        renderRows();
        break;
      case 'summary':
        S.rows = m.rows || S.rows;
        S.summary = m.summary;
        renderRows();
        setRunning(false);
        setPhase('finished: ' + (m.summary ? m.summary.survived + ' of ' + m.summary.seeds + ' courses survived' : 'no course completed'), '');
        $('#ctl-note').textContent = '';
        break;
      case 'stopped':
        setRunning(false);
        setPhase('stopped');
        $('#ctl-note').textContent = '';
        break;
      case 'busy':
        $('#ctl-note').textContent = 'Another viewer started a run at ' + (m.started_at || '').replace('T', ' ') + '; watching it.';
        setRunning(true);
        break;
      case 'error':
        setPhase('error: ' + m.error, 'err');
        $('#ctl-note').textContent = m.error;
        setRunning(false);
        break;
      default:
        break;
    }
  }

  function phaseText(m) {
    const course = 'course ' + (m.course + 1) + ' of ' + m.seeds;
    if (m.phase === 'starting') return course + ': starting the player';
    if (m.phase === 'warming') return course + ': measuring latency';
    if (m.phase === 'playing') return course + ': playing' + (m.detail ? ', ' + m.detail : '');
    if (m.phase === 'between') return 'next course in a moment';
    if (m.phase === 'finished') return 'finished';
    if (m.phase === 'error') return 'error: ' + m.detail;
    return m.phase;
  }

  function cleanEvent(text) {
    return text.replace(/ — /g, ': ');
  }

  function pushEvent(frame, text) {
    S.events.unshift({ frame, text: cleanEvent(text) });
    S.events.length = Math.min(S.events.length, 8);
    const ul = $('#events');
    ul.innerHTML = '';
    for (const e of S.events) {
      const li = document.createElement('li');
      const f = document.createElement('span');
      f.className = 'f';
      f.textContent = (e.frame / 60).toFixed(1) + ' s';
      const t = document.createElement('span');
      t.textContent = e.text;
      li.append(f, t);
      ul.appendChild(li);
    }
  }

  function onDecision(d) {
    if (d.dropped || d.error) {
      S.flashText = d.error ? 'request failed: ' + d.error : d.dropped + ' (answer #' + d.seq + ', ' + d.latency_ms.toFixed(0) + ' ms)';
      S.dropFlashUntil = performance.now() + 1500;
      pushEvent(d.frame, d.event);
      return;
    }
    if (d.seq < S.lastSeq) return;
    S.lastSeq = d.seq;
    S.decision = d;
    if (d.intervened) {
      S.flashText = 'shield replaced ' + d.proposed + ' with ' + d.executed;
      S.dropFlashUntil = performance.now() + 1200;
    }
    if (d.event && d.event.indexOf('Answer') !== 0) pushEvent(d.frame, d.event);
    S.series.lat.push(d.latency_ms);
    S.series.inf.push(d.inference_ms);
    if (typeof d.server_ms === 'number') S.series.srv.push(d.server_ms);
    for (const k of Object.keys(S.series)) if (S.series[k].length > 120) S.series[k].splice(0, S.series[k].length - 120);
    renderDecision(d);
  }

  function renderDecision(d) {
    $('#dec-seq').textContent = '#' + d.seq;
    $('#dec-state').textContent = d.state || '';
    $('#dec-instructions').textContent = d.instructions || '';
    for (const a of ACTIONS) {
      const row = document.querySelector('.row[data-a="' + a + '"]');
      row.querySelector('.text').textContent = (d.criteria && d.criteria[a]) || '';
      const p = (d.p && d.p[a]) || 0;
      row.querySelector('.bar i').style.width = (p * 100).toFixed(1) + '%';
      row.querySelector('.pct').textContent = (p * 100).toFixed(1) + '%';
      row.classList.toggle('proposed', d.proposed === a);
      row.classList.toggle('executed', d.executed === a);
      row.classList.toggle('best', d.best === a);
      row.classList.toggle('unsafe', !!(d.safe && d.safe[a] === false));
    }
    $('#t-lat').textContent = d.latency_ms.toFixed(1) + ' ms';
    $('#t-inf').textContent = d.inference_ms.toFixed(1) + ' ms';
    $('#t-srv').textContent = typeof d.server_ms === 'number' ? d.server_ms.toFixed(1) + ' ms' : 'n/a';
    spark($('#sp-lat'), S.series.lat);
    spark($('#sp-inf'), S.series.inf);
    spark($('#sp-srv'), S.series.srv);
  }

  function spark(canvas, values) {
    const c = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    c.clearRect(0, 0, w, h);
    if (values.length < 2) return;
    const max = Math.max(1, ...values);
    const styles = getComputedStyle(document.documentElement);
    const line = styles.getPropertyValue('--teal').trim() || '#007c92';
    const fill = styles.getPropertyValue('--teal-dim').trim() || 'rgba(0,124,146,.09)';
    c.beginPath();
    values.forEach((v, i) => {
      const x = (i / 119) * (w - 2) + 1;
      const y = h - 2 - (v / max) * (h - 6);
      if (i === 0) c.moveTo(x, y); else c.lineTo(x, y);
    });
    c.strokeStyle = line;
    c.lineWidth = 1.5;
    c.stroke();
    c.lineTo(((values.length - 1) / 119) * (w - 2) + 1, h);
    c.lineTo(1, h);
    c.closePath();
    c.fillStyle = fill;
    c.fill();
  }

  function renderStats(m) {
    $('#c-dec').textContent = m.decisions.toLocaleString();
    $('#c-rate').textContent = m.rate.toFixed(1) + ' / s';
    $('#c-inflight').textContent = m.inflight_active + ' / ' + m.inflight;
    $('#c-drop').textContent = m.discarded.toLocaleString();
    const saves = m.interventions + m.arrival_saves + m.emergency_saves;
    $('#c-shield').textContent = saves.toLocaleString();
    $('#c-shield-split').textContent = m.interventions + ' replaced, ' + m.arrival_saves + ' arrival, ' + m.emergency_saves + ' emergency';
    $('#c-agree').textContent = m.agreement === null ? 'n/a' : (m.agreement * 100).toFixed(1) + '%';
    $('#c-score').textContent = m.score;
    $('#c-best').textContent = 'best ' + m.best_score;
    $('#c-deaths').textContent = m.deaths;
    $('#c-time').textContent = m.game_seconds.toFixed(0) + ' s played, ' + m.seconds_left.toFixed(0) + ' s left';
    $('#c-err').textContent = m.errors;
    $('#c-stall').textContent = 'host stall ' + m.host_stall_s.toFixed(2) + ' s' + (m.last_error ? ', last error: ' + m.last_error : '');
  }

  function pct(v) { return v === null || v === undefined ? 'n/a' : (v * 100).toFixed(1) + '%'; }
  function ms(v) { return v === null || v === undefined ? 'n/a' : Number(v).toFixed(1) + ' ms'; }

  function refRow(seed) {
    if (!S.reference || !S.reference.results) return null;
    return S.reference.results.find((r) => r.seed === seed) || null;
  }

  function cell(text, ref, cls) {
    const td = document.createElement('td');
    td.textContent = text;
    if (cls) td.className = cls;
    if (ref !== null && ref !== undefined) {
      const span = document.createElement('span');
      span.className = 'ref';
      span.textContent = '4090: ' + ref;
      td.appendChild(span);
    }
    return td;
  }

  function renderRows() {
    $('#summary').hidden = false;
    const body = $('#summary-rows');
    body.innerHTML = '';
    for (const r of S.rows) {
      const ref = refRow(r.seed);
      const tr = document.createElement('tr');
      const saves = r.shield_interventions + r.arrival_saves + r.emergency_saves;
      tr.append(
        cell(String(r.seed)),
        cell(r.survived ? 'yes' : 'no', ref ? (ref.survived ? 'yes' : 'no') : null, r.survived ? 'yes' : 'no'),
        cell(String(r.best_score), ref ? ref.best_score : null),
        cell(r.decisions.toLocaleString(), ref ? ref.decisions.toLocaleString() : null),
        cell(pct(r.agreement_with_planner), ref ? pct(ref.agreement_with_planner) : null),
        cell(ms(r.latency_ms_p50), ref ? ms(ref.latency_ms_p50) : null),
        cell(ms(r.model_ms_p50), ref ? ms(ref.model_ms_p50) : null),
        cell(ms(r.server_ms_p50)),
        cell(String(r.answers_discarded), ref ? String(ref.answers_discarded) : null),
        cell(String(saves), ref ? String(ref.shield_interventions + ref.arrival_saves + ref.emergency_saves) : null),
        cell(String(r.deaths), ref ? String(ref.deaths) : null),
        cell(r.host_stall_seconds_dropped.toFixed(2) + ' s'),
      );
      body.appendChild(tr);
    }
    const foot = $('#summary-foot');
    foot.innerHTML = '';
    if (S.summary) {
      const s = S.summary;
      const tr = document.createElement('tr');
      tr.append(cell('this run'), cell(s.survived + ' of ' + s.seeds), cell(String(s.mean_best_score)), cell(s.mean_decisions.toLocaleString()), cell(pct(s.mean_agreement_with_planner)), cell(ms(s.latency_ms_p50_median)), cell(ms(s.model_ms_p50_median)), cell(ms(s.server_ms_p50_median)), cell(String(s.answers_discarded)), cell(String(s.shield_interventions)), cell(String(s.deaths)), cell(''));
      foot.appendChild(tr);
    }
    if (S.reference && S.reference.summary) {
      const s = S.reference.summary;
      const tr = document.createElement('tr');
      tr.className = 'ref';
      tr.append(cell('authors, RTX 4090'), cell(s.survived + ' of ' + s.seeds), cell(String(s.mean_best_score)), cell(s.mean_decisions.toLocaleString()), cell(pct(s.mean_agreement_with_planner)), cell(ms(s.latency_ms_p50_median)), cell(ms(s.model_ms_p50_median)), cell('n/a'), cell(String(s.answers_discarded)), cell(String(s.shield_interventions)), cell(String(s.deaths)), cell(''));
      foot.appendChild(tr);
    }
    const ep = S.course ? S.course.endpoint.replace(/ · /g, ', ') : '';
    $('#summary-note').textContent = 'Each row is one 60 FPS course played in real time; the same fields as the CLM repository’s examples/t_rex results. Reference: the authors’ clm_realtime.json (one RTX 4090, 2026-09-23, commit bb42c6c5), shown per seed in grey where the seeds match.' + (ep ? ' This run: ' + ep + '.' : '');
  }

  const canvas = $('#game');
  const ctx = canvas.getContext('2d');

  function fit() {
    const cw = canvas.parentElement.clientWidth || 1200;
    const k = Math.min(2, cw / W);
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(W * k * dpr);
    canvas.height = Math.round(H * k * dpr);
    canvas.style.height = Math.round(H * k) + 'px';
    scale = k * dpr;
  }

  function mix(a, b, t) {
    const c = a.map((v, i) => Math.round(v + (b[i] - v) * t));
    return 'rgb(' + c.join(',') + ')';
  }

  function rect(x, y, w, h) { ctx.fillRect(x, y, w, h); }

  function roundRect(x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
    ctx.fill();
  }

  function tri(ax, ay, bx, by, cx, cy) {
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.lineTo(cx, cy);
    ctx.closePath();
    ctx.fill();
  }

  function drawTrex(t, crashed, fg, bg) {
    const [x, y, status, frame, ducking] = t;
    ctx.fillStyle = fg;
    if (ducking) {
      roundRect(x + DUCK_BOX[0], y + DUCK_BOX[1], DUCK_BOX[2], DUCK_BOX[3], 4);
      rect(x + 40, y + 14, 17, 14);
      ctx.fillStyle = bg;
      rect(x + 50, y + 17, 2, 2);
      ctx.fillStyle = fg;
      if (frame % 2 === 0) rect(x + 12, y + 43, 4, 4); else rect(x + 24, y + 43, 4, 4);
      return;
    }
    for (const b of TREX_BOXES) rect(x + b[0], y + b[1], b[2], b[3]);
    rect(x + 28, y + 22, 2, 4);
    tri(x + 1, y + 18, x - 4, y + 22, x + 1, y + 27);
    if (status === 2) {
      rect(x + 12, y + 41, 4, 4);
      rect(x + 22, y + 41, 4, 4);
    } else if (status === 1 && frame % 2 === 1) {
      rect(x + 12, y + 40, 4, 7);
      rect(x + 22, y + 42, 4, 5);
    } else {
      rect(x + 12, y + 42, 4, 5);
      rect(x + 22, y + 40, 4, 7);
    }
    ctx.fillStyle = bg;
    if (crashed) {
      rect(x + 29, y + 3, 1, 4);
      rect(x + 31, y + 3, 1, 4);
    } else {
      rect(x + 30, y + 4, 2, 2);
    }
  }

  function drawObstacle(o, fg) {
    const [, kind, size, x, y, frame] = o;
    ctx.fillStyle = fg;
    if (kind === 2) {
      ctx.beginPath();
      ctx.ellipse(x + 24, y + 22, 14, 6, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(x + 38, y + 17, 5, 0, Math.PI * 2);
      ctx.fill();
      tri(x + 43, y + 17, x + 46, y + 16, x + 43, y + 19);
      tri(x + 2, y + 20, x + 12, y + 17, x + 12, y + 24);
      if (frame === 0) tri(x + 18, y + 20, x + 26, y + 6, x + 30, y + 20); else tri(x + 18, y + 24, x + 26, y + 36, x + 30, y + 24);
      return;
    }
    const c = CACTUS[kind];
    for (let i = 0; i < size; i++) for (const b of c.boxes) roundRect(x + i * c.w + b[0], y + b[1], b[2], b[3], 2);
  }

  function drawClouds(clouds, fg) {
    ctx.strokeStyle = fg;
    ctx.globalAlpha = 0.55;
    ctx.lineWidth = 1;
    for (const [x, y] of clouds) {
      ctx.beginPath();
      ctx.arc(x + 12, y + 8, 7, Math.PI, 0);
      ctx.arc(x + 24, y + 5, 9, Math.PI * 1.05, Math.PI * 1.95);
      ctx.arc(x + 36, y + 8, 7, Math.PI, 0);
      ctx.lineTo(x + 43, y + 14);
      ctx.lineTo(x + 5, y + 14);
      ctx.closePath();
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  function drawHorizon(h, fg) {
    ctx.fillStyle = fg;
    rect(0, HORIZON_Y, W, 1);
    for (let i = 0; i < 2; i++) {
      const x0 = h[i];
      const bumpy = h[2 + i] !== 0;
      for (let k = 0; k < 24; k++) {
        const px = x0 + k * 25 + (bumpy ? (k * 7) % 11 : (k * 3) % 5);
        if (px < -4 || px > W) continue;
        rect(px, HORIZON_Y + (bumpy ? 3 + (k % 3) : 2), bumpy ? 3 : 2, 1);
      }
    }
  }

  function drawNight(n, fg) {
    const [opacity, moonX, phase, stars, drawStars] = n;
    if (opacity <= 0) return;
    ctx.globalAlpha = opacity;
    ctx.fillStyle = fg;
    if (drawStars) for (const [sx, sy] of stars) rect(sx, sy, 2, 2);
    const cx = moonX + 20, cy = 30;
    ctx.beginPath();
    ctx.arc(cx, cy, 20, 0, Math.PI * 2);
    ctx.fill();
    const off = MOON_OFFSET[phase] || 0;
    if (off !== 0) {
      ctx.save();
      ctx.globalCompositeOperation = 'destination-out';
      ctx.beginPath();
      ctx.arc(cx + off, cy, 19, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
    ctx.globalAlpha = 1;
  }

  function drawScore(g, fg) {
    ctx.fillStyle = fg;
    ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'right';
    if (g.vis) ctx.fillText(String(g.shown).padStart(5, '0'), 592, 22);
    if (g.hi > 0) {
      ctx.globalAlpha = 0.5;
      ctx.fillText('HI ' + String(g.hi).padStart(5, '0'), 538, 22);
      ctx.globalAlpha = 1;
    }
    ctx.textAlign = 'left';
  }

  function drawGameOver(g, fg) {
    ctx.fillStyle = fg;
    ctx.font = '600 12px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'center';
    ctx.fillText('G A M E  O V E R', 300, 60);
    ctx.textAlign = 'left';
    if (g.rf > 0) {
      ctx.strokeStyle = fg;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(300, 85, 8, Math.PI * 0.2, Math.PI * 1.8);
      ctx.stroke();
      tri(303, 76, 309, 80, 303, 84);
    }
  }

  function drawOverlay(g, s) {
    const styles = getComputedStyle(document.documentElement);
    ctx.strokeStyle = styles.getPropertyValue('--teal').trim() || '#007c92';
    ctx.lineWidth = 0.75;
    const [x, y, , , ducking] = g.t;
    const boxes = ducking ? [DUCK_BOX] : TREX_BOXES;
    for (const b of boxes) ctx.strokeRect(x + b[0] + 0.5, y + b[1] + 0.5, b[2], b[3]);
    for (const o of g.o) {
      const [, kind, size, ox, oy] = o;
      if (kind === 2) for (const b of BIRD.boxes) ctx.strokeRect(ox + b[0] + 0.5, oy + b[1] + 0.5, b[2], b[3]);
      else { const c = CACTUS[kind]; for (let i = 0; i < size; i++) for (const b of c.boxes) ctx.strokeRect(ox + i * c.w + b[0] + 0.5, oy + b[1] + 0.5, b[2], b[3]); }
    }
    ctx.fillStyle = styles.getPropertyValue('--accent').trim() || '#b1040e';
    for (let i = 0; i < Math.min(8, s.thinking); i++) rect(6 + i * 7, 6, 5, 5);
  }

  function drawIdle(fg) {
    ctx.fillStyle = fg;
    rect(0, HORIZON_Y, W, 1);
    drawTrex([50, 93, 0, 0, 0], 0, fg, mix(DAY.bg, NIGHT.bg, 0));
    ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillText('press Start', 300, 60);
  }

  function draw() {
    requestAnimationFrame(draw);
    const f = S.frame;
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    const inv = f ? f.g.inv : 0;
    const fg = mix(DAY.fg, NIGHT.fg, inv);
    const bg = mix(DAY.bg, NIGHT.bg, inv);
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, W, H);
    if (!f) { drawIdle(fg); return; }
    const g = f.g;
    ctx.save();
    if (g.rv < W) { ctx.beginPath(); ctx.rect(0, 0, g.rv, H); ctx.clip(); }
    drawNight(g.n, fg);
    drawClouds(g.c, fg);
    drawHorizon(g.h, fg);
    for (const o of g.o) drawObstacle(o, fg);
    drawTrex(g.t, g.cr, fg, bg);
    ctx.restore();
    drawScore(g, fg);
    if (g.cr) drawGameOver(g, fg);
    if (S.overlay) drawOverlay(g, f.s);
    const flash = $('#dec-flash');
    const show = performance.now() < S.dropFlashUntil;
    if (show) flash.textContent = S.flashText;
    flash.hidden = !show;
  }

  function start(ev) {
    ev.preventDefault();
    const msg = {
      type: 'start',
      seeds: Number($('#f-seeds').value),
      seed: Number($('#f-seed').value),
      duration: Number($('#f-duration').value),
      inflight: Number($('#f-inflight').value),
      shield: $('#f-shield').checked,
    };
    if (!$('#f-key-wrap').hidden) {
      msg.api_key = $('#f-key').value;
      try { localStorage.setItem(LS.key, msg.api_key); } catch (e) { }
    }
    S.rows = [];
    S.summary = null;
    S.lastSeq = 0;
    S.events = [];
    S.series = { lat: [], inf: [], srv: [] };
    $('#summary').hidden = true;
    $('#ctl-note').textContent = '';
    setPhase('starting');
    send(msg);
  }

  async function probe() {
    try {
      const r = await fetch('../v1/models', { cache: 'no-store' });
      if (r.status === 401) {
        $('#f-key-wrap').hidden = false;
        try { $('#f-key').value = localStorage.getItem(LS.key) || ''; } catch (e) { }
      }
    } catch (e) { }
    try {
      const r = await fetch('../health', { cache: 'no-store' });
      const j = await r.json();
      $('#mock-banner').hidden = !j.mock;
    } catch (e) { }
    try {
      const r = await fetch('reference_rtx4090.json', { cache: 'no-store' });
      S.reference = await r.json();
    } catch (e) { }
  }

  function init() {
    let theme = '';
    try { theme = localStorage.getItem(LS.theme) === 'dark' ? 'dark' : ''; } catch (e) { }
    applyTheme(theme);
    $('#btn-theme').addEventListener('click', () => {
      const now = document.documentElement.dataset.theme === 'dark' ? '' : 'dark';
      try { localStorage.setItem(LS.theme, now); } catch (e) { }
      applyTheme(now);
    });
    $('#controls').addEventListener('submit', start);
    $('#btn-stop').addEventListener('click', () => send({ type: 'stop' }));
    $('#overlay').addEventListener('change', (e) => { S.overlay = e.target.checked; });
    $('#status').addEventListener('click', () => { if (!ws || ws.readyState !== WebSocket.OPEN) connect(); });
    window.addEventListener('resize', fit);
    fit();
    probe();
    connect();
    requestAnimationFrame(draw);
    setInterval(() => send({ type: 'ping' }), 25000);
  }

  init();
})();
