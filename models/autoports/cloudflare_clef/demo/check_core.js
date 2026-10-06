"use strict";
const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");

const demo = __dirname;
const html = fs.readFileSync(path.join(demo, "index.html"), "utf8");
const start = html.indexOf('<script id="clef-core">');
const end = html.indexOf("</script>", start);
if (start < 0 || end < 0) throw new Error("clef-core script block not found in index.html");
const source = html.slice(html.indexOf(">", start) + 1, end);
const module_ = { exports: {} };
new Function("module", source)(module_);
const CLEF = module_.exports;

const data = JSON.parse(fs.readFileSync(path.join(demo, "presets.json"), "utf8"));
const model = data.model || "clef";

function editorRoundTrip(c) {
  const isString = typeof c.state === "string";
  const stateText = isString ? c.state : JSON.stringify(c.state, null, 2);
  const questionsText = JSON.stringify(c.questions, null, 2);
  const images = (c.images || []).map((p) => fs.readFileSync(path.join(demo, p)).toString("base64"));
  return CLEF.buildRequest(CLEF.parseState(stateText, !isString), CLEF.parseQuestions(questionsText), model, images);
}

function cliRequests(args) {
  const python = process.env.PYTHON || "python3";
  const out = execFileSync(python, [path.join(demo, "quickstart.py"), "--show-request", ...args], { maxBuffer: 1 << 28 }).toString();
  const bodies = [];
  let inside = false, buf = "";
  for (const line of out.split("\n")) {
    if (line === "{") { inside = true; buf = ""; }
    if (!inside) continue;
    buf += line + "\n";
    if (line === "}") { bodies.push(JSON.parse(buf)); inside = false; }
  }
  return bodies;
}

const pageCases = [];
for (const p of data.presets) for (const c of p.cases) pageCases.push(editorRoundTrip(c));
const pageReplay = data.replay.map(editorRoundTrip);
const cliCases = cliRequests(["--all"]);
const cliReplay = cliRequests(["--replay"]);

let same = 0, total = 0;
function compare(name, a, b) {
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    total += 1;
    const x = JSON.stringify(a[i]), y = JSON.stringify(b[i]);
    if (x === y) same += 1;
    else console.log(`MISMATCH ${name}[${i}]: page ${x ? x.length : 0} chars, cli ${y ? y.length : 0} chars`);
  }
}
compare("cases", pageCases, cliCases);
compare("replay", pageReplay, cliReplay);

const probe = { type: "score", score: 1.82, probabilities: { "0": 0.0451, "1": 0.0898, "2": 0.8651 } };
if (CLEF.predicted(probe) !== 2) throw new Error("predicted(score) wrong");
if (CLEF.predicted({ type: "noul", noul: 0.5 }) !== true) throw new Error("predicted(noul) wrong");
if (!CLEF.same("B", "B") || CLEF.same(true, false)) throw new Error("same() wrong");
const bad = ["{}", "[]", '{"q": {"type": "pick"}}', '{"q": {"type": "choice", "criteria": {}}}', '{"q": {"type": "score", "criteria": {}}}', '{"q": {"type": "noul", "criteria": []}}'];
for (const text of bad) {
  let threw = false;
  try { CLEF.parseQuestions(text); } catch (e) { threw = true; }
  if (!threw) throw new Error("parseQuestions accepted " + text);
}
const req = pageCases[pageCases.length - 1];
const curl = CLEF.snippetCurl("http://127.0.0.1:8008", "", req, ["assets/receipt_northwind.png"]);
if (!curl.includes('IMG0=$(base64 -w0 "assets/receipt_northwind.png")') || !curl.includes('"$IMG0"')) throw new Error("curl snippet lacks the image variable");
const py = CLEF.snippetRequests("http://127.0.0.1:8008", "k", req, ["assets/receipt_northwind.png"]);
if (!py.includes("base64.b64encode") || !py.includes('"authorization": "Bearer k"')) throw new Error("python snippet wrong");

console.log(`request bodies identical: ${same} of ${total} (${pageCases.length} cases, ${pageReplay.length} replay records); validation, predicted and snippet checks passed`);
if (same !== total) process.exit(1);
