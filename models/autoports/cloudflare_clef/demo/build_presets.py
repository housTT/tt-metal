import argparse
import ast
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DEMO = Path(__file__).resolve().parent
ASSETS = DEMO / "assets"
OUT = DEMO / "presets.json"
REFERENCE = Path("/home/hous/dev/clef/reports/reference")
EVALS = Path("/home/hous/dev/clef/evals")
KEV_BENCH = Path("/home/hous/dev/kev/kev/scripts/serving_bench.py")
KEV_COMMIT = "952ce9d"
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf")

FIXED_IDS = {"readme_invoice", "readme_checkout", "blog_support_triage"}
CARTOONS = ["2a7ddcfe4724ee1403a6291d21347162", "6c1478b40fa4ab7f72933d7e3fe58d0a"]
REPLAY_PLAN = [
    ("ARC-Challenge", EVALS / "arc_challenge_test_sample100.jsonl", 8, "first"),
    ("BANKING77", EVALS / "banking77_test_sample100.jsonl", 8, "short_intents"),
    ("New Yorker", EVALS / "newyorker_matching_test_sample100.jsonl", 8, "first"),
]
FIVE = ("department", "return_reason", "requested_resolution", "escalate", "frustration")
SECOND_SET = ("tone", "escalate", "frustration")
TONE_DESCRIPTIONS = {
    "calm": "No sign of irritation",
    "frustrated": "Annoyed but still civil",
    "angry": "Hostile, threatening or shouting",
}
RECEIPT_LINES = [
    ("NORTHWIND HARDWARE", True),
    ("412 Harbor Street, Portland ME", False),
    ("Store 0412   Register 3", False),
    ("2026-09-28  17:42", False),
    ("", False),
    ("Cordless drill 18V        79.00", False),
    ("Drill bit set, 21 pc      24.50", False),
    ("Wood screws #8 x 1.5 in    6.95", False),
    ("Safety glasses             9.00", False),
    ("", False),
    ("SUBTOTAL                 119.45", False),
    ("TAX 6.7%                   8.00", False),
    ("TOTAL                    127.45", True),
    ("", False),
    ("VISA ****4471           127.45", False),
    ("APPROVED  AUTH 03921", False),
    ("", False),
    ("Thank you for shopping with us", False),
    ("Returns within 30 days with receipt", False),
]


def read_jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def bench_constants():
    tree = ast.parse(KEV_BENCH.read_text())
    wanted = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("QUESTIONS", "PARAGRAPH"):
                wanted[node.targets[0].id] = ast.literal_eval(node.value)
    missing = {"QUESTIONS", "PARAGRAPH"} - set(wanted)
    if missing:
        raise SystemExit(f"{KEV_BENCH}: missing {sorted(missing)}")
    return wanted


def copy_asset(src):
    src = Path(src)
    dst = ASSETS / src.name
    if not dst.exists() or dst.read_bytes() != src.read_bytes():
        shutil.copyfile(src, dst)
    return f"assets/{src.name}"


def case_from_record(rec, source):
    case = {"name": rec["id"], "source": source, "state": rec["state"], "questions": rec["questions"]}
    if rec.get("images"):
        case["images"] = [copy_asset(p) for p in rec["images"]]
    if rec.get("_label"):
        case["labels"] = rec["_label"]
    return case


def fixed_examples():
    rows = {r["id"]: r for r in read_jsonl(REFERENCE / "records_text.jsonl") if r["id"] in FIXED_IDS}
    missing = FIXED_IDS - set(rows)
    if missing:
        raise SystemExit(f"records_text.jsonl: missing {sorted(missing)}")
    return rows


def cartoon_cases():
    rows = {r["id"]: r for r in read_jsonl(REFERENCE / "records_image.jsonl")}
    out = []
    for cid in CARTOONS:
        if cid not in rows:
            raise SystemExit(f"records_image.jsonl: missing {cid}")
        rec = rows[cid]
        case = case_from_record(rec, f"{rec['_source']}, row {rec['_source_row']}, contest {rec['_contest_number']}")
        case["name"] = f"contest {rec['_contest_number']} ({cid[:8]})"
        out.append(case)
    return out


def draw_receipt():
    width, height = 520, 760
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    regular = ImageFont.truetype(str(FONT), 22)
    bold = ImageFont.truetype(str(FONT_BOLD), 26)
    y = 40
    for text, strong in RECEIPT_LINES:
        font = bold if strong else regular
        if text:
            box = draw.textbbox((0, 0), text, font=font)
            x = (width - (box[2] - box[0])) // 2 if strong else 40
            draw.text((x, y), text, fill="black", font=font)
        y += 34 if strong else 30
    draw.line([(40, y + 6), (width - 40, y + 6)], fill="black", width=2)
    dst = ASSETS / "receipt_northwind.png"
    img.save(dst, format="PNG", optimize=True)
    return "assets/receipt_northwind.png", (width, height)


def receipt_case():
    path, size = draw_receipt()
    return {
        "name": "northwind receipt",
        "source": f"synthetic PNG drawn by build_presets.py with Pillow, {size[0]}x{size[1]}, DejaVu Sans Mono",
        "state": "A photo of a printed store receipt.",
        "images": [path],
        "questions": {
            "legible": {"type": "noul", "instructions": "Is the text on the receipt legible?"},
            "total_over_100": {"type": "noul", "instructions": "Is the receipt total above 100 dollars?"},
            "vendor": {
                "type": "choice",
                "instructions": "Which store issued this receipt?",
                "criteria": {
                    "northwind": "Northwind Hardware",
                    "acme": "Acme Supply",
                    "globex": "Globex Market",
                    "initech": "Initech Office Store",
                },
            },
        },
        "labels": {"legible": True, "total_over_100": True, "vendor": "northwind"},
    }


def replay_records():
    out = []
    for suite, path, count, rule in REPLAY_PLAN:
        rows = read_jsonl(path)
        if rule == "short_intents":
            seen = set()
            picked = []
            for rec in sorted(rows, key=lambda r: (len(r["_label"]["intent"]), r["_label"]["intent"], r["id"])):
                intent = rec["_label"]["intent"]
                if intent in seen:
                    continue
                seen.add(intent)
                picked.append(rec)
                if len(picked) == count:
                    break
        else:
            picked = rows[:count]
        if len(picked) < count:
            raise SystemExit(f"{path}: only {len(picked)} usable records, wanted {count}")
        for rec in picked:
            case = case_from_record(rec, f"{path.name} row {rec['_source_row']} ({rec['_source']})")
            out.append({"suite": suite, **case})
    return out


def build():
    ASSETS.mkdir(exist_ok=True)
    fixed = fixed_examples()
    bench = bench_constants()
    questions, paragraph = bench["QUESTIONS"], bench["PARAGRAPH"]
    tone = dict(questions["tone"])
    tone["criteria"] = {k: TONE_DESCRIPTIONS[k] for k in questions["tone"]["criteria"]}
    second = {k: (tone if k == "tone" else questions[k]) for k in SECOND_SET}
    long_state = paragraph * 30
    presets = [
        {
            "title": "Support ticket triage",
            "what_it_shows": "The blog post's curl example: a yes/no urgency check, a team choice and a four-level severity score on one short ticket, answered jointly in one pass.",
            "cases": [case_from_record(fixed["blog_support_triage"], fixed["blog_support_triage"]["_source"])],
        },
        {
            "title": "Invoice JSON state",
            "what_it_shows": "The HF README Usage example. The state is a JSON object, not text. Clef encodes choice options in sorted order (draft, overdue, paid) and the server returns them in request order.",
            "cases": [case_from_record(fixed["readme_invoice"], fixed["readme_invoice"]["_source"])],
        },
        {
            "title": "Checkout outage",
            "what_it_shows": "The HF README SystemOne API example: department choice, urgency score with its legend and expected value, and an outage yes/no.",
            "cases": [case_from_record(fixed["readme_checkout"], fixed["readme_checkout"]["_source"])],
        },
        {
            "title": "Long document, repeated state",
            "what_it_shows": "A state of more than 2,000 tokens. The first run prefills it; a second run with the same state hits the prefix cache and skips the prefill. Case 2 asks a different question set on the same state and also hits the cache.",
            "cases": [
                {
                    "name": "5 questions",
                    "source": f"kev {KEV_COMMIT} scripts/serving_bench.py PARAGRAPH x30 with FIVE",
                    "state": long_state,
                    "questions": {k: questions[k] for k in FIVE},
                },
                {
                    "name": "second question set",
                    "source": f"kev {KEV_COMMIT} scripts/serving_bench.py PARAGRAPH x30 with tone, escalate, frustration (tone descriptions added here)",
                    "state": long_state,
                    "questions": second,
                },
            ],
        },
        {
            "title": "New Yorker cartoon",
            "what_it_shows": "An image request: the cartoon goes in as base64 and the model picks the caption written for it out of five. The gold caption is shown next to the model's choice.",
            "cases": cartoon_cases(),
        },
        {
            "title": "Receipt image",
            "what_it_shows": "A synthetic receipt PNG drawn with Pillow (no third-party asset). Two yes/no questions about what is printed and a vendor choice show image reading.",
            "cases": [receipt_case()],
        },
    ]
    return {
        "generated_by": "build_presets.py",
        "model": "clef",
        "sources": {
            "reference_records": str(REFERENCE),
            "evals": str(EVALS),
            "kev_bench": f"{KEV_BENCH} at {KEV_COMMIT}",
        },
        "presets": presets,
        "replay": replay_records(),
    }


def main():
    ap = argparse.ArgumentParser(description="Rebuild presets.json and the assets directory.")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    data = build()
    Path(args.out).write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    n_cases = sum(len(p["cases"]) for p in data["presets"])
    n_assets = len(list(ASSETS.glob("*.png")))
    print(
        f"wrote {args.out}: {len(data['presets'])} presets, {n_cases} cases, {len(data['replay'])} replay records, {n_assets} PNG assets"
    )


if __name__ == "__main__":
    main()
