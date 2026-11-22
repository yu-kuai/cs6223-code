"""Text-only human labeling sheet: the 150 MMLU audit cases from results/audit/items.jsonl,
without any judge verdicts or label-error flags (those stay in key.csv).
Writes results/audit/text_labels.csv (one row per case, empty `label` column) and text_labels.html (printable view).
usage: python src/build_text_label_sheet.py"""
import csv, html, json

seeds = {json.loads(l)["id"]: json.loads(l) for l in open("data/mmlu_seeds.jsonl")}
items = [json.loads(l) for l in open("results/audit/items.jsonl")]
items = [x for x in items if x["task"] == "mcq"]
L = "ABCDE"


def opts(ch):
    return "\n".join(f"{L[i]}) {c}" for i, c in enumerate(ch))


rows = []
for x in items:
    s = seeds[x["seed_id"]]; m = x["mutated"]
    rows.append({
        "id": x["id"], "subject": s["subject"].replace("_", " "),
        "original_question": s["question"], "original_options": opts(s["choices"]),
        "original_correct": f"{s['answer']}) {s['choices'][L.index(s['answer'])]}",
        "changed_question": m["question"], "changed_options": opts(m["choices"]),
        "expected_answer": f"{x['expected']}) {m['choices'][L.index(x['expected'])]}",
        "model_answer": f"{x['got']}) {m['choices'][L.index(x['got'])]}" if x["got"] in L[:len(m["choices"])] else str(x["got"]),
        "label": "", "notes": ""})
rows.sort(key=lambda r: r["id"])

with open("results/audit/text_labels.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)

esc = lambda t: html.escape(t).replace("\n", "<br>")
cards = []
for r in rows:
    cards.append(f"""<div class=card><h3>{r['id']} · {esc(r['subject'])}</h3>
<div class=cols><div><b>Original</b><p>{esc(r['original_question'])}</p><p>{esc(r['original_options'])}</p>
<p>Correct: <b>{esc(r['original_correct'])}</b></p></div>
<div><b>Changed</b><p>{esc(r['changed_question'])}</p><p>{esc(r['changed_options'])}</p>
<p>Expected: <b>{esc(r['expected_answer'])}</b> · Model said: <b>{esc(r['model_answer'])}</b></p></div></div>
<p class=lab>Label: ☐ REAL &nbsp; ☐ BROKEN &nbsp; ☐ UNSURE</p></div>""")
open("results/audit/text_labels.html", "w").write(f"""<!DOCTYPE html><html><head><meta charset=utf-8><title>Text audit: 150 cases</title>
<style>body{{font-family:Arial,sans-serif;max-width:1100px;margin:auto;padding:20px}}.card{{border:1px solid #ccc;border-radius:8px;padding:12px 16px;margin:14px 0;page-break-inside:avoid}}
.cols{{display:flex;gap:24px}}.cols>div{{flex:1}}h3{{margin:0 0 6px}}.lab{{color:#555}}</style></head><body>
<h1>Text audit: {len(rows)} cases</h1>
<p><b>For each case, decide:</b> would a careful person give the <i>Expected</i> answer to the <i>Changed</i> question?</p>
<ul><li><b>REAL</b>: yes. The change kept the meaning and the expected answer is right, so the model's different answer is a real mistake.</li>
<li><b>BROKEN</b>: no. The change altered the meaning, added a hint or ambiguity, or the expected letter points to the wrong option.</li>
<li><b>UNSURE</b>: cannot tell (write why in notes).</li></ul>
<p>Enter labels in <code>text_labels.csv</code> (column <code>label</code>: REAL / BROKEN / UNSURE). Do not look at the judge results.</p>
{''.join(cards)}</body></html>""")
print(f"wrote {len(rows)} cases -> results/audit/text_labels.csv, results/audit/text_labels.html")
