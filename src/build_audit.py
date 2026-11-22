"""Build the manual audit set: a stratified sample of violations (text + VQA) and a self-contained HTML labeling page.

Strata per task, by the two judges (in-loop judge vs external 32B referee):
  both_valid / disagree / both_invalid  -> default 60 / 60 / 30 items per task.
Each item keeps its stratum weight (population / sampled), so precision and recall of each judge can be
re-weighted to all violations. Judge verdicts are NOT shown on the page (blind labeling); they are in key.jsonl.

Outputs (in --out-dir): audit.html (open in a browser; labels autosave in the browser; "Export CSV" downloads them),
items.jsonl (everything incl. verdicts), key.csv.
usage: python src/build_audit.py [--per-task 60 60 30] [--seed 0]"""
import argparse, base64, csv, glob, html, io, json, os, random, re
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--per-task", type=int, nargs=3, default=[60, 60, 30], help="both_valid disagree both_invalid")
ap.add_argument("--max-per-run", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out-dir", default="results/audit")
a = ap.parse_args()
rng = random.Random(a.seed)
EXT = {"mcq": "qwen3-32b", "vqa": "qwen3-vl-32b"}
LET = "ABCDE"

seeds = {}
for p in ("data/mmlu_seeds.jsonl", "data/scienceqa_img_seeds.jsonl"):
    for l in open(p):
        d = json.loads(l); seeds[d["id"]] = d


def _n(t):
    return " ".join(str(t).lower().split())


def label_err(e):
    s = seeds[e["seed_id"]]; gold = s.get("answer_letter") or s["answer"]
    ch = [_n(c) for c in e["mutated"]["choices"]]; gt = _n(s["choices"][LET.index(gold)])
    return ch.count(gt) == 1 and ch.index(gt) != LET.index(e["expected"])


def mechanism(run, pid, ver):
    path = os.path.join(run, "programs", f"{pid}_v{ver}.py")
    if not os.path.exists(path):
        return ""
    m = re.search(r'^MECHANISM\s*=\s*(["\'])(.*?)\1', open(path).read(), flags=re.M)
    return m.group(2) if m else ""


pool = defaultdict(list)   # (task, stratum) -> items
for run in sorted(glob.glob("results/fuzz*/*/*")):
    if "smoke" in run:
        continue
    task = "vqa" if "fuzz_vqa" in run else "mcq"
    jp = os.path.join(run, f"judge_ext_{EXT[task]}.jsonl")
    if not os.path.exists(jp):
        continue
    ext = {}
    for l in open(jp):
        r = json.loads(l); ext[r["line"]] = r
    for i, l in enumerate(open(os.path.join(run, "events.jsonl"))):
        if i not in ext:
            continue
        e = json.loads(l); r = ext[i]
        sv, xv = e["judge"]["verdict"], r["ext"]
        if sv not in ("VALID", "INVALID") or xv not in ("VALID", "INVALID"):
            continue
        st = "both_valid" if sv == xv == "VALID" else "both_invalid" if sv == xv else "disagree"
        pool[(task, st)].append((run, i, e, r))

items = []
for task in ("mcq", "vqa"):
    for st, k in zip(("both_valid", "disagree", "both_invalid"), a.per_task):
        cand = pool[(task, st)]
        by_run = defaultdict(list)
        for c in cand:
            by_run[c[0]].append(c)
        for v in by_run.values():
            rng.shuffle(v)
        runs = list(by_run); rng.shuffle(runs)
        pick, taken = [], defaultdict(int)
        while len(pick) < k and any(by_run[r] and taken[r] < a.max_per_run for r in runs):   # round-robin over runs
            for r in runs:
                if len(pick) < k and by_run[r] and taken[r] < a.max_per_run:
                    pick.append(by_run[r].pop()); taken[r] += 1
        w = len(cand) / max(len(pick), 1)
        for run, i, e, r in pick:
            items.append({"task": task, "stratum": st, "weight": round(w, 3), "run": run, "line": i,
                          "program": e["program"], "version": e["version"], "seed_id": e["seed_id"],
                          "mechanism": mechanism(run, e["program"], e["version"]),
                          "mutated": e["mutated"], "expected": e["expected"], "got": e["got"],
                          "target_reply": e["target_reply"], "self": e["judge"]["verdict"],
                          "self_reason": e["judge"].get("reason", ""), "ext": r["ext"], "ext_reason": r["ext_reason"],
                          "label_err": label_err(e)})
rng.shuffle(items)
for n, it in enumerate(items):
    it["id"] = f"A{n:03d}"
os.makedirs(a.out_dir, exist_ok=True)
with open(os.path.join(a.out_dir, "items.jsonl"), "w") as f:
    for it in items:
        f.write(json.dumps(it) + "\n")
with open(os.path.join(a.out_dir, "key.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "task", "stratum", "weight", "run", "line", "program", "self", "ext", "label_err"])
    for it in items:
        w.writerow([it[k] for k in ("id", "task", "stratum", "weight", "run", "line", "program", "self", "ext", "label_err")])

# ---------------------------------------------------------------- HTML page
from PIL import Image  # noqa: E402


def thumb(path):
    im = Image.open(path).convert("RGB"); im.thumbnail((480, 480))
    b = io.BytesIO(); im.save(b, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def opts(choices, mark=None):
    return "".join(f"<li{' class=gold' if LET[j] == mark else ''}>({LET[j]}) {html.escape(str(c))}</li>"
                   for j, c in enumerate(choices))


E = html.escape
cards = []
for it in items:
    s = seeds[it["seed_id"]]; m = it["mutated"]
    gold = s["answer_letter"] if it["task"] == "vqa" else s["answer"]
    hint_o, hint_m = s.get("hint", ""), m.get("hint", s.get("hint", ""))
    img = ""
    if it["task"] == "vqa":
        oi = thumb(s["image"])
        mi = thumb(os.path.join(it["run"], m["image"])) if m.get("image") else oi
        img = (f"<div class=imgs><figure><img src='{oi}'><figcaption>original</figcaption></figure>"
               f"<figure><img src='{mi}'><figcaption>mutated{'' if m.get('image') else ' (image unchanged)'}"
               f"</figcaption></figure></div>")
    cards.append(f"""
<section class=card id={it['id']} data-task={it['task']}>
 <h2>{it['id']} <small>{'VQA' if it['task']=='vqa' else 'Text'} · program: {E(it['program'].split('_',1)[1][:80])}</small></h2>
 <p class=mech><b>Mutation claims:</b> {E(it['mechanism'][:400]) or '-'}</p>
 {img}
 <div class=cols>
  <div><h3>Original (answer {gold})</h3>{f'<p class=hint>Context: {E(hint_o)}</p>' if hint_o else ''}
   <p>{E(s['question'])}</p><ul>{opts(s['choices'], gold)}</ul></div>
  <div><h3>Mutated (expected answer {it['expected']})</h3>{f'<p class=hint>Context: {E(hint_m)}</p>' if hint_m else ''}
   <p>{E(m['question'])}</p><ul>{opts(m['choices'], it['expected'])}</ul></div>
 </div>
 <details><summary>Model answered <b>{it['got']}</b> (show reply)</summary><pre>{E(it['target_reply'][:2500])}</pre></details>
 <div class=form>
  <label>1. Is the expected answer ({it['expected']}) still the single correct answer of the mutated input?
   <select name=valid><option></option><option>yes</option><option>no</option><option>unsure</option></select></label>
  <label>2. If no: why? <select name=invalid_reason><option></option><option>meaning/answer changed</option>
   <option>needed information removed or destroyed</option><option>text garbled/ungrammatical</option>
   <option>image changed what is asked (e.g. colours, mirroring, labels)</option><option>expected-answer mapping wrong</option>
   <option>other</option></select></label>
  <label>3. If yes: failure type <select name=failure_type><option></option><option>option order / position</option>
   <option>paraphrase / wording</option><option>irrelevant context / distractor text</option><option>notation / format / typo</option>
   <option>negation / framing</option><option>context (hint) removed or changed</option><option>image quality (noise, blur, jpeg, scale)</option>
   <option>image colour / grayscale</option><option>image geometry / layout</option><option>visual distractor / marker</option>
   <option>several stacked changes</option><option>other</option></select></label>
  <label>Notes <input name=notes size=60></label>
 </div>
</section>""")

page = f"""<!DOCTYPE html><html><head><meta charset=utf-8><title>Fuzzing audit ({len(items)} items)</title>
<style>body{{font:14px system-ui,sans-serif;max-width:1100px;margin:auto;padding:1em;background:#f6f6f4}}
.card{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:1em;margin:1em 0}}.card.done{{border-left:6px solid #3a8}}
h2 small{{color:#777;font-weight:normal;font-size:13px}}.cols{{display:grid;grid-template-columns:1fr 1fr;gap:1em}}
.gold{{font-weight:bold;background:#eef7ee}}.hint{{color:#555;font-style:italic}}.imgs{{display:flex;gap:1em}}
.imgs img{{max-width:480px;border:1px solid #ccc}}.mech{{color:#444}}pre{{white-space:pre-wrap;background:#f3f3f3;padding:.5em}}
.form label{{display:block;margin:.4em 0}}#bar{{position:sticky;top:0;background:#222;color:#fff;padding:.5em 1em;border-radius:6px;z-index:9}}
#bar button{{margin-left:1em}}</style></head><body>
<div id=bar>Labeled <span id=cnt>0</span>/{len(items)} · labels autosave in this browser
 <button onclick="exportCsv()">Export CSV</button> <label><input type=checkbox id=hide onchange="hideDone()"> hide labeled</label>
 <label style="margin-left:1em">Show <select id=flt onchange="hideDone()"><option value="">all</option><option value=mcq>text</option><option value=vqa>VQA</option></select></label></div>
<p>Label each case: is the mutated input still answered by the expected letter? Judge verdicts are hidden on purpose.
Split the work by ID range; each person exports a CSV at the end.</p>
{''.join(cards)}
<script>
const KEY='cs6223_audit_v1';let S=JSON.parse(localStorage.getItem(KEY)||'{{}}');
document.querySelectorAll('.card').forEach(c=>{{const id=c.id;c.querySelectorAll('select,input').forEach(el=>{{
 if(S[id]&&S[id][el.name]!==undefined) el.value=S[id][el.name];
 el.addEventListener('change',()=>{{S[id]=S[id]||{{}};S[id][el.name]=el.value;localStorage.setItem(KEY,JSON.stringify(S));mark();}});}});}});
function mark(){{let n=0;document.querySelectorAll('.card').forEach(c=>{{const d=S[c.id]&&S[c.id].valid;c.classList.toggle('done',!!d);if(d)n++;}});
 document.getElementById('cnt').textContent=n;}}
function hideDone(){{const h=document.getElementById('hide').checked,f=document.getElementById('flt').value;
 document.querySelectorAll('.card').forEach(c=>{{c.style.display=((h&&c.classList.contains('done'))||(f&&c.dataset.task!==f))?'none':'';}});}}
function exportCsv(){{const cols=['valid','invalid_reason','failure_type','notes'];let rows=['id,'+cols.join(',')];
 document.querySelectorAll('.card').forEach(c=>{{const s=S[c.id]||{{}};rows.push(c.id+','+cols.map(k=>'"'+(s[k]||'').replace(/"/g,'""')+'"').join(','));}});
 const b=new Blob([rows.join('\\n')],{{type:'text/csv'}});const u=document.createElement('a');u.href=URL.createObjectURL(b);u.download='audit_labels.csv';u.click();}}
mark();
</script></body></html>"""
open(os.path.join(a.out_dir, "audit.html"), "w").write(page)
from collections import Counter  # noqa: E402
print("pool sizes:", {f"{k[0]}/{k[1]}": len(v) for k, v in sorted(pool.items())})
print("label_err in sample:", sum(it["label_err"] for it in items))
print("sampled:", Counter((it["task"], it["stratum"]) for it in items))
print("html MB:", round(os.path.getsize(os.path.join(a.out_dir, "audit.html")) / 1e6, 1))
