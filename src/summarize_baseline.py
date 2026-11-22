"""Summarize baseline JSONL: accuracy, unparsed rate, sampling inconsistency (unparsed kept separate)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import mcq, vqa

SEEDS = {}
def reparse(r):  # always re-parse from raw text so parser fixes apply to old runs
    ch = SEEDS.get(r["id"])
    parse_answer = vqa.parse_answer if "n_choices" in r else mcq.parse_answer  # VQA rows carry n_choices
    r["greedy"] = parse_answer(r["greedy_raw"], ch)
    r["samples"] = [parse_answer(t, ch) for t in r["samples_raw"]]
    return r

for fn in ("mmlu_seeds.jsonl", "scienceqa_img_seeds.jsonl"):
    fp = os.path.join(os.path.dirname(__file__), "..", "data", fn)
    if os.path.exists(fp):
        for l in open(fp):
            d = json.loads(l); SEEDS[d["id"]] = d["choices"]
for path in sys.argv[1:]:
    rows = [reparse(json.loads(l)) for l in open(path)]
    n = len(rows); ns = sum(len(r["samples"]) for r in rows)
    g_unp = sum(r["greedy"] is None for r in rows)
    s_unp = sum(x is None for r in rows for x in r["samples"])
    trunc = sum(f == "length" for r in rows for f in r.get("finish", []))
    acc = sum(r["greedy"] == r["gold"] for r in rows) / n
    sacc = sum(x == r["gold"] for r in rows for x in r["samples"]) / ns
    full = [r for r in rows if r["greedy"] is not None and None not in r["samples"]]
    incons = sum(len(set(r["samples"])) > 1 for r in full) / max(len(full), 1)
    gdiff = sum(any(x != r["greedy"] for x in r["samples"]) for r in full) / max(len(full), 1)
    print(f"{path}\n  n={n} greedy_acc={acc:.3f} sampled_acc={sacc:.3f} (unparsed counted wrong)\n"
          f"  unparsed: greedy={g_unp}/{n} samples={s_unp}/{ns} truncated={trunc}\n"
          f"  on {len(full)} fully-parsed seeds: samples_inconsistent={incons:.3f} any_sample!=greedy={gdiff:.3f}")
