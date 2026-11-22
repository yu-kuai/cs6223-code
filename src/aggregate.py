"""Aggregate all fuzz runs (self, cross-model, repeat seeds) into one CSV + summary tables.
Run dirs: results/fuzz[_vqa]/<F>_self/<strategy>[_s<seed>]  or  results/fuzz[_vqa]/<F>_to_<T>/<strategy>[_s<seed>]
Referee = external judge file judge_ext_<tag>.jsonl (Qwen3-32B for text, Qwen3-VL-32B for VQA) -> the same referee
for every run, so ext-confirmed counts are comparable across fuzzers/judges.
usage: python src/aggregate.py [--out results/analysis/runs_v2]"""
import argparse, csv, glob, json, os, re, statistics as stt
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results/analysis/runs_v2")
a = ap.parse_args()
EXT = {"mcq": "qwen3-32b", "vqa": "qwen3-vl-32b"}
L5 = "ABCDE"
seeds = {}
for _p in ("data/mmlu_seeds.jsonl", "data/scienceqa_img_seeds.jsonl"):
    for _l in open(_p):
        _d = json.loads(_l); seeds[_d["id"]] = _d


def _n(t):
    return " ".join(str(t).lower().split())


def label_err(e):
    """Deterministic oracle check: gold option text still present exactly once but not at the expected letter."""
    sd = seeds[e["seed_id"]]
    gold = sd.get("answer_letter") or sd["answer"]
    ch = [_n(c) for c in e["mutated"]["choices"]]
    gt = _n(sd["choices"][L5.index(gold)])
    return ch.count(gt) == 1 and ch.index(gt) != L5.index(e["expected"])


SIZES = {"mcq": ["qwen3-1.7b", "qwen3-8b", "qwen3-32b"], "vqa": ["qwen3-vl-2b", "qwen3-vl-8b", "qwen3-vl-32b"]}

rows = []
for run in sorted(glob.glob("results/fuzz*/*/*")):
    if "smoke" in run or not os.path.exists(os.path.join(run, "state.json")):
        continue
    task = "vqa" if "fuzz_vqa" in run else "mcq"
    pair, sdir = run.split("/")[-2:]
    m = re.fullmatch(r"(.+)_self", pair)
    F, T = (m.group(1), m.group(1)) if m else pair.split("_to_")
    m = re.fullmatch(r"(.+?)(?:_s(\d+))?", sdir)
    strat, seed = m.group(1), int(m.group(2) or 0)
    st = json.load(open(os.path.join(run, "state.json")))
    calls = sum(st["used"].values())
    c, tp_cl, tp_seeds, ev_cluster = Counter(), set(), set(), {}
    for i, l in enumerate(open(os.path.join(run, "events.jsonl"))):
        if '"violation"' not in l:
            if '"status": "budget"' not in l and '"infra_error"' not in l:
                c["tests"] += 1
            continue
        e = json.loads(l)
        c["tests"] += 1
        if e["status"] != "violation":
            continue
        c["viol"] += 1
        v = e["judge"]["verdict"]
        c["self_" + v] += 1
        bad = label_err(e)
        c["label_err"] += bad
        c["self_clean"] += (v == "VALID" and not bad)
        ev_cluster[i] = (e.get("cluster"), e["seed_id"], v, bad)
    ext_path = os.path.join(run, f"judge_ext_{EXT[task]}.jsonl")
    has_ext = os.path.exists(ext_path)
    if has_ext:
        for l in open(ext_path):
            r = json.loads(l)
            ext = r["ext"]
            c["ext_" + ext] += 1
            c["agree"] += (r["self"] == ext)
            if ext == "VALID":
                cl, sid, sv, bad = ev_cluster.get(r["line"], (None, None, None, False))
                if bad:
                    continue
                c["ext_clean"] += 1
                tp_cl.add(cl); tp_seeds.add(sid)
                if sv == "VALID":
                    c["both"] += 1
    rows.append({
        "task": task, "fuzzer": F, "target": T, "cross": F != T, "strategy": strat, "seed": seed,
        "done": st.get("done", False), "calls": calls, "tests": c["tests"], "viol": c["viol"],
        "self_tp": c["self_VALID"], "self_fp": c["self_INVALID"], "label_err": c["label_err"], "self_clean": c["self_clean"],
        "ext_tp_raw": c["ext_VALID"] if has_ext else "", "ext_tp": c["ext_clean"] if has_ext else "",
        "both": c["both"] if has_ext else "",
        "agree": round(c["agree"] / c["viol"], 2) if has_ext and c["viol"] else "",
        "ext_tp_clusters": len(tp_cl - {None}) if has_ext else "", "ext_tp_seeds": len(tp_seeds - {None}) if has_ext else "",
        "calls_per_ext_tp": round(calls / c["ext_clean"], 1) if has_ext and c["ext_clean"] else "",
        "programs": len(st["programs"]), "run": run})

os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out + ".csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

out = []
def P(s=""):
    out.append(s); print(s)
def ms(xs):
    xs = [x for x in xs if x != ""]
    if not xs: return "-"
    return f"{stt.mean(xs):.0f}" if len(xs) == 1 else f"{stt.mean(xs):.0f}±{stt.stdev(xs):.0f}"

P(f"runs: {len(rows)}  (not done: {sum(not r['done'] for r in rows)})")
P("ext_tp/both/ext_clu/calls exclude label errors (gold option text present at a letter other than expected);")
P("self_cl = self-judged VALID minus label errors; lab_err = label errors among all violations.")
for task in ("mcq", "vqa"):
    P(f"\n=== {task}: SELF runs, mean±sd over seeds (n)  [ext = external {EXT[task]} referee]")
    P(f"{'target':14s}{'strategy':20s}{'n':>3s}{'viol':>10s}{'lab_err':>10s}{'self_tp':>10s}{'self_cl':>10s}{'ext_tp':>10s}{'both':>10s}{'ext_clu':>10s}{'calls/ext_tp':>14s}")
    for T in SIZES[task]:
        for s in ["fixed", "inconsistency", "inconsistency_guard", "full", "full_guard", "no_fp", "no_tp"]:
            g = [r for r in rows if r["task"] == task and not r["cross"] and r["target"] == T and r["strategy"] == s and r["done"]]
            if g:
                P(f"{T:14s}{s:20s}{len(g):>3d}{ms([r['viol'] for r in g]):>10s}{ms([r['label_err'] for r in g]):>10s}{ms([r['self_tp'] for r in g]):>10s}{ms([r['self_clean'] for r in g]):>10s}"
                  f"{ms([r['ext_tp'] for r in g]):>10s}{ms([r['both'] for r in g]):>10s}{ms([r['ext_tp_clusters'] for r in g]):>10s}"
                  f"{ms([r['calls_per_ext_tp'] for r in g]):>14s}")
    for s in ["full", "fixed"]:
        for metric in ["ext_tp", "self_clean", "ext_tp_clusters", "label_err", "agree"]:
            P(f"\n--- {task} {s}: {metric}   rows = fuzzer+judge, cols = target (diagonal = self, seed 0)")
            P(" " * 14 + "".join(f"{T:>14s}" for T in SIZES[task]))
            for F in SIZES[task]:
                cells = []
                for T in SIZES[task]:
                    g = [r for r in rows if r["task"] == task and r["fuzzer"] == F and r["target"] == T
                         and r["strategy"] == s and r["seed"] == 0 and r["done"]]
                    cells.append(str(g[0][metric]) if g and g[0][metric] != "" else "-")
                P(f"{F:14s}" + "".join(f"{x:>14s}" for x in cells))
open(a.out + ".txt", "w").write("\n".join(out) + "\n")
