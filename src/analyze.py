"""Per-run summary table for fuzzing runs (streams events.jsonl; low memory).
usage: python src/analyze.py results/fuzz/*_self/*  [--csv out.csv]"""
import argparse, csv, json, os
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--csv")
a = ap.parse_args()

rows = []
for run in sorted(a.runs):
    ev_path, st_path = os.path.join(run, "events.jsonl"), os.path.join(run, "state.json")
    if not (os.path.exists(ev_path) and os.path.exists(st_path)) or "smoke" in run:
        continue
    c = Counter()
    rounds = []                       # (round, fp, violations) for trend
    tp_seeds, tp_pairs = set(), set()
    last_round = 0
    for line in open(ev_path):
        e = json.loads(line)
        s = e["status"]
        if s in ("budget", "infra_error"):
            c[s] += 1
            continue
        c["tests"] += 1
        c[s] += 1
        last_round = max(last_round, e["round"])
        if s == "violation":
            v = e["judge"]["verdict"]
            c["v_" + v] += 1
            rounds.append((e["round"], v))
            if v == "VALID":
                tp_seeds.add(e["seed_id"])
                fam = e["program"].split("_", 1)[1]
                tp_pairs.add((e["seed_id"], fam))
    st = json.load(open(st_path))
    used = st["used"]
    progs = st["programs"]
    synth = Counter()
    sp = os.path.join(run, "synth.jsonl")
    if os.path.exists(sp):
        for line in open(sp):
            r = json.loads(line)
            synth[f"{r['kind']}_{'ok' if r['ok'] else 'fail'}"] += 1
    half = last_round / 2
    def fp_rate(sel):
        vv = [v for r, v in rounds if sel(r) and v in ("VALID", "INVALID")]
        return sum(v == "INVALID" for v in vv) / len(vv) if vv else float("nan")
    tp = c["v_VALID"]
    rows.append({
        "run": run.split("fuzz/")[-1],
        "rounds": st["round"], "tests": c["tests"],
        "err%": round(100 * c["error"] / max(c["tests"], 1), 1),
        "reject%": round(100 * c["check_reject"] / max(c["tests"], 1), 1),
        "asked": c["pass"] + c["violation"] + c["unparsed"],
        "viol": c["violation"], "TP": tp, "FP": c["v_INVALID"],
        "self_prec": round(tp / max(tp + c["v_INVALID"], 1), 2),
        "FP%_1st": round(100 * fp_rate(lambda r: r <= half), 1),
        "FP%_2nd": round(100 * fp_rate(lambda r: r > half), 1),
        "TP_seeds": len(tp_seeds), "online_clusters": len(st["clusters"]),
        "calls/TP": round(sum(used.get(k, 0) for k in ("fuzzer_mutate", "fuzzer_synth", "target", "judge")) / max(tp, 1), 1),
        "programs": len(progs), "refine_ok": synth["refine_ok"], "refine_fail": synth["refine_fail"],
        "explore_ok": synth["explore_ok"], "explore_fail": synth["explore_fail"],
        "calls_mut": used.get("fuzzer_mutate", 0), "calls_synth": used.get("fuzzer_synth", 0),
        "calls_target": used.get("target", 0), "calls_judge": used.get("judge", 0),
    })

cols = list(rows[0].keys())
w = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in cols}
show = ["run", "rounds", "tests", "err%", "reject%", "asked", "viol", "TP", "FP", "self_prec", "FP%_1st", "FP%_2nd",
        "TP_seeds", "online_clusters", "calls/TP", "refine_ok", "refine_fail", "explore_ok", "explore_fail"]
print("  ".join(k.ljust(w[k]) for k in show))
for r in rows:
    print("  ".join(str(r[k]).ljust(w[k]) for k in show))
print()
print("calls by role:")
for r in rows:
    print(f"  {r['run']:<28} mutate={r['calls_mut']:<5} synth={r['calls_synth']:<4} target={r['calls_target']:<5} judge={r['calls_judge']}")
if a.csv:
    with open(a.csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols); wr.writeheader(); wr.writerows(rows)
