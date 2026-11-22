"""Sample 60 text v3 violations for labeling: 45 DeepSeek-VALID (15 from the 8B s1 option_reordering_with_* family, 15 from the 32B s2 *negation*
program, 15 from all other v3 runs) + 15 DeepSeek-INVALID. Writes results/audit/v3_items.jsonl (no verdicts) and
results/audit/v3_key.jsonl (GLM + DeepSeek verdicts). Blind ids V000..V059 in shuffled order."""
import glob, json, os, random
rng = random.Random(0)
pool = []
for run in sorted(glob.glob("results/fuzz/qwen3-*_self/full_v3*")):
    ext = {json.loads(l)["line"]: json.loads(l) for l in open(f"{run}/judge_ext_deepseek.jsonl")}
    for i, l in enumerate(open(f"{run}/events.jsonl")):
        if '"violation"' in l and i in ext:
            e = json.loads(l)
            if e["status"] == "violation":
                pool.append((run, i, e, ext[i]))
top = {"results/fuzz/qwen3-8b_self/full_v3_s1": "option_reordering_with",
       "results/fuzz/qwen3-32b_self/full_v3_s2": "negation"}
is_top = lambda r, e: r in top and top[r] in e["program"]
groups = [[p for p in pool if p[3]["ext"] == "VALID" and p[0] == r and is_top(r, p[2])] for r in top]
groups.append([p for p in pool if p[3]["ext"] == "VALID" and not is_top(p[0], p[2])])
groups.append([p for p in pool if p[3]["ext"] == "INVALID"])
pick = []
for g, k, name in zip(groups, (15, 15, 15, 15), ("8b_top", "32b_top", "other_valid", "ds_invalid")):
    print(name, len(g)); pick += [(name, p) for p in rng.sample(g, k)]
rng.shuffle(pick)
with open("results/audit/v3_items.jsonl", "w") as fi, open("results/audit/v3_key.jsonl", "w") as fk:
    for n, (grp, (run, i, e, x)) in enumerate(pick):
        vid = f"V{n:03d}"
        fi.write(json.dumps({"id": vid, "seed_id": e["seed_id"], "mutated": e["mutated"], "expected": e["expected"],
                             "got": e["got"]}) + "\n")
        fk.write(json.dumps({"id": vid, "group": grp, "run": run, "line": i, "program": e["program"],
                             "glm": e["judge"]["verdict"], "deepseek": x["ext"], "ds_reason": x["ext_reason"]}) + "\n")
