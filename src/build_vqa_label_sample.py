"""Blind sample of DeepSeek-accepted ('real') image failures: 20 from v4, 20 from hand-written (A).
Output: results/audit/vqa_items.jsonl (shuffled, no setup) + vqa_key.jsonl (setup per item)."""
import json, glob, random
rng = random.Random(7)
seeds = {json.loads(l)["id"]: json.loads(l) for l in open("data/scienceqa_img_seeds.jsonl")}
plan = {"v4": ("full_v4", {"vl-2b": 4, "vl-8b": 8, "vl-32b": 8}), "A": ("fixed", {"vl-2b": 6, "vl-8b": 7, "vl-32b": 7})}
items = []
for setup, (pref, per) in plan.items():
    for m, k in per.items():
        pool = []
        for d in glob.glob(f"results/fuzz_vqa/qwen3-{m}_self/{pref}*"):
            if "smoke" in d: continue
            ev = open(d + "/events.jsonl").read().splitlines()
            for l in open(d + "/judge_ext_deepseek.jsonl"):
                j = json.loads(l)
                if j["ext"] == "VALID":
                    pool.append((d, j, json.loads(ev[j["line"]])))
        for d, j, e in rng.sample(pool, k):
            s = seeds[e["seed_id"]]; mu = e["mutated"]
            img = mu.get("image")
            items.append({"setup": setup, "model": m, "program": j["program"], "seed_id": e["seed_id"],
                          "orig_q": s["question"], "orig_hint": s.get("hint", ""), "orig_choices": s["choices"],
                          "answer": s["answer_letter"], "new_q": mu.get("question"), "new_hint": mu.get("hint", s.get("hint", "")),
                          "new_choices": mu.get("choices"), "expected": e.get("expected"),
                          "orig_img": s["image"], "new_img": (d + "/" + img) if img else s["image"]})
rng.shuffle(items)
with open("results/audit/vqa_items.jsonl", "w") as f, open("results/audit/vqa_key.jsonl", "w") as k:
    for i, it in enumerate(items):
        k.write(json.dumps({"i": i, "setup": it["setup"], "model": it["model"], "program": it["program"]}) + "\n")
        f.write(json.dumps({"i": i, **{x: y for x, y in it.items() if x not in ("setup", "program")}}) + "\n")
print(len(items))
