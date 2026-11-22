"""Sample N questions per MMLU subject (test split) -> JSONL seeds."""
import argparse, json, random
from collections import defaultdict
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--per-subject", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

ds = load_dataset("cais/mmlu", "all", split="test")
by_subj = defaultdict(list)
for i, r in enumerate(ds):
    by_subj[r["subject"]].append((i, r))

rng = random.Random(a.seed)
n = 0
with open(a.out, "w") as f:
    for subj in sorted(by_subj):
        rows = by_subj[subj]
        for i, r in rng.sample(rows, min(a.per_subject, len(rows))):
            f.write(json.dumps({
                "id": f"mmlu-{subj}-{i}", "subject": subj, "question": r["question"],
                "choices": r["choices"], "answer": "ABCD"[r["answer"]],
            }) + "\n")
            n += 1
print(f"subjects={len(by_subj)} seeds={n} -> {a.out}")
