"""Build VQA seeds from ScienceQA (derek-thomas/ScienceQA): test split, questions WITH an image.

Writes data/scienceqa_img_seeds.jsonl (one row per question) and saves each image as PNG under data/scienceqa_img/.
Row: id, question, choices (2-5), answer (int index), answer_letter, hint (text context, may be ''), image (path),
subject, topic, category, grade.
"""
import argparse, json, os, random
from collections import Counter

from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/scienceqa_img_seeds.jsonl")
ap.add_argument("--img-dir", default="data/scienceqa_img")
ap.add_argument("--split", default="test")
ap.add_argument("--max", type=int, default=0, help="0 = keep all image questions; else stratified sample by topic")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

ds = load_dataset("derek-thomas/ScienceQA", split=a.split)
print(ds)
rows = [(i, r) for i, r in enumerate(ds) if r["image"] is not None]
print(f"{len(rows)} of {len(ds)} {a.split} questions have an image")

if a.max and a.max < len(rows):  # round-robin over topics so small topics are kept
    rng = random.Random(a.seed)
    by_topic = {}
    for i, r in rows:
        by_topic.setdefault(r["topic"], []).append((i, r))
    for v in by_topic.values():
        rng.shuffle(v)
    picked, k = [], 0
    while len(picked) < a.max:
        added = False
        for t in sorted(by_topic):
            if k < len(by_topic[t]) and len(picked) < a.max:
                picked.append(by_topic[t][k]); added = True
        if not added:
            break
        k += 1
    rows = sorted(picked, key=lambda x: x[0])

os.makedirs(a.img_dir, exist_ok=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w") as f:
    for i, r in rows:
        sid = f"sqa-{a.split}-{i}"
        path = os.path.join(a.img_dir, sid + ".png")
        r["image"].convert("RGB").save(path)
        f.write(json.dumps({
            "id": sid, "question": r["question"], "choices": r["choices"], "answer": r["answer"],
            "answer_letter": "ABCDE"[r["answer"]], "hint": r["hint"] or "", "image": path,
            "subject": r["subject"], "topic": r["topic"], "category": r["category"], "grade": r["grade"],
        }) + "\n")
print("wrote", len(rows), "->", a.out)
print("subjects", Counter(r["subject"] for _, r in rows))
print("n_choices", Counter(len(r["choices"]) for _, r in rows))
print("topics", len({r["topic"] for _, r in rows}))
