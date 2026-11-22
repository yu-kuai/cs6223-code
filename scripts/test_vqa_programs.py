"""Offline test of VQA seed programs in the sandbox (no LLM server): llm() echoes the question back slightly
reworded, similarity() returns 0.9. Reports per program: ok / check_reject / error, timing, and saves a few
before/after images to --out-dir for eyeballing."""
import argparse, base64, glob, io, json, os, random, sys, time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from sandbox import run_program  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--programs", default=os.path.join(ROOT, "programs", "seed_vqa"))
ap.add_argument("--seeds", default=os.path.join(ROOT, "data", "scienceqa_img_seeds.jsonl"))
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--out-dir", default=os.path.join(ROOT, "results", "vqa_program_test"))
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)

rows = [json.loads(l) for l in open(a.seeds)]
rng = random.Random(0)
sample = rng.sample(rows, a.n)


def sandbox_seed(s):
    d = {k: s[k] for k in ("id", "question", "choices", "hint", "subject", "topic")}
    d["answer"] = s["answer_letter"]
    d["image"] = base64.b64encode(open(s["image"], "rb").read()).decode()
    return d


handlers = {"llm": lambda prompt, **kw: prompt.rsplit("Question:", 1)[-1].strip().replace("Which", "What"),
            "similarity": lambda a, b: 0.9,
            "vlm": lambda prompt, image=None, **kw: "a picture"}
for path in sorted(glob.glob(os.path.join(a.programs, "*.py"))):
    code = open(path).read()
    name = os.path.basename(path)[:-3]
    c, why, t0, saved = Counter(), Counter(), time.time(), 0
    for i, s in enumerate(sample):
        r = run_program(code, sandbox_seed(s), i, handlers, timeout=60)
        if r.get("error"):
            c["error"] += 1; why[r["error"].strip().splitlines()[-1][:120]] += 1; continue
        if r.get("check") is not True:
            c["check_reject"] += 1; why[r["check"]] += 1; continue
        c["ok"] += 1
        m = r["mutated"]
        assert len(m["choices"]) == len(s["choices"]), "choices count changed"
        if saved < 2 and m["image"] != sandbox_seed(s)["image"]:
            from PIL import Image
            before = Image.open(s["image"]).convert("RGB")
            after = Image.open(io.BytesIO(base64.b64decode(m["image"]))).convert("RGB")
            W = before.width + after.width + 20
            pair = Image.new("RGB", (W, max(before.height, after.height)), "white")
            pair.paste(before, (0, 0)); pair.paste(after, (before.width + 20, 0))
            pair.thumbnail((1200, 600))
            pair.save(os.path.join(a.out_dir, f"{name}_{s['id']}.png")); saved += 1
    dt = (time.time() - t0) / len(sample)
    print(f"{name:28s} {dict(c)}  {dt:.2f}s/seed  {dict(why.most_common(3))}", flush=True)
print("TEST_DONE")
