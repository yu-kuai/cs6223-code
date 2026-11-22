"""Re-judge every violation of finished fuzzing runs with an external judge model (offline vLLM batch).
Uses exactly the same judge prompt as the in-loop self-judge.
Writes <run>/judge_ext_<tag>.jsonl, one line per violation: {"line": index in events.jsonl, ...}."""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompts as P  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--task", choices=["mcq", "vqa"], default="mcq")
ap.add_argument("--seeds", default=None)
ap.add_argument("runs", nargs="+")
a = ap.parse_args()

a.seeds = a.seeds or ("data/scienceqa_img_seeds.jsonl" if a.task == "vqa" else "data/mmlu_seeds.jsonl")
seeds = {json.loads(l)["id"]: json.loads(l) for l in open(a.seeds)}
if a.task == "vqa":
    import vqa  # noqa: E402
    for d in seeds.values():
        d["answer"] = d["answer_letter"]
mech_cache = {}


def mechanism(run, pid, ver):
    k = (run, pid, ver)
    if k not in mech_cache:
        path = os.path.join(run, "programs", f"{pid}_v{ver}.py")
        m = re.search(r'^MECHANISM\s*=\s*(["\'])(.*?)\1', open(path).read(), flags=re.M) if os.path.exists(path) else None
        mech_cache[k] = m.group(2) if m else ""
    return mech_cache[k]


jobs = []   # (run, line, event)
for run in a.runs:
    out = os.path.join(run, f"judge_ext_{a.tag}.jsonl")
    if os.path.exists(out):
        print(f"skip (exists): {out}"); continue
    for i, l in enumerate(open(os.path.join(run, "events.jsonl"))):
        if '"violation"' not in l:
            continue
        e = json.loads(l)
        if e["status"] == "violation":
            jobs.append((run, i, e))
print(f"violations to judge: {len(jobs)}", flush=True)
if not jobs:
    sys.exit(0)

from vllm import LLM, SamplingParams  # noqa: E402
if a.task == "vqa":
    llm = LLM(model=a.model, max_model_len=16384, gpu_memory_utilization=0.90, seed=0, limit_mm_per_prompt={"image": 2},
              tensor_parallel_size=int(os.environ.get("TP", "1")))
    sp = SamplingParams(temperature=0.0, max_tokens=800)   # same as the in-loop VQA judge
else:
    llm = LLM(model=a.model, max_model_len=16384, gpu_memory_utilization=0.90, seed=0,
              tensor_parallel_size=int(os.environ.get("TP", "1")))
    sp = SamplingParams(temperature=0.0, max_tokens=400)
msgs = []
def _clip(t, n):
    return t if len(t) <= n else t[:n] + " ...[truncated]"


for run, i, e in jobs:
    s = seeds[e["seed_id"]]
    # a few degenerate mutations/replies are huge; clip so every prompt fits the context window
    e["target_reply"] = _clip(e["target_reply"], 6000)
    e["mutated"] = dict(e["mutated"], question=_clip(e["mutated"]["question"], 15000),
                        choices=[_clip(str(c), 2000) for c in e["mutated"]["choices"]])
    mech = mechanism(run, e["program"], e["version"])
    if a.task == "vqa":
        orig = vqa.image_url(s["image"])
        mi = e["mutated"].get("image")
        mut = vqa.image_url(os.path.join(run, mi)) if mi else orig
        msgs.append(P.judge_messages_vqa(s, e["mutated"], e["expected"], e["got"], e["target_reply"], mech, orig, mut))
    else:
        pr = P.judge_prompt(s, e["mutated"], e["expected"], e["got"], e["target_reply"], mech)
        msgs.append([{"role": "user", "content": pr}])
outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)

files = {}
for (run, i, e), o in zip(jobs, outs):
    j = P.parse_judge(o.outputs[0].text)
    if run not in files:
        files[run] = open(os.path.join(run, f"judge_ext_{a.tag}.jsonl"), "w")
    files[run].write(json.dumps({"line": i, "round": e["round"], "program": e["program"], "version": e["version"],
                                 "seed_id": e["seed_id"], "self": e["judge"]["verdict"], "ext": j["verdict"],
                                 "ext_reason": j["reason"], "ext_pattern": j["pattern"]}) + "\n")
for f in files.values():
    f.close()

# quick agreement summary
for run in a.runs:
    p = os.path.join(run, f"judge_ext_{a.tag}.jsonl")
    if not os.path.exists(p):
        continue
    rows = [json.loads(l) for l in open(p)]
    agree = sum(r["self"] == r["ext"] for r in rows)
    sv = sum(r["self"] == "VALID" for r in rows); ev = sum(r["ext"] == "VALID" for r in rows)
    both = sum(r["self"] == "VALID" and r["ext"] == "VALID" for r in rows)
    print(f"{run}: n={len(rows)} agree={agree / max(len(rows), 1):.2f} self_VALID={sv} ext_VALID={ev} "
          f"self-TP confirmed by ext={both}/{sv} ({both / max(sv, 1):.2f})")
print("EXT_JUDGE_DONE")
