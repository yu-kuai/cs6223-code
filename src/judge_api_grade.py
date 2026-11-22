"""Grade every violation of finished text runs with a hosted judge (OpenAI-compatible API, e.g. DeepSeek via SoClaaS).
Same judge prompt as the in-loop judge and as judge_external.py; same output format: <run>/judge_ext_<tag>.jsonl.
Resumable: lines already graded are skipped. Raw replies go to <run>/judge_ext_<tag>_raw.jsonl.
usage: python src/judge_api_grade.py --model x-test-2 --tag deepseek <run dir> [<run dir> ...]"""
import argparse, base64, json, os, re, sys, threading
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompts as P  # noqa: E402
from llm_client import ApiLLM, Budget  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("runs", nargs="+")
a = ap.parse_args()
seeds = {json.loads(l)["id"]: json.loads(l) for l in open("data/mmlu_seeds.jsonl")}
for l in open("data/scienceqa_img_seeds.jsonl"):   # VQA seeds: answer letter + image path, as in the fuzz loop
    d = json.loads(l); d["answer"] = d["answer_letter"]; seeds[d["id"]] = d


def img_url(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()


def clip(t, n):
    return t if len(t) <= n else t[:n] + " ...[truncated]"


def mechanism(run, pid, ver):
    path = os.path.join(run, "programs", f"{pid}_v{ver}.py")
    m = re.search(r'^MECHANISM\s*=\s*(["\'])(.*?)\1', open(path).read(), flags=re.M) if os.path.exists(path) else None
    return m.group(2) if m else ""


budget = Budget(10 ** 9, set())
for run in a.runs:
    out = os.path.join(run, f"judge_ext_{a.tag}.jsonl")
    done = {json.loads(l)["line"] for l in open(out)} if os.path.exists(out) else set()
    jobs = []
    for i, l in enumerate(open(os.path.join(run, "events.jsonl"))):
        if '"violation"' in l and i not in done:
            e = json.loads(l)
            if e["status"] == "violation":
                jobs.append((i, e))
    print(f"{run}: {len(jobs)} to grade ({len(done)} done)", flush=True)
    if not jobs:
        continue
    judge = ApiLLM(a.model, budget, os.path.join(run, f"judge_ext_{a.tag}_raw.jsonl"), max_parallel=a.workers)
    lock = threading.Lock()
    f = open(out, "a")

    def grade(job):
        i, e = job
        s = seeds[e["seed_id"]]
        m = dict(e["mutated"], question=clip(e["mutated"]["question"], 15000),
                 choices=[clip(str(c), 2000) for c in e["mutated"]["choices"]])
        mech = mechanism(run, e["program"], e["version"])
        if "image" in s:   # VQA: original image + mutated image (saved in the run dir, or unchanged)
            mi = e["mutated"].get("image")
            msgs = P.judge_messages_vqa(s, m, e["expected"], e["got"], clip(e.get("target_reply", ""), 6000), mech,
                                        img_url(s["image"]), img_url(os.path.join(run, mi)) if mi else img_url(s["image"]))
        else:
            msgs = [{"role": "user", "content": P.judge_prompt(s, m, e["expected"], e["got"],
                                                              clip(e.get("target_reply", ""), 6000), mech)}]
        try:
            j = P.parse_judge(judge.chat(msgs, "grade", max_tokens=4096))
        except Exception as ex:
            print(f"  line {i}: failed ({ex}); will retry on next invocation", flush=True)
            return
        with lock:
            f.write(json.dumps({"line": i, "round": e["round"], "program": e["program"], "version": e["version"],
                                "seed_id": e["seed_id"], "self": e["judge"]["verdict"], "ext": j["verdict"],
                                "ext_reason": j["reason"], "ext_pattern": j["pattern"]}) + "\n")
            f.flush()

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(grade, jobs))
    f.close()
    rows = [json.loads(l) for l in open(out)]
    print(f"  graded={len(rows)} ext_VALID={sum(r['ext'] == 'VALID' for r in rows)} "
          f"agree_with_loop_judge={sum(r['self'] == r['ext'] for r in rows) / max(len(rows), 1):.2f}", flush=True)
print("GRADE_DONE", flush=True)
