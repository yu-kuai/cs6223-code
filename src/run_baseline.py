"""Stochasticity baseline: ask each unchanged seed once greedily + k sampled times.
Appends JSONL per seed; skips seeds already in output (resumable)."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from mcq import messages, parse_answer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--seeds", default="data/mmlu_seeds.jsonl")
ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--chunk", type=int, default=200)
ap.add_argument("--max-model-len", type=int, default=4096)
ap.add_argument("--max-tokens", type=int, default=1024)
a = ap.parse_args()

seeds = [json.loads(l) for l in open(a.seeds)]
done = set()
if os.path.exists(a.out):
    done = {json.loads(l)["id"] for l in open(a.out) if l.strip()}
todo = [s for s in seeds if s["id"] not in done]
print(f"seeds={len(seeds)} done={len(done)} todo={len(todo)}", flush=True)
if not todo:
    sys.exit(0)

from vllm import LLM, SamplingParams
llm = LLM(model=a.model, max_model_len=a.max_model_len, gpu_memory_utilization=0.90, seed=0)
greedy = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)
# Qwen3 recommended non-thinking sampling settings
sampled = SamplingParams(n=a.k, temperature=0.7, top_p=0.8, top_k=20, max_tokens=a.max_tokens, seed=0)
kw = dict(chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)

for i in range(0, len(todo), a.chunk):
    batch = todo[i:i + a.chunk]
    msgs = [messages(s["question"], s["choices"]) for s in batch]
    g = llm.chat(msgs, greedy, **kw)
    r = llm.chat(msgs, sampled, **kw)
    with open(a.out, "a") as f:
        for s, go, ro in zip(batch, g, r):
            gt = go.outputs[0].text
            st = [o.text for o in ro.outputs]
            f.write(json.dumps({
                "id": s["id"], "subject": s["subject"], "gold": s["answer"],
                "greedy": parse_answer(gt, s["choices"]), "greedy_raw": gt,
                "samples": [parse_answer(t, s["choices"]) for t in st], "samples_raw": st,
                "finish": [go.outputs[0].finish_reason] + [o.finish_reason for o in ro.outputs],
            }) + "\n")
    print(f"wrote {i + len(batch)}/{len(todo)}", flush=True)
