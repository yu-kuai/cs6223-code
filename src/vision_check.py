"""Does a hosted model really use the image? Ask N ScienceQA image questions with and without the picture and compare
accuracy; also ask it to describe a few images. usage: python src/vision_check.py --models x-test-2 x-test-1 --n 40
Output: results/analysis/vision_check.json + a printed table."""
import argparse, base64, json, os, random, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_client import ApiLLM, Budget  # noqa: E402
import vqa  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--out", default="results/analysis/vision_check.json")
a = ap.parse_args()
rows = [json.loads(l) for l in open("data/scienceqa_img_seeds.jsonl")]
random.Random(0).shuffle(rows)
rows = rows[:a.n]
os.makedirs("results/analysis", exist_ok=True)
budget = Budget(10 ** 9, set())


def url(r):
    return "data:image/png;base64," + base64.b64encode(open(r["image"], "rb").read()).decode()


def ask(llm, r, with_image):
    text = vqa.format_question(r["question"], r["choices"], r.get("hint", "")) + \
        ("" if with_image else "\n(The image is not available; answer as best you can.)") + \
        "\nAnswer with the letter of the correct option only."
    content = ([{"type": "image_url", "image_url": {"url": url(r)}}] if with_image else []) + [{"type": "text", "text": text}]
    try:
        reply = llm.chat([{"role": "user", "content": content}], "check", max_tokens=6000)
    except Exception as e:
        return None, str(e)[:100]
    return vqa.parse_answer(reply, r["choices"]), reply[-200:]


res = {}
for m in a.models:
    llm = ApiLLM(m, budget, f"results/analysis/vision_check_{m}_raw.jsonl", max_parallel=8)
    jobs = [(r, w) for r in rows for w in (True, False)]
    with ThreadPoolExecutor(8) as ex:
        outs = list(ex.map(lambda j: ask(llm, *j), jobs))
    acc = {True: 0, False: 0}; fail = 0; per = []
    for (r, w), (got, rep) in zip(jobs, outs):
        ok = got == r["answer_letter"]; acc[w] += ok; fail += got is None
        per.append({"id": r["id"], "image": w, "got": got, "gold": r["answer_letter"]})
    desc = []
    for r in rows[:5]:
        d = llm.chat([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url(r)}},
                      {"type": "text", "text": "Describe this image in one sentence (what it shows, any text or labels in it)."}]}],
                     "check", max_tokens=4000)
        desc.append({"id": r["id"], "question": r["question"], "description": d[:300]})
    res[m] = {"n": len(rows), "acc_with_image": acc[True] / len(rows), "acc_without_image": acc[False] / len(rows),
              "unparsed": fail, "per": per, "descriptions": desc}
    print(f"{m:12s} with image {acc[True]}/{len(rows)}   without image {acc[False]}/{len(rows)}   unparsed {fail}", flush=True)
    for d in desc:
        print(f"   [{d['id']}] Q: {d['question'][:70]}\n      -> {d['description'][:220]}", flush=True)
json.dump(res, open(a.out, "w"), indent=1)
print("VISION_CHECK_DONE", flush=True)
