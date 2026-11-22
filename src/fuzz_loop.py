"""Closed-loop metamorphic fuzzing with self-evolving mutation programs.

Strategies:
  fixed          seed programs only, uniform selection, no feedback                (judge = eval only, not counted)
  inconsistency  programs selected/evolved to maximize answer flips, judge ignored  (judge = eval only, not counted)
  full           FP -> tighten program (refine);  TP -> novelty-guided selection + exploration
  no_fp          like full, but no refinement on false positives
  no_tp          like full, but selection/exploration ignore true positives (blind exploration)

Every round: pick (program, seed) pairs -> sandbox mutate+check -> target answers -> judge on violations ->
update stats, cluster failures -> feedback. Resumable: state.json + events.jsonl are written once per round.
"""
import argparse, base64, glob, hashlib, json, math, os, random, re, sys, threading, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_client import LLM, ApiLLM, Budget, BudgetExhausted, wait_ready  # noqa: E402
import mcq, vqa  # noqa: E402
from debatable import shredded  # noqa: E402
from sandbox import run_program  # noqa: E402
import prompts as P  # noqa: E402

STRATEGIES = ["fixed", "inconsistency", "full", "no_fp", "no_tp"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _norm_opt(t):
    return " ".join(str(t).lower().split())


def label_guard(seed, m):
    """If the gold option's text appears exactly once (verbatim, case/space-insensitive) among the mutated choices,
    return its letter (the only correct expected answer); else None (fall back to the program's label_map)."""
    L = "ABCDE"
    gt = _norm_opt(seed["choices"][L.index(seed["answer"])])
    ch = [_norm_opt(c) for c in m["choices"]]
    return L[ch.index(gt)] if ch.count(gt) == 1 else None


_NUM = re.compile(r"\d+(?:\.\d+)?")
_NEG_KEEP = {"not", "no", "never", "none", "except", "neither", "nor", "without", "cannot", "false", "incorrect", "least"}
_NEG_NEW = {"except", "false", "incorrect", "least", "untrue", "wrong"}


def _words(t):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", str(t).lower())


_NEGC = re.compile(r"\b(?:not|no|never|none|neither|nor|cannot|without)\b|n't\b", re.I)


def fixed_checks(seed, m):
    """Harness checks the fuzzer cannot edit (text MCQ and VQA text parts). Returns None if OK, else a short reason."""
    ch = [_norm_opt(c) for c in m["choices"]]
    if len(set(ch)) != len(ch) or not all(ch):
        return "options are not distinct non-empty strings"
    q0, q1 = seed["question"], m["question"]
    missing = [n for n in set(_NUM.findall(q0)) if n not in _NUM.findall(q1)]
    if missing:
        return f"question lost number(s) {missing[:3]}"
    w0, w1 = set(_words(q0)), set(_words(q1))
    if lost := sorted((w0 & _NEG_KEEP) - w1):
        return f"question lost negation word(s) {lost}"
    if added := sorted((w1 & _NEG_NEW) - w0):
        return f"question gained framing word(s) {added}"
    opt_nums0 = sorted(n for c in seed["choices"] for n in _NUM.findall(str(c)))
    opt_nums1 = sorted(n for c in m["choices"] for n in _NUM.findall(str(c)))
    if [n for n in opt_nums0 if n not in opt_nums1]:
        return "options lost a number"
    if "hint" in seed:   # VQA: the text context must keep its numbers too
        h1 = m.get("hint", seed["hint"])
        if [n for n in set(_NUM.findall(seed["hint"])) if n not in _NUM.findall(str(h1))]:
            return "hint lost a number"
    if getattr(fixed_checks, "v4", False):   # v4: tricks the AI judges wrongly accept (see results/audit/claude_labels_summary.md)
        if len(_NEGC.findall(q1)) > len(_NEGC.findall(q0)):
            return "question gained a negation (not/no/never/...)"
        if shredded(q0, q1):
            return "original question was broken up by inserted text (add context only before or after it)"
        L = "ABCDE"
        for old, new in m.get("label_map", {}).items():
            a, b = str(seed["choices"][L.index(old)]), str(m["choices"][L.index(new)])
            if len(b) > 2 * len(a) + 100:
                return f"option {new} became much longer than the original option"
    nq0, nq1 = _norm_opt(q0), _norm_opt(q1)
    for c in seed["choices"]:
        nc = _norm_opt(c)
        if len(nc) > 3 and not nc.isdigit() and nc in nq1 and nc not in nq0:
            return "question newly mentions an option's text"
    return None


# ============================================================================ embeddings
class Embedder:
    def __init__(self, name="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(name, device="cuda" if _has_cuda() else "cpu")
        self.lock, self.cache = threading.Lock(), {}

    def emb(self, texts):
        with self.lock:
            missing = [t for t in texts if t not in self.cache]
            if missing:
                for t, v in zip(missing, self.m.encode(missing, normalize_embeddings=True)):
                    self.cache[t] = [float(x) for x in v]
            return [self.cache[t] for t in texts]

    def similarity(self, a, b):
        ea, eb = self.emb([a, b])
        return sum(x * y for x, y in zip(ea, eb))


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def cos(a, b):
    return sum(x * y for x, y in zip(a, b))


# ============================================================================ seeds
def stable_correct_seeds(baseline_path, seeds_path, task="mcq"):
    seeds = {}
    for l in open(seeds_path):
        d = json.loads(l)
        if task == "vqa":       # ScienceQA rows: answer index -> letter; keep image path separately
            d["answer"] = d["answer_letter"]; d["image_path"] = d.pop("image")
        seeds[d["id"]] = d
    parse_answer = vqa.parse_answer if task == "vqa" else mcq.parse_answer
    keep = []
    for l in open(baseline_path):
        r = json.loads(l)
        s = seeds[r["id"]]
        ans = [parse_answer(r["greedy_raw"], s["choices"])] + [parse_answer(t, s["choices"]) for t in r["samples_raw"]]
        if all(a == s["answer"] for a in ans):
            keep.append(s)
    return keep


# ============================================================================ main loop
class Fuzzer:
    def __init__(self, a):
        self.a = a
        self.run = a.run_dir
        os.makedirs(os.path.join(self.run, "programs"), exist_ok=True)
        self.state_path = os.path.join(self.run, "state.json")
        self.vqa = a.task == "vqa"
        P.configure(a.task, a.hide_answer, a.v4_checks)
        fixed_checks.v4 = a.v4_checks
        self.parse_answer = vqa.parse_answer if self.vqa else mcq.parse_answer
        self.seeds = stable_correct_seeds(a.baseline, a.seeds, a.task)
        self.seed_index = {s["id"]: s for s in self.seeds}
        self.by_subject = defaultdict(list)   # MMLU: balanced over 57 subjects; VQA: uniform over seeds
        for s in self.seeds:
            self.by_subject[s["subject"] if not self.vqa else "all"].append(s)
        os.makedirs(os.path.join(self.run, "images"), exist_ok=True)
        self._b64 = {}
        counted = {"fuzzer_mutate", "fuzzer_synth", "target"}
        if a.strategy in ("full", "no_fp", "no_tp"):
            counted.add("judge")          # judge drives feedback -> part of the method's cost
        if os.path.exists(self.state_path):
            self.st = json.load(open(self.state_path))
            print(f"resuming at round {self.st['round']}", flush=True)
        else:
            self.st = {"round": 0, "used": {}, "programs": {}, "clusters": [], "done": False,
                       "config": vars(a), "n_seeds": len(self.seeds)}
            sdir = ("seed_vqa_hidden" if a.hide_answer else "seed_vqa") if self.vqa else ("seed_hidden" if a.hide_answer else "seed")
            for f in sorted(glob.glob(os.path.join(ROOT, "programs", sdir, "*.py"))):
                self._add_program(open(f).read(), origin="seed", parent=None)
        self.budget = Budget(a.budget, counted, self.st["used"])
        self.fuzzer = LLM(a.fuzzer_url, a.served_name, self.budget)
        self.target = LLM(a.target_url, a.served_name, self.budget)
        if a.judge_api:   # hosted judge (e.g. GLM via SoClaaS): its calls are logged but never counted in the budget
            self.judge = ApiLLM(a.judge_api, self.budget, os.path.join(self.run, "judge_api.jsonl"))
            self.judge_role = "judge_api"
        else:
            self.judge = LLM(a.judge_url, a.served_name, self.budget)
            self.judge_role = "judge"
        self.repair_f = open(os.path.join(self.run, "repair_tests.jsonl"), "a")
        self.emb = Embedder()
        self.events_f = open(os.path.join(self.run, "events.jsonl"), "a")
        self.synth_f = open(os.path.join(self.run, "synth.jsonl"), "a")
        print(f"strategy={a.strategy} stable-correct seeds={len(self.seeds)} counted_roles={sorted(counted)}", flush=True)

    # ------------------------------------------------------------------ programs
    def _meta(self, code):
        ns = {}
        for key in ("FAMILY", "MECHANISM"):
            for line in code.splitlines():
                if line.startswith(key):
                    try:
                        ns[key] = eval(line.split("=", 1)[1].strip(), {})  # string literal only
                    except Exception:
                        pass
        return ns.get("FAMILY", "unknown"), ns.get("MECHANISM", "")

    def _add_program(self, code, origin, parent, pid=None):
        fam, mech = self._meta(code)
        progs = self.st["programs"]
        if pid is None:
            pid = f"p{len(progs):03d}_{fam}"[:60]
            progs[pid] = {"family": fam, "origin": origin, "parent": parent, "version": 0, "active": True, "lineage": pid,
                          "created_round": self.st["round"], "history": []}
        else:
            p = progs[pid]
            p["history"].append({"version": p["version"], "stats": p["stats"]})
            p["version"] += 1
        p = progs[pid]
        p["mechanism"] = mech
        p["stats"] = {}
        path = os.path.join(self.run, "programs", f"{pid}_v{p['version']}.py")
        open(path, "w").write(code)
        p["path"] = path
        return pid

    def code(self, pid):
        return open(self.st["programs"][pid]["path"]).read()

    def active(self):
        return [k for k, v in self.st["programs"].items() if v["active"]]

    def score(self, pid):
        s = self.st["programs"][pid]["stats"]
        T = max(1, sum(self.st["programs"][k]["stats"].get("tries", 0) for k in self.active()))
        ucb = 0.3 * math.sqrt(math.log(T + 1) / (s.get("tries", 0) + 1))
        strat = self.a.strategy
        if strat == "fixed":
            return 1.0
        if strat == "inconsistency":
            return (s.get("violations", 0) + 1) / (s.get("passed", 0) + 2) + ucb
        if strat == "no_tp":   # validity only
            return (s.get("passed", 0) - s.get("fp", 0) + 1) / (s.get("tries", 0) + 2) + ucb
        if self.a.bold:   # v3: real-failure yield dominates; small exploration bonus
            return ((s.get("tp", 0) + 0.5 * s.get("novel", 0) + 0.5) / (s.get("tries", 0) + 2)
                    - 0.5 * s.get("fp", 0) / (s.get("passed", 0) + 1) + ucb / 3)
        # full / no_fp: novel distinct failures dominate, then valid failures, minus false alarms
        return ((2 * s.get("novel", 0) + s.get("tp", 0) + 1) / (s.get("tries", 0) + 2)
                - 0.5 * s.get("fp", 0) / (s.get("passed", 0) + 1) + ucb)

    # ------------------------------------------------------------------ one test
    def handlers(self):
        def llm(prompt, temperature=0.7, max_tokens=512):
            return self.fuzzer.prompt(str(prompt), "fuzzer_mutate", temperature=float(temperature),
                                      max_tokens=min(int(max_tokens), 1024))
        def vlm(prompt, image=None, temperature=0.7, max_tokens=512):
            content = [{"type": "text", "text": str(prompt)}]
            if image:
                content.insert(0, {"type": "image_url", "image_url": {"url": "data:image/png;base64," + str(image)}})
            return self.fuzzer.chat([{"role": "user", "content": content}], "fuzzer_mutate",
                                    temperature=float(temperature), max_tokens=min(int(max_tokens), 1024))
        h = {"llm": llm, "similarity": lambda a, b: self.emb.similarity(str(a), str(b))}
        if self.vqa:
            h["vlm"] = vlm
        return h

    # ------------------------------------------------------------------ VQA images
    def img_b64(self, seed):
        if seed["id"] not in self._b64:
            self._b64[seed["id"]] = base64.b64encode(open(seed["image_path"], "rb").read()).decode()
        return self._b64[seed["id"]]

    def sbx_seed(self, seed):
        """Seed as sent to the sandbox (VQA: with the image as base64 PNG). --hide-answer: no gold label."""
        if not self.vqa:
            if self.a.hide_answer:
                return {"id": seed["id"], "subject": seed["subject"], "question": seed["question"],
                        "choices": list(seed["choices"])}
            return seed
        hidden = {"image_path", "answer", "answer_letter"} if self.a.hide_answer else {"image_path"}
        d = {k: v for k, v in seed.items() if k not in hidden}
        d["image"] = self.img_b64(seed)
        return d

    def save_image(self, b64):
        """Store a mutated image once (content-addressed) and return its path relative to the run dir."""
        h = hashlib.sha1(b64.encode()).hexdigest()[:16]
        rel = os.path.join("images", h + ".png")
        path = os.path.join(self.run, rel)
        if not os.path.exists(path):
            open(path, "wb").write(base64.b64decode(b64))
        return rel

    @staticmethod
    def _validate(m, n=4):
        L = "ABCDE"[:n]
        if not isinstance(m, dict) or not isinstance(m.get("question"), str) or not m["question"].strip():
            return "mutate() must return a dict with a non-empty 'question'"
        ch = m.get("choices")
        if not isinstance(ch, list) or len(ch) != n or not all(isinstance(c, str) for c in ch):
            return f"mutate() must return 'choices' = list of {n} strings"
        lm = m.get("label_map") or {x: x for x in L}
        if not isinstance(lm, dict) or sorted(lm.keys()) != list(L) or sorted(lm.values()) != list(L):
            return f"label_map must be a bijection on {L[0]}-{L[-1]}"
        m["label_map"] = lm
        if "hint" in m and not isinstance(m["hint"], str):
            return "'hint' must be a string"
        return None

    def test_one(self, pid, seed, rng_seed, code=None):
        ev = {"round": self.st["round"], "program": pid, "version": self.st["programs"][pid]["version"],
              "seed_id": seed["id"], "subject": seed["subject"], "gold": seed["answer"]}
        try:
            r = run_program(code or self.code(pid), self.sbx_seed(seed), rng_seed, self.handlers(), timeout=self.a.sandbox_timeout)
            ev["mutate_calls"] = len(r.get("rpc_calls", []))
            if r.get("error"):
                ev.update(status="budget" if "BudgetExhausted" in r["error"] else "error", error=r["error"][-600:])
                return ev
            m = r["mutated"]
            if err := self._validate(m, len(seed["choices"])):
                ev.update(status="error", error=err); return ev
            ev["mutated"] = {"question": m["question"], "choices": m["choices"], "label_map": m["label_map"]}
            if self.vqa:
                ev["mutated"]["hint"] = m.get("hint", seed.get("hint", ""))
                ev["mutated"]["image_changed"] = m["image"] != self.img_b64(seed)
            if r.get("check") is not True:
                ev.update(status="check_reject", check=r.get("check")); return ev
            if self.a.fixed_checks and (why := fixed_checks(seed, m)):
                ev.update(status="fixed_reject", check="fixed: " + why); return ev
            if self.vqa:
                url = "data:image/png;base64," + m["image"]
                msgs = vqa.messages(m["question"], m["choices"], url, ev["mutated"]["hint"])
            else:
                msgs = mcq.messages(m["question"], m["choices"])
            reply = self.target.chat(msgs, "target", temperature=0.0, max_tokens=1024)
            got = self.parse_answer(reply, m["choices"])
            expected = m["label_map"][seed["answer"]]
            if self.a.label_guard or self.a.hide_answer:   # deterministic oracle: the gold option's text decides the expected letter
                g = label_guard(seed, m)
                if g and g != expected:
                    ev["label_fixed"] = expected; expected = g
            ev.update(target_reply=reply, got=got, expected=expected)
            if got is None:
                ev["status"] = "unparsed"; return ev
            if got == expected:
                ev["status"] = "pass"; return ev
            ev["status"] = "violation"
            mech = self._meta(code)[1] if code else self.st["programs"][pid].get("mechanism", "")
            if self.vqa:
                ev["mutated"]["image"] = self.save_image(m["image"]) if ev["mutated"]["image_changed"] else None
                jm = P.judge_messages_vqa(seed, ev["mutated"], expected, got, reply, mech,
                                          "data:image/png;base64," + self.img_b64(seed), url)
                jt = self.judge.chat(jm, self.judge_role, temperature=0.0, max_tokens=4096 if self.a.judge_api else 800)
            else:
                jt = self.judge.prompt(P.judge_prompt(seed, m, expected, got, reply, mech), self.judge_role,
                                       temperature=0.0, max_tokens=4096 if self.a.judge_api else 400)
            ev["judge"] = P.parse_judge(jt)
            return ev
        except BudgetExhausted:
            ev.update(status="budget"); return ev
        except Exception as e:
            ev.update(status="infra_error", error=f"{type(e).__name__}: {e}"[:600]); return ev

    # ------------------------------------------------------------------ bookkeeping
    def record(self, ev):
        if ev["status"] in ("budget", "infra_error"):
            return
        p = self.st["programs"][ev["program"]]
        if ev["version"] != p["version"]:
            return
        s = p["stats"]

        def inc(k):
            s[k] = s.get(k, 0) + 1
        inc("tries")
        st = ev["status"]
        if st == "error":
            inc("errors")
        elif st in ("check_reject", "fixed_reject"):
            inc("check_reject")
            if st == "fixed_reject":
                inc("fixed_reject")
        else:
            inc("passed")
            if st == "unparsed":
                inc("unparsed")
            if st == "violation":
                inc("violations")
                v = ev["judge"]["verdict"]
                if v == "VALID":
                    inc("tp")
                    p.setdefault("tp_cases", []).append(self.case(ev, self.seed_index[ev["seed_id"]]))
                    p["tp_cases"] = p["tp_cases"][-5:]
                    self.cluster(ev)
                    if ev.get("novel"):
                        inc("novel")
                    self.st["recent_tp"] = (self.st.get("recent_tp", []) +
                                            [self.case(ev, self.seed_index[ev["seed_id"]])])[-20:]
                elif v == "INVALID":
                    inc("fp")

    def cluster(self, ev):
        """Online leader clustering of failure patterns -> distinct failures."""
        prog = self.st["programs"][ev["program"]]
        desc = ev["judge"]["pattern"].strip() or f"{prog['family']}: {ev['judge']['reason']}"
        e = self.emb.emb([desc])[0]
        best, bs = None, -1
        for c in self.st["clusters"]:
            sim = cos(e, c["emb"])
            if sim > bs:
                best, bs = c, sim
        if best is not None and bs >= self.a.cluster_threshold:
            best["count"] += 1
            ev.update(cluster=best["id"], novel=False)
        else:
            cid = len(self.st["clusters"])
            self.st["clusters"].append({"id": cid, "desc": desc, "emb": e, "count": 1,
                                        "round": self.st["round"], "program": ev["program"]})
            ev.update(cluster=cid, novel=True)

    @staticmethod
    def case(ev, seed):
        return {"seed_id": seed["id"], "orig_q": seed["question"], "orig_choices": seed["choices"], "mut_q": ev["mutated"]["question"],
                "mut_choices": ev["mutated"]["choices"], "reason": ev["judge"]["reason"],
                "image_changed": ev["mutated"].get("image_changed"),
                "pattern": ev["judge"].get("pattern", ""), "expected": ev["expected"], "got": ev["got"]}

    # ------------------------------------------------------------------ synthesis
    def synthesize(self, prompt, kind, parent, rng, replace_pid=None, commit=True):
        rec = {"round": self.st["round"], "kind": kind, "parent": parent}
        try:
            out = self.fuzzer.prompt(prompt, "fuzzer_synth", temperature=0.7, max_tokens=self.a.synth_max_tokens)
        except BudgetExhausted:
            return None
        code = P.extract_code(out)
        if code is None:
            rec.update(ok=False, why="no program in output", raw=out[-1500:])
            self.synth_f.write(json.dumps(rec) + "\n"); return None
        # smoke test: must run, and at least one of 3 seeds must pass its own check
        smoke = rng.sample(self.seeds, 3)
        results = [run_program(code, self.sbx_seed(s), rng.randrange(1 << 30), self.handlers(), timeout=self.a.sandbox_timeout)
                   for s in smoke]
        ok_runs = [r for r, s in zip(results, smoke)
                   if not r.get("error") and r.get("check") is True and not self._validate(r["mutated"], len(s["choices"]))]
        if not ok_runs:
            errs = [r.get("error") or r.get("check") for r in results]
            rec.update(ok=False, why="smoke test failed", errors=[str(e)[-300:] for e in errs], code=code)
            self.synth_f.write(json.dumps(rec) + "\n"); return None
        if not commit:   # caller tests the candidate before committing it
            rec.update(ok=True, pending=True); self.synth_f.write(json.dumps(rec) + "\n")
            return code
        pid = self._add_program(code, origin=kind, parent=parent, pid=replace_pid)
        rec.update(ok=True, pid=pid, version=self.st["programs"][pid]["version"])
        self.synth_f.write(json.dumps(rec) + "\n")
        print(f"  synth[{kind}] ok -> {pid} v{self.st['programs'][pid]['version']}", flush=True)
        return pid

    def test_repair(self, pid, code, fp_cases, tp_cases, rng):
        """Accept a repaired program only if, on fresh runs, it (1) no longer breaks the questions it broke
        (at most 1 of 3 still judged INVALID), (2) never crashes, (3) rejects at most half of its own mutations,
        and (4) still produces tests on questions where the old version found real failures."""
        jobs = [("fp", self.seed_index[c["seed_id"]]) for c in fp_cases if c.get("seed_id") in self.seed_index]
        jobs += [("tp", self.seed_index[c["seed_id"]]) for c in tp_cases if c.get("seed_id") in self.seed_index]
        jobs += [("rand", s) for s in rng.sample(self.seeds, 3)]
        seeds = [rng.randrange(1 << 30) for _ in jobs]
        with ThreadPoolExecutor(len(jobs)) as ex:
            evs = list(ex.map(lambda js: self.test_one(pid, js[0][1], js[1], code=code), zip(jobs, seeds)))
        kinds = [k for k, _ in jobs]
        n_fp = kinds.count("fp")
        broken = sum(1 for k, e in zip(kinds, evs) if k == "fp" and e["status"] == "violation"
                     and e.get("judge", {}).get("verdict") == "INVALID")
        crashes = sum(1 for e in evs if e["status"] in ("error", "infra_error"))
        rejects = sum(1 for e in evs if e["status"] in ("check_reject", "fixed_reject"))
        tp_ok = "tp" not in kinds or any(e["status"] in ("pass", "violation") for k, e in zip(kinds, evs) if k == "tp")
        budget_hit = any(e["status"] == "budget" for e in evs)
        ok = (not budget_hit and crashes == 0 and broken <= n_fp // 3 and rejects <= len(evs) // 2 and tp_ok)
        self.repair_f.write(json.dumps({"round": self.st["round"], "program": pid,
                                        "old_version": self.st["programs"][pid]["version"], "accepted": ok,
                                        "n_fp": n_fp, "still_broken": broken, "crashes": crashes, "rejects": rejects,
                                        "n": len(evs), "tp_ok": tp_ok, "code": code,
                                        "events": [dict(e, kind=k) for k, e in zip(kinds, evs)]}) + "\n")
        self.repair_f.flush()
        return ok

    @staticmethod
    def rate_str(s):
        return f"{s.get('tp', 0)}/{s.get('tries', 0)} ({100 * s.get('tp', 0) / max(s.get('tries', 0), 1):.1f}%)"

    def champion(self, min_tries=10):
        """Active program with the highest real-failure yield (needs at least one real failure)."""
        c = [k for k in self.active() if self.st["programs"][k]["stats"].get("tries", 0) >= min_tries
             and self.st["programs"][k]["stats"].get("tp", 0) > 0]
        return max(c, key=lambda k: self.st["programs"][k]["stats"]["tp"] / self.st["programs"][k]["stats"]["tries"]) if c else None

    def avg_new_rate(self):
        """Real-failure yield of fuzzer-written programs so far (all versions ever run)."""
        tp = tries = 0
        for p in self.st["programs"].values():
            if p.get("origin") == "seed":
                continue
            for s in [h["stats"] for h in p.get("history", [])] + [p["stats"]]:
                tp += s.get("tp", 0); tries += s.get("tries", 0)
        return f"{100 * tp / max(tries, 1):.1f}%"

    def feedback(self, evs, seed_by_id, rng):
        strat = self.a.strategy
        # FP -> tighten the program that produced it
        if strat in ("full", "no_tp"):
            fps = defaultdict(list)
            for ev in evs:
                if ev["status"] == "violation" and ev["judge"]["verdict"] == "INVALID":
                    fps[ev["program"]].append(self.case(ev, seed_by_id[ev["seed_id"]]))
            for pid, cases in sorted(fps.items(), key=lambda kv: -len(kv[1]))[: self.a.max_refine]:
                if self.budget.exhausted():
                    break
                tps = self.st["programs"][pid].get("tp_cases", [])[-2:]
                pr = (P.refine_prompt2(self.code(pid), cases[:3], tps) if self.a.rich_repair
                      else P.refine_prompt(self.code(pid), cases[:3]))
                if not self.a.test_repairs:
                    self.synthesize(pr, "refine", pid, rng, replace_pid=pid)
                    continue
                code = self.synthesize(pr, "refine", pid, rng, commit=False)
                if code is None:
                    continue
                if self.test_repair(pid, code, cases[:3], tps, rng):
                    self._add_program(code, origin="refine", parent=pid, pid=pid)
                    print(f"  repair accepted -> {pid} v{self.st['programs'][pid]['version']}", flush=True)
                else:
                    print(f"  repair rejected for {pid} (kept v{self.st['programs'][pid]['version']})", flush=True)
        # exploration: new programs
        if strat == "fixed" or self.st["round"] % self.a.gen_every != 0:
            return
        act = self.active()
        fams = sorted({self.st["programs"][k]["family"] for k in act})
        fam_tries = defaultdict(int)
        for k in act:
            fam_tries[self.st["programs"][k]["family"]] += self.st["programs"][k]["stats"].get("tries", 0)

        def diverse_example():   # avoid mode collapse: example from a least-explored family
            least = min(fam_tries[self.st["programs"][k]["family"]] for k in act)
            return rng.choice([k for k in act if fam_tries[self.st["programs"][k]["family"]] <= least + 10])
        lineage = None
        for _ in range(self.a.n_new):
            if self.budget.exhausted():
                break
            if strat == "inconsistency":
                best = max(act, key=self.score)
                s = self.st["programs"][best]["stats"]
                rate = f"{s.get('violations', 0)}/{max(s.get('passed', 0), 1)}"
                pr = P.EXPLORE_INCONSISTENCY.format(api=P.API, example=P.EXAMPLE_HEADER.format(code=self.code(best)), rate=rate,
                                                    families=", ".join(fams))
                parent = best
            elif self.a.bold and strat in ("full", "no_fp") and self.st.get("recent_tp") and _ % 2 == 0 \
                    and (champ := self.champion()):
                cs = self.st["programs"][champ]
                tps = "\n\n".join(P.tp_text(c) for c in cs.get("tp_cases", [])[-2:]) or "(none recorded)"
                pr = P.EXPLORE_EXPLOIT.format(api=P.API, example=P.EXAMPLE_HEADER.format(code=self.code(champ)),
                                              rate=self.rate_str(cs["stats"]), avg=self.avg_new_rate(), tps=tps)
                parent = champ
                lineage = self.st["programs"][champ].get("lineage", champ)   # variants share the champion's budget cap
            elif strat in ("full", "no_fp") and self.st.get("recent_tp"):
                best = diverse_example()
                known = "\n".join(f"- {c['desc']} ({c['count']} found)" for c in
                                  sorted(self.st["clusters"], key=lambda c: -c["count"])[:20])
                pr = P.EXPLORE_TP.format(api=P.API, example=P.EXAMPLE_HEADER.format(code=self.code(best)), known=known,
                                         tp=P.tp_text(rng.choice(self.st["recent_tp"])), families=", ".join(fams))
                if self.a.bold:
                    champ = self.champion()
                    pr = pr.replace("Give it a new FAMILY name.", P.STRENGTH_NOTE.format(
                        avg=self.avg_new_rate(), best=self.rate_str(self.st["programs"][champ]["stats"]) if champ else "more")
                        + "\nGive it a new FAMILY name.")
                parent = best
            else:   # no_tp, or full/no_fp before any TP exists
                parent = diverse_example()
                pr = P.EXPLORE_BLIND.format(api=P.API, example=P.EXAMPLE_HEADER.format(code=self.code(parent)),
                                            families=", ".join(fams))
            new = self.synthesize(pr, "explore", parent, rng)
            if new and lineage:
                self.st["programs"][list(self.st["programs"])[-1]]["lineage"] = lineage
            lineage = None
            if new:
                act = self.active(); fams = sorted({self.st["programs"][k]["family"] for k in act})
        # cap population: retire the lowest-scoring programs
        act = self.active()
        if self.a.bold:   # v3: retire programs that keep failing to find anything real
            for pid in act:
                st = self.st["programs"][pid]["stats"]
                if st.get("tries", 0) >= self.a.retire_after and st.get("tp", 0) == 0:
                    self.st["programs"][pid]["active"] = False
            self.keep_min_active()
            act = self.active()
        if len(act) > self.a.max_programs:
            for pid in sorted(act, key=self.score)[: len(act) - self.a.max_programs]:
                self.st["programs"][pid]["active"] = False

    # ------------------------------------------------------------------ run
    def save(self):
        tmp = self.state_path + ".tmp"
        json.dump(self.st, open(tmp, "w"))
        os.replace(tmp, self.state_path)

    def keep_min_active(self, k=3):
        """never retire everything: if fewer than k programs are active, bring back the best-scoring retired ones."""
        act = self.active()
        if len(act) >= k:
            return
        off = [p for p in self.st["programs"] if p not in act]
        for pid in sorted(off, key=self.score, reverse=True)[: k - len(act)]:
            self.st["programs"][pid]["active"] = True
            print(f"  revived {pid} (too few active programs)", flush=True)

    def cap_lineages(self, act, w):
        """v4: no program lineage (a program and its 'stronger variants') may get more than --max-share of the tests."""
        lin = [self.st["programs"][p].get("lineage", p) for p in act]
        for _ in range(3):
            tot = sum(w); share = defaultdict(float)
            for l, x in zip(lin, w):
                share[l] += x
            big = [l for l, x in share.items() if x > self.a.max_share * tot and len(share) > 1]
            if not big:
                break
            for l in big:
                rest = tot - share[l]
                f = self.a.max_share * rest / (1 - self.a.max_share) / share[l]
                w = [x * f if li == l else x for li, x in zip(lin, w)]
        return w

    def run_rounds(self):
        t_end = time.time() + self.a.deadline_min * 60
        pool = ThreadPoolExecutor(self.a.workers)
        while not self.budget.exhausted():
            if time.time() > t_end:
                print("deadline reached; state saved, resubmit to continue", flush=True)
                return 3
            rnd = self.st["round"]
            rng = random.Random(self.a.seed * 100003 + rnd)
            self.keep_min_active()
            act = self.active()
            w = [max(self.score(p), 0.01) for p in act]
            if self.a.max_share < 1:
                w = self.cap_lineages(act, w)
            jobs = []
            for _ in range(self.a.batch):
                pid = rng.choices(act, weights=w)[0]
                subj = rng.choice(sorted(self.by_subject))
                jobs.append((pid, rng.choice(self.by_subject[subj]), rng.randrange(1 << 30)))
            evs = list(pool.map(lambda j: self.test_one(*j), jobs))
            for ev in evs:
                self.record(ev)
            seed_by_id = {j[1]["id"]: j[1] for j in jobs}
            self.feedback(evs, seed_by_id, rng)
            for ev in evs:
                self.events_f.write(json.dumps(ev) + "\n")
            self.events_f.flush(); self.synth_f.flush()
            self.st["round"] = rnd + 1
            self.st["used"] = self.budget.used
            self.save()
            c = defaultdict(int)
            for ev in evs:
                c[ev["status"] if ev["status"] != "violation" else "viol_" + ev["judge"]["verdict"]] += 1
            print(f"round {rnd}: counted_calls={self.budget.counted}/{self.a.budget} active={len(act)} "
                  f"clusters={len(self.st['clusters'])} {dict(c)}", flush=True)
        self.st["done"] = True
        self.save()
        print("FUZZ_DONE", flush=True)
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=STRATEGIES, required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--baseline", required=True, help="baseline jsonl of the TARGET (for stable-correct seeds)")
    ap.add_argument("--task", choices=["mcq", "vqa"], default="mcq")
    ap.add_argument("--seeds", default=None, help="default: data/mmlu_seeds.jsonl (mcq) or data/scienceqa_img_seeds.jsonl (vqa)")
    ap.add_argument("--fuzzer-url", required=True)
    ap.add_argument("--target-url", required=True)
    ap.add_argument("--judge-url", required=True)
    ap.add_argument("--served-name", default="m")
    ap.add_argument("--budget", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--gen-every", type=int, default=2)
    ap.add_argument("--n-new", type=int, default=2)
    ap.add_argument("--max-refine", type=int, default=4)
    ap.add_argument("--max-programs", type=int, default=30)
    ap.add_argument("--cluster-threshold", type=float, default=0.80)
    ap.add_argument("--synth-max-tokens", type=int, default=3000)
    ap.add_argument("--sandbox-timeout", type=float, default=300)
    ap.add_argument("--deadline-min", type=float, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label-guard", action="store_true",
                    help="expected letter follows the gold option's text when it is still present (fixes bad label_maps)")
    ap.add_argument("--judge-api", default=None,
                    help="hosted judge model id (OpenAI-compatible; SOCLAAS_BASE_URL/SOCLAAS_API_KEY); not counted in budget")
    ap.add_argument("--hide-answer", action="store_true",
                    help="programs never see the gold label; expected letter = label guard (gold text) else label_map")
    ap.add_argument("--fixed-checks", action="store_true", help="harness checks the fuzzer cannot edit")
    ap.add_argument("--test-repairs", action="store_true", help="a repaired program must pass a re-test before it replaces the old one")
    ap.add_argument("--v4-checks", action="store_true", help="extra fixed checks: no added negation, question kept in one piece, options not inflated")
    ap.add_argument("--max-share", type=float, default=1.0, help="max share of tests per program lineage (v4: 0.25)")
    ap.add_argument("--bold", action="store_true",
                    help="v3: yield-based scoring, build stronger variants of the best program, strength feedback, retire harmless programs")
    ap.add_argument("--retire-after", type=int, default=40, help="--bold: retire a program after this many tries with no real failure")
    ap.add_argument("--rich-repair", action="store_true", help="repair prompt v2: judge reasons + the program's real failures")
    a = ap.parse_args()
    a.seeds = a.seeds or os.path.join(ROOT, "data", "scienceqa_img_seeds.jsonl" if a.task == "vqa" else "mmlu_seeds.jsonl")
    for u in {a.fuzzer_url, a.target_url} | (set() if a.judge_api else {a.judge_url}):
        wait_ready(u)
    sys.exit(Fuzzer(a).run_rounds())


if __name__ == "__main__":
    main()
