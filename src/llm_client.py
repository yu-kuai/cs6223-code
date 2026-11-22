"""Minimal thread-safe client for a vLLM OpenAI-compatible server, with per-role call accounting
and a shared call budget."""
import json, re, threading, time, urllib.request


class BudgetExhausted(Exception):
    pass


class Budget:
    """Counts LLM calls per role. Calls in `counted_roles` consume the shared budget."""

    def __init__(self, limit: int, counted_roles: set[str], used: dict | None = None):
        self.limit, self.counted_roles = limit, counted_roles
        self.used = dict(used or {})
        self.lock = threading.Lock()

    @property
    def counted(self) -> int:
        return sum(v for k, v in self.used.items() if k in self.counted_roles)

    def take(self, role: str):
        with self.lock:
            if role in self.counted_roles and self.counted >= self.limit:
                raise BudgetExhausted(f"budget {self.limit} exhausted")
            self.used[role] = self.used.get(role, 0) + 1

    def exhausted(self) -> bool:
        return self.counted >= self.limit


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


class LLM:
    def __init__(self, base_url: str, model: str, budget: Budget, timeout: float = 600):
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.model, self.budget, self.timeout = model, budget, timeout

    def chat(self, messages, role: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        self.budget.take(role)
        body = {"model": self.model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}}
        if temperature > 0:
            body.update(top_p=0.8, top_k=20)
        data = json.dumps(body).encode()
        for attempt in range(3):
            try:
                req = urllib.request.Request(self.url, data=data, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read())
                return strip_think(out["choices"][0]["message"]["content"] or "")
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))

    def prompt(self, text: str, role: str, **kw) -> str:
        return self.chat([{"role": "user", "content": text}], role, **kw)


class ApiLLM:
    """OpenAI-compatible hosted API (e.g. NUS SoClaaS) used as the in-loop judge.
    Reads SOCLAAS_BASE_URL / SOCLAAS_API_KEY from the environment (never logged). Calls are tallied under `role`
    in the shared Budget (pass a role that is NOT counted to keep them outside the call budget).
    Every raw reply is appended to `log_path` with the model name the server reports."""

    def __init__(self, model: str, budget: Budget, log_path: str, timeout: float = 300, max_parallel: int = 8):
        import os
        envf = os.path.expanduser("~/.config/soclaas/soclaas.env")
        if "SOCLAAS_API_KEY" not in os.environ and os.path.exists(envf):
            for line in open(envf):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.removeprefix("export ").split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        self.url = os.environ["SOCLAAS_BASE_URL"].rstrip("/") + "/chat/completions"
        self._key = os.environ["SOCLAAS_API_KEY"]
        self.model, self.budget, self.timeout = model, budget, timeout
        self.sem = threading.Semaphore(max_parallel)
        self.log_lock = threading.Lock()
        self.log_path = log_path

    def chat(self, messages, role: str, temperature: float = 0.0, max_tokens: int = 4096) -> str:
        self.budget.take(role)
        body = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        data = json.dumps(body).encode()
        headers = {"Content-Type": "application/json", "Authorization": "Bearer " + self._key}
        err = None
        for attempt in range(6):
            try:
                with self.sem:
                    req = urllib.request.Request(self.url, data=data, headers=headers)
                    with urllib.request.urlopen(req, timeout=self.timeout) as r:
                        out = json.loads(r.read())
                msg = out["choices"][0]["message"]
                content = msg.get("content") or ""
                with self.log_lock, open(self.log_path, "a") as f:
                    f.write(json.dumps({"t": time.time(), "requested": self.model, "served": out.get("model"),
                                        "usage": out.get("usage"), "prompt": messages[-1]["content"][-4000:]
                                        if isinstance(messages[-1]["content"], str) else None,
                                        "content": content, "reasoning": (msg.get("reasoning_content")
                                        or msg.get("reasoning") or "")[-4000:],
                                        "finish": out["choices"][0].get("finish_reason")}) + "\n")
                if not content.strip() and attempt < 5:
                    err = "empty content"; time.sleep(2 * (attempt + 1)); continue
                return strip_think(content)
            except Exception as e:   # 429 / 5xx / network: back off and retry
                err = e
                time.sleep(min(60, 3 * 2 ** attempt))
        raise RuntimeError(f"API judge failed after retries: {err}")

    def prompt(self, text: str, role: str, **kw) -> str:
        return self.chat([{"role": "user", "content": text}], role, **kw)


def wait_ready(base_url: str, timeout: float = 1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(base_url.rstrip("/") + "/health", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(f"server {base_url} not ready after {timeout}s")
