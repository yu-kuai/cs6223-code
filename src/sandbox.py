"""Parent side of the sandbox: spawn a worker subprocess with resource limits, serve its RPCs.
No network namespace is available on this cluster (unshare not permitted), so isolation =
import allow-list + restricted builtins + rlimits + timeout + throwaway cwd + empty env."""
import json, os, resource, select, subprocess, sys, tempfile, time

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox_worker.py")


def _cap(res, val):
    soft, hard = resource.getrlimit(res)
    if hard != resource.RLIM_INFINITY:
        val = min(val, hard)
    resource.setrlimit(res, (val, hard))       # lower soft limit only (raising hard needs root)


def _limits():
    _cap(resource.RLIMIT_AS, 2 << 30)          # 2 GB memory
    _cap(resource.RLIMIT_CPU, 30)              # 30 s CPU
    _cap(resource.RLIMIT_FSIZE, 0)             # cannot write files


def run_program(code: str, seed: dict, rng_seed: int, rpc_handlers: dict, timeout: float = 300.0) -> dict:
    """rpc_handlers: {"llm": fn(**kw)->str, "similarity": fn(a,b)->float}. Returns the worker's result dict
    (keys: stage, mutated?, check?, error?) plus 'rpc_calls'."""
    with tempfile.TemporaryDirectory(prefix="mutsbx_") as tmp:
        env = {"PATH": "/usr/bin:/bin", "HOME": tmp, "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1"}
        p = subprocess.Popen([sys.executable, "-I", "-B", WORKER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, cwd=tmp, env=env, text=True, preexec_fn=_limits,
                             start_new_session=True)
        deadline = time.time() + timeout
        calls = []
        try:
            p.stdin.write(json.dumps({"code": code, "seed": seed, "rng_seed": rng_seed}) + "\n"); p.stdin.flush()
            while True:
                left = deadline - time.time()
                if left <= 0:
                    return {"stage": "timeout", "error": f"timeout after {timeout}s", "rpc_calls": calls}
                r, _, _ = select.select([p.stdout], [], [], left)
                if not r:
                    continue
                line = p.stdout.readline()
                if not line:
                    return {"stage": "crash", "error": f"worker exited rc={p.poll()}", "rpc_calls": calls}
                msg = json.loads(line)
                if msg.get("type") == "result":
                    msg.pop("type"); msg["rpc_calls"] = calls
                    return msg
                fn = msg.get("fn")
                try:
                    if fn not in rpc_handlers:
                        raise ValueError(f"unknown rpc {fn}")
                    val = rpc_handlers[fn](**msg.get("kwargs", {}))
                    calls.append(fn)
                    reply = {"value": val}
                except Exception as e:  # budget exhausted, server error, bad args...
                    reply = {"error": f"{type(e).__name__}: {e}"}
                p.stdin.write(json.dumps(reply) + "\n"); p.stdin.flush()
        finally:
            try:
                os.killpg(p.pid, 9)
            except Exception:
                pass
            p.wait()
