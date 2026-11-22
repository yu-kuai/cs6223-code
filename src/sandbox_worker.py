"""Runs ONE mutation program on ONE seed. Protocol (JSON lines):
  parent -> worker: {"code", "seed", "rng_seed"}
  worker -> parent: {"type":"rpc","fn":..,"kwargs":..}   parent -> worker: {"value":..} | {"error":..}
  worker -> parent: {"type":"result", ...}
The program's own prints go to stderr so they cannot corrupt the protocol."""
import builtins, json, os, random, sys, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import helpers  # noqa: E402

PROTO_OUT = sys.stdout
sys.stdout = sys.stderr
ALLOWED_IMPORTS = {"re", "math", "random", "string", "itertools", "collections", "json", "functools", "unicodedata", "textwrap"}
IMAGE_IMPORTS = {"io", "PIL", "numpy"}   # only for the VQA task (seed has an image)
BLOCKED_BUILTINS = {"open", "exec", "eval", "compile", "input", "breakpoint", "help", "exit", "quit", "globals", "vars", "memoryview"}


def send(obj):
    PROTO_OUT.write(json.dumps(obj) + "\n"); PROTO_OUT.flush()


def rpc(fn, **kwargs):
    send({"type": "rpc", "fn": fn, "kwargs": kwargs})
    reply = json.loads(sys.stdin.readline())
    if "error" in reply:
        raise RuntimeError(f"rpc {fn} failed: {reply['error']}")
    return reply["value"]


def safe_import(name, *a, **k):
    if name.split(".")[0] not in ALLOWED_IMPORTS | (IMAGE_IMPORTS if IMAGE_TASK else set()):
        raise ImportError(f"import of '{name}' is not allowed in mutation programs")
    return builtins.__import__(name, *a, **k)


IMAGE_TASK = False


def b64_to_img(b64):
    import base64, io
    from PIL import Image
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def img_to_b64(img):
    import base64, io
    buf = io.BytesIO(); img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def llm(prompt, temperature=0.7, max_tokens=512):
    return rpc("llm", prompt=prompt, temperature=temperature, max_tokens=max_tokens)


def main():
    global IMAGE_TASK
    req = json.loads(sys.stdin.readline())
    IMAGE_TASK = bool(req["seed"].get("image"))
    helpers._rpc = rpc
    safe_builtins = {k: v for k, v in vars(builtins).items() if k not in BLOCKED_BUILTINS}
    safe_builtins["__import__"] = safe_import
    ns = {"__builtins__": safe_builtins, "__name__": "mutation_program"}
    for name in helpers.EXPORTS:
        ns[name] = getattr(helpers, name)
    import importlib
    for mod in ALLOWED_IMPORTS:               # pre-import all allowed modules (forgiving of missing imports)
        ns[mod] = importlib.import_module(mod)
    if IMAGE_TASK:
        import io, numpy
        from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
        ns.update(io=io, np=numpy, numpy=numpy, Image=Image, ImageDraw=ImageDraw, ImageEnhance=ImageEnhance,
                  ImageFilter=ImageFilter, ImageFont=ImageFont, ImageOps=ImageOps)
    out = {"type": "result", "stage": "load"}
    try:
        exec(compile(req["code"], "<program>", "exec"), ns)
        for name in helpers.EXPORTS:          # programs may not shadow the provided helpers
            ns[name] = getattr(helpers, name)
        out["stage"] = "mutate"
        def fresh_seed():   # each call gets its own copy (programs may modify it in place)
            s = dict(req["seed"]); s["choices"] = list(s["choices"])
            if IMAGE_TASK:
                s["image"] = b64_to_img(req["seed"]["image"])
            return s
        mutated = ns["mutate"](fresh_seed(), llm, random.Random(req["rng_seed"]))
        if IMAGE_TASK and isinstance(mutated, dict):
            from PIL import Image
            mutated.setdefault("hint", req["seed"].get("hint", ""))
            img = mutated.get("image")
            if img is None:
                mutated["image"] = b64_to_img(req["seed"]["image"])
            elif not isinstance(img, Image.Image):
                raise TypeError("mutate() must return 'image' as a PIL.Image (or omit it to keep the original)")
        out["stage"] = "check"
        c = ns["check"](fresh_seed(), mutated)
        if IMAGE_TASK and isinstance(mutated, dict):
            mutated = dict(mutated); mutated["image"] = img_to_b64(mutated["image"])
        out["mutated"] = mutated
        out["check"] = True if c is True else (str(c) if c else "check returned falsy")
        out["stage"] = "done"
    except Exception:
        out["error"] = traceback.format_exc(limit=3)[-1500:]
    send(out)


if __name__ == "__main__":
    main()
