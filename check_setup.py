"""
check_setup.py - run this FIRST:   python check_setup.py

It tests every connection the demo needs, one step at a time,
and tells you exactly what to fix if a step fails.
"""
import os
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv()
import brains  # noqa: E402  (import after load_dotenv so brains sees the keys)

ok_all = True


def step(title, fn):
    global ok_all
    print(f"\n▶ {title}")
    try:
        fn()
    except Exception as e:  # show the error in plain words, keep going
        ok_all = False
        print(f"  ❌ FAILED: {type(e).__name__}: {e}")


def check_keys():
    needed = ["GWDG_API_KEY"]
    if brains.SYSTEM1_BACKEND == "jev":  # CLM runs locally, no key needed for it
        needed.append("OPENROUTER_API_KEY")
    for name in needed:
        if not os.getenv(name):
            raise RuntimeError(f"{name} is empty. Open the .env file and paste your key after {name}=")
        print(f"  ✅ {name} found ({os.getenv(name)[:6]}…)")


def check_gwdg_models():
    headers = {"Authorization": f"Bearer {os.getenv('GWDG_API_KEY')}"}
    r = requests.post(f"{brains.GWDG_URL}/models", headers=headers, timeout=30)
    if r.status_code == 405:  # some deployments only allow GET here
        r = requests.get(f"{brains.GWDG_URL}/models", headers=headers, timeout=30)
    if r.status_code == 401:
        raise RuntimeError("GWDG rejected the key (401). Check GWDG_API_KEY in .env.")
    r.raise_for_status()
    available = {m["id"] for m in r.json()["data"]}
    print(f"  ✅ GWDG answered: {len(available)} models available")
    needed = [m for m, _ in brains.SPECIALISTS.values()] + [brains.ESCALATION_MODEL, brains.ROUTER_LLM]
    missing = [m for m in dict.fromkeys(needed) if m not in available]
    for m in dict.fromkeys(needed):
        print(f"     {'✅' if m not in missing else '❌'} {m}")
    if missing:
        raise RuntimeError(f"these model ids are not offered right now: {missing}. "
                           "Pick a replacement from the list above and edit SPECIALISTS in brains.py.")


def check_gwdg_chat():
    t0 = time.perf_counter()
    r = brains.gwdg.chat.completions.create(
        model=brains.ROUTER_LLM, max_tokens=20,
        messages=[{"role": "user", "content": "Reply with the single word: ready"}])
    print(f"  ✅ {brains.ROUTER_LLM} replied {r.choices[0].message.content!r} "
          f"in {(time.perf_counter() - t0) * 1000:.0f} ms")


def check_jev():
    try:
        out = brains.jev_route("Write a Python function that reverses a list.")
    except requests.HTTPError as e:
        code = e.response.status_code
        hint = {401: "OpenRouter rejected the key - check OPENROUTER_API_KEY.",
                402: "No credit on your OpenRouter account - add a few dollars at openrouter.ai/credits.",
                404: f"Model '{brains.JEV_MODEL}' not found - set JEV_MODEL=typesafe/jev-1.13 in .env."
                }.get(code, e.response.text[:300])
        raise RuntimeError(f"HTTP {code}: {hint}") from None
    print(f"  ✅ Jev ({out['served_by']}) chose '{out['label']}' "
          f"with confidence {out['confidence']:.2f} in {out['ms']:.0f} ms, cost ${out['cost_usd']}")
    print(f"     needs_reasoning (Noul) = {out['needs_reasoning']:.2f}, difficulty (Score) = {out['difficulty']:.2f}")


def check_clm():
    try:
        r = requests.get("http://127.0.0.1:8090/health", timeout=5)
        r.raise_for_status()
    except requests.RequestException:
        raise RuntimeError("llama-server not reachable on :8090. Start it first: "
                            ".local/llama.cpp/llama-b11912/llama-server -m models/Qwen3-8B-Q8_0.gguf "
                            "--embeddings --pooling last --port 8090") from None
    try:
        r = requests.get("http://127.0.0.1:8700/health", timeout=5)
        r.raise_for_status()
    except requests.RequestException:
        raise RuntimeError("clm-serve not reachable on :8700. Start it: "
                            "clm-serve --port 8700 --emb-url http://127.0.0.1:8090/v1/embeddings") from None
    out = brains.clm_route("Write a Python function that reverses a list.")
    print(f"  ✅ CLM ({out['served_by']}) chose '{out['label']}' "
          f"with confidence {out['confidence']:.2f} in {out['ms']:.0f} ms, cost: free (local)")
    print(f"     needs_reasoning (Noul) = {out['needs_reasoning']:.2f}, difficulty (Score) = {out['difficulty']:.2f}")


def check_llm_router():
    out = brains.llm_route("Write a Python function that reverses a list.")
    print(f"  ✅ LLM router answered in {out['ms']:.0f} ms, valid JSON decision: {out['valid']}")
    print(f"     raw output: {out['raw'][:120]!r}")


step("1/5  API keys in .env", check_keys)
step("2/5  GWDG connection + model list", check_gwdg_models)
step("3/5  GWDG chat call", check_gwdg_chat)
step(f"4/5  System 1 decision call ({brains.SYSTEM1_BACKEND})",
     check_jev if brains.SYSTEM1_BACKEND == "jev" else check_clm)
step("5/5  LLM-as-router baseline", check_llm_router)

print("\n" + ("🎉 Everything works. Start the demo with:  streamlit run app.py"
              if ok_all else "⚠️  Fix the ❌ steps above, then run this script again."))
sys.exit(0 if ok_all else 1)
