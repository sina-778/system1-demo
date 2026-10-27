"""
brains.py - the two "brains" of the demo.

  System 1 = Jev (TypeSafe), reached through OpenRouter's Decisions API.
             It does NOT write text. It answers typed questions with probabilities.
  System 2 = big generative LLMs on the GWDG SAIA supercomputer (OpenAI-compatible API).

Every function returns a plain dict, so the Streamlit app (app.py) only has to display them.
"""
import json
import os
import time

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads the keys from the .env file in this folder (local machine only)


def _secret(name: str, default: str = "") -> str:
    """Read config from the environment first (.env locally), then from Streamlit's
    secrets manager (st.secrets, set via the app's dashboard when deployed to
    Streamlit Community Cloud - there's no .env file there). Safe to call even
    outside a Streamlit app (e.g. from check_setup.py) or with no secrets.toml at all."""
    val = os.getenv(name)
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(name, default)
    except Exception:
        return default


JEV_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_MODEL = _secret("JEV_MODEL", "~typesafe/jev-latest")
GWDG_URL = "https://chat-ai.academiccloud.de/v1"

# Which System 1 brain answers the routing questions: "jev" (OpenRouter, paid, works
# anywhere) or "clm" (CLM-8B running locally for free, see README "Run CLM-8B locally").
# CLM only works on a machine that's also running llama-server + clm-serve on localhost -
# system1_route() below falls back to Jev automatically if that's not reachable, which
# is exactly what happens on a cloud deployment (it has no access to your Mac's ports).
SYSTEM1_BACKEND = _secret("SYSTEM1_BACKEND", "jev").lower()
CLM_URL = _secret("CLM_URL", "http://127.0.0.1:8700/v1/systemone")

# The GWDG specialists that System 1 can route a task to.
# key = label Jev chooses from, value = (GWDG model id, description Jev reads)
SPECIALISTS = {
    "coding": ("qwen3-coder-next", "writing, fixing or explaining program code"),
    "reasoning": ("qwen3.8-27b", "math, logic puzzles, proofs, multi-step reasoning"),
    "quick_fact": ("meta-llama-3.1-8b-instruct", "a short factual answer, definition or translation"),
    "general": ("gemma-4-31b-it", "explanations, writing, advice and everything else"),
}
ESCALATION_MODEL = "qwen3.5-397b-a17b"  # the biggest model GWDG offers right now, used when System 1 is unsure
ROUTER_LLM = "meta-llama-3.1-8b-instruct"                # System 2 doing the routing job, for comparison

# Score levels, lowest to highest. The index Jev picks sets the answer length.
DIFFICULTY = ["trivial: a one-line answer is enough",
              "moderate: a short explanation or a small piece of code",
              "hard: needs several steps of reasoning or a long answer"]
MAX_TOKENS = [300, 900, 2000]

gwdg = OpenAI(base_url=GWDG_URL, api_key=_secret("GWDG_API_KEY", "missing"))


def _routing_questions() -> dict:
    """The three typed questions both System 1 backends answer, in ONE parallel pass.
    Verified against both real servers: both need an explicit "type" per question
    (Jev's docs say so; CLM's README implies it can be inferred, but the live
    server returned HTTP 422 without it - always test, don't trust docs blindly).
    """
    return {
        "specialist": {   # Choice: one label + a probability for every label
            "type": "choice",
            "instructions": "Which specialist model should handle this user task?",
            "criteria": {label: desc for label, (_, desc) in SPECIALISTS.items()},
        },
        "needs_reasoning": {   # Noul: probability that the statement is true
            "type": "noul",
            "instructions": "Does answering this task require careful step-by-step reasoning?",
            "criteria": {"true": "a correct answer needs several reasoning steps",
                         "false": "the answer can be given directly"},
        },
        "difficulty": {   # Score: position on an ordered rubric
            "type": "score",
            "instructions": "How difficult is this task to answer well?",
            "criteria": DIFFICULTY,
        },
    }


def jev_route(task: str) -> dict:
    """System 1: ask Jev the routing questions through OpenRouter (paid, needs internet)."""
    payload = {"model": JEV_MODEL, "state": {"user_task": task}, "questions": _routing_questions()}
    t0 = time.perf_counter()
    r = requests.post(
        JEV_URL,
        headers={"Authorization": f"Bearer {_secret('OPENROUTER_API_KEY')}"},
        json=payload,
        timeout=30,
    )
    ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    data = r.json()
    a = data["answers"]
    return {
        "label": a["specialist"]["choice"],
        "confidence": a["specialist"]["confidence"],
        "probabilities": a["specialist"]["probabilities"],
        "needs_reasoning": a["needs_reasoning"]["noul"],
        "difficulty": a["difficulty"]["score"],   # e.g. 1.4 on the 0..2 scale
        "ms": ms,
        "cost_usd": data.get("usage", {}).get("cost"),
        "served_by": data.get("model"),
        "request": payload,        # exactly what was sent - for the "under the hood" view
        "raw_response": data,      # exactly what came back
    }


def clm_route(task: str) -> dict:
    """System 1: ask the same questions to CLM-8B running locally (free, no internet).
    Needs `llama-server --embeddings --pooling last` + `clm-serve` running first,
    see README "Run CLM-8B locally". Same question shape as Jev, confirmed by
    calling the real /v1/systemone endpoint and reading its JSON back (not assumed)."""
    payload = {"model": "clm-latest", "state": task, "questions": _routing_questions()}
    t0 = time.perf_counter()
    r = requests.post(CLM_URL, json=payload, timeout=30)
    ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    data = r.json()
    a = data["answers"]
    return {
        "label": a["specialist"]["choice"],
        "confidence": a["specialist"]["confidence"],
        "probabilities": a["specialist"]["probabilities"],
        "needs_reasoning": a["needs_reasoning"]["noul"],
        "difficulty": a["difficulty"]["score"],
        "ms": ms,
        "cost_usd": 0.0,                 # local model - never billed
        "served_by": data.get("model"),  # "clm-latest"
        "request": payload,
        "raw_response": data,
    }


def system1_route(task: str) -> dict:
    """Pick the System 1 backend named by SYSTEM1_BACKEND ("jev" or "clm"). If CLM is
    selected but its local servers aren't reachable (e.g. this is running on a cloud
    deployment, not your Mac), fall back to Jev instead of crashing the demo."""
    if SYSTEM1_BACKEND == "clm":
        try:
            return clm_route(task)
        except requests.exceptions.ConnectionError:
            result = jev_route(task)
            result["fallback_note"] = ("CLM-8B isn't reachable from here (it only runs on "
                                        "your Mac) - used Jev instead.")
            return result
    return jev_route(task)


def llm_route(task: str) -> dict:
    """System 2 doing the same routing job: a generative LLM asked to reply in JSON."""
    labels = list(SPECIALISTS)
    prompt = (
        "You route user tasks to a specialist. Reply with ONLY this JSON and nothing else:\n"
        f'{{"specialist": one of {labels}, "confidence": number 0-1, '
        '"needs_reasoning": true or false, "difficulty": 0, 1 or 2}'
    )
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": task}]
    t0 = time.perf_counter()
    r = gwdg.chat.completions.create(model=ROUTER_LLM, temperature=0, messages=messages)
    ms = (time.perf_counter() - t0) * 1000
    raw = r.choices[0].message.content or ""
    # Be fair to the LLM: remove ```json fences, which models often add despite the instruction.
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        out = json.loads(cleaned)
        valid = out.get("specialist") in labels   # parsed, but is the label one we allowed?
    except (json.JSONDecodeError, AttributeError):
        out, valid = {}, False                    # broken JSON = unusable decision
    return {
        "label": out.get("specialist") if valid else None,
        "confidence": out.get("confidence") if valid else None,
        "valid": valid,
        "raw": raw,
        "ms": ms,
        "output_tokens": r.usage.completion_tokens if r.usage else None,
        "request": {"model": ROUTER_LLM, "messages": messages},  # the exact prompt sent
    }


def pick_model(route: dict, threshold: float):
    """The gatekeeper: trust System 1 only when its confidence clears the threshold."""
    if route["confidence"] >= threshold:
        return SPECIALISTS[route["label"]][0], False
    return ESCALATION_MODEL, True


def answer_stream(model: str, task: str, difficulty: float):
    """System 2 does the actual work: stream the answer from the chosen GWDG model."""
    max_tokens = MAX_TOKENS[min(2, max(0, round(difficulty)))]
    stream = gwdg.chat.completions.create(
        model=model, max_tokens=max_tokens, stream=True,
        messages=[{"role": "user", "content": task}],
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
