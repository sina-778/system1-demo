# Project context for Claude Code

## Who I am and how to help me
- I'm Sina, MSc Data Science student at BHT Berlin. I'm new to building with AI APIs, RAG and agents.
- Explain the basic idea/intuition first, then the details. I want to understand every line I run,
  because I'll present this live and put it on my GitHub.
- Keep code short, commented and readable. Prefer small changes and explain each one.

## What this project is
Course: **Enterprise Data Science** (BHT, WiSe 2026/27), group project on **"System 1" decision models**.
Group repo: https://github.com/RamezCh/Decision-Models-in-Agent-Architectures
(a 20-min group presentation, 4 speakers × 5 min plus Q&A; speaker 4 covers the production hybrid
architecture plus a live demo). This folder is my live demo.

**Core idea:** a normal LLM ("System 2") writes its answer token by token, which is slow and costly,
and its JSON can be broken. A decision model ("System 1") doesn't generate text. It answers typed
questions with probabilities in one fast pass. A confidence threshold decides when to trust it and
when to escalate to a big LLM.

The two System 1 models:
- **Jev** (TypeSafe AI, Sept 2026): paid API through OpenRouter. Calibrated confidence (trained with RLCD).
- **CLM-8B** (Stanford + NVIDIA): open weights (Apache 2.0), frozen Qwen3-8B plus a contrastive
  state/action head, and it caches action embeddings. Runs locally and for free via
  `llama-server --embeddings --pooling last` (port 8090) + `clm-serve` (port 8700, from the
  `contrastive-lm` pip package - install with `--no-deps`, since its `vllm` dependency has no
  macOS wheel and isn't needed when pointing `--emb-url` at llama.cpp instead). It's a **switch**,
  not a third column: `SYSTEM1_BACKEND=jev|clm` in `.env` picks which one `brains.system1_route()`
  calls. A true three-way side-by-side (Jev + CLM + LLM router together) is still open, see Next steps.

## The demo ("Hybrid Brain")
The user types a task. Jev routes it to the right GWDG model. In parallel, a GWDG LLM does the same
routing job by writing JSON, for comparison. A gatekeeper threshold escalates unsure cases to the biggest model.
- `brains.py`: all API logic (jev_route, clm_route, system1_route, llm_route, pick_model,
  answer_stream). SPECIALISTS dict = GWDG models.
- `check_setup.py`: 5-step connection test with ✅/❌ and fix hints. Run this first.
- `app.py`: Streamlit UI (`streamlit run app.py`).
- `.env`: my keys (OPENROUTER_API_KEY, GWDG_API_KEY, JEV_MODEL) plus SYSTEM1_BACKEND (jev|clm)
  and CLM_URL. **Never print, log or commit the keys.** Don't open `.env` unless I ask.
- `models/`, `.local/`: CLM-8B's GGUF weights and the llama.cpp binary (gitignored, multi-GB,
  downloaded straight from GitHub releases / Hugging Face - not via Homebrew, which isn't installed).

## API facts (verified from the official docs, Oct 2026)
**Jev via OpenRouter:** `POST https://openrouter.ai/api/alpha/decisions`, header `Authorization: Bearer <key>`.
Body: `model` (`~typesafe/jev-latest`, or pinned e.g. `typesafe/jev-1.13`), `state` (string/object),
`questions` = {name: question}. Question types:
- `noul`: `criteria` = {"true": "...", "false": "..."} → answer `{"noul": p}`
- `choice`: `criteria` = {label: description} → `{"choice", "confidence", "probabilities"}`
- `score`: `criteria` = [levels, lowest first] → `{"score", "confidence", "probabilities", "legend"}`
- (`box` = image bounding box, not used)
Response also has `usage.cost` (USD) and `model` (the exact snapshot). Output tokens are free.
Common errors: 401 bad key, 402 no credit, 404 unknown model id.

**GWDG SAIA:** OpenAI-compatible, base URL `https://chat-ai.academiccloud.de/v1`, Bearer key.
`/v1/models` lists models (the docs use POST). There are rate limits (the docs example shows ~10 requests/min,
see the `x-ratelimit-*` headers). Each demo click uses 2 GWDG calls. **The lineup rotates fast -
it had already changed within the same day as this file's last edit** - so always re-query
`/v1/models` instead of trusting this list. As of 27 Oct 2026 it returned 11 models: deepseek-v4-flash-0731,
eurollm-22b-instruct-2512, gemma-4-31b-it, glm-5.3-flash, meta-llama-3.1-8b-instruct,
qwen3-30b-a3b-instruct-2507, qwen3-coder-next, qwen3-omni-30b-a3b-instruct, qwen3.5-397b-a17b,
qwen3.6-35b-a3b, qwen3.8-27b. The response's `output` field marks which models support visible
reasoning (`["text", "thought"]` vs plain `["text"]`) - that's the current equivalent of the old
`-thinking-` suffix models, and how SPECIALISTS["reasoning"] was chosen this time.

## My machine
Apple Silicon MacBook Pro, macOS, VS Code, Python `.venv` in this folder, Streamlit, ChromaDB available.

## Status (27 Oct 2026)
- `check_setup.py` is fully green with real keys: GWDG connection, GWDG chat call, System 1
  decision call, and the LLM-router baseline all pass. SPECIALISTS/ESCALATION_MODEL were updated
  to match GWDG's current lineup (see API facts above).
- CLM-8B runs locally (SYSTEM1_BACKEND=clm in `.env` is the active setting) via llama-server
  (port 8090) + clm-serve (port 8700), both left running in the background after setup -
  restart with the commands in README.md "Run CLM-8B locally" if the Mac reboots.
- `app.py` now has two tabs, confirmed working by hand and via `AppTest`: **"How it works"**
  (one task, a collapsed step-by-step "Under the hood" timeline with the real request/response
  JSON - collapsed by default, click to expand) and **"Speed & reliability showdown"** (a
  shufflable/editable batch of tasks fired at System 1 in parallel vs. the LLM router one at a
  time - the "wow" moment: typically 10-30× faster, with live JSON failures and a wasted-tokens
  counter). `brains.py` falls back from CLM to Jev automatically if CLM's local servers aren't
  reachable (`_secret()` reads `.env` locally or `st.secrets` when deployed).
- Not yet pushed to GitHub or deployed - this folder still isn't a git repo. README.md has a
  full "Deploy to Streamlit Community Cloud" section (git init → GitHub → share.streamlit.io →
  paste `.streamlit/secrets.toml.example` filled in, with `SYSTEM1_BACKEND=jev`).

## Next steps (in order, only when I ask)
1. Push to GitHub and deploy on Streamlit Community Cloud, following README.md's Deploy section.
   Watch the OpenRouter balance after sharing the link - nothing in the app rate-limits reuse.
2. Add "replay mode": save real results to JSON so the demo works without Wi-Fi in class.
3. Optional: show the `thought` content from GWDG's reasoning-capable models in a collapsible expander.
4. Consider a true **three-way comparison** (Jev + CLM-8B + LLM router side by side) instead of
   the current either/or SYSTEM1_BACKEND switch, if the presentation wants to show both System 1
   models at once rather than just one vs. the LLM.
5. Later demo ideas: a calibration game (reliability diagram + Brier score, Jev vs CLM vs LLM
   self-reported confidence), a fair same-size comparison (CLM-8B vs meta-llama-3.1-8b-instruct),
   a RAG "bouncer", optional file/document upload via GWDG's `/documents` (Docling) endpoint for
   the showdown tab (not wired up yet - untested, would need its own validation pass).
