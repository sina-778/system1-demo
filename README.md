# Hybrid Brain: System 1 routes, System 2 answers

A live classroom demo of **System 1 decision models**. You type a task. Two routers decide which
GWDG supercomputer model should answer it:

| | System 1 — **Jev** (TypeSafe, via OpenRouter) or **CLM-8B** (local, free) | System 2 — **LLM as router** (GWDG) |
|---|---|---|
| How it decides | answers typed questions (`Choice`, `Noul`, `Score`) with probabilities | generates the decision as JSON text |
| What you see | decision time, probability bars, real cost | decision time, raw text, whether the JSON was usable |

A **gatekeeper** then trusts Jev's choice only if its confidence is above the threshold slider.
Otherwise it escalates to the biggest model (`mistral-large-3-675b`). The chosen GWDG model streams the answer.

```
task ──► Jev: which specialist? (Choice) · needs reasoning? (Noul) · difficulty? (Score)
              │
     confidence ≥ threshold ?
        yes ─► specialist model (coder / thinking / fast 8B / 70B)
        no  ─► mistral-large-3-675b (escalation)
```

## Files
- `brains.py`: all API calls (Jev, CLM, GWDG router, gatekeeper, answer streaming). Read this first.
- `check_setup.py`: tests every connection step by step and says what to fix.
- `app.py`: the Streamlit demo UI.
- `.env.example`: template for your API keys and the `SYSTEM1_BACKEND` switch (local use).
- `.streamlit/secrets.toml.example`: the same, in the format Streamlit Community Cloud wants
  (see "Deploy" below).
- `models/`, `.local/`: CLM-8B's local weights and llama.cpp binary (gitignored, see below).

## Setup (macOS, once)

**1. Get the two API keys**
- **OpenRouter (for Jev):** sign in at https://openrouter.ai, add a few dollars of credit
  (Jev costs fractions of a cent per call), then create a key at https://openrouter.ai/settings/keys.
- **GWDG SAIA:** if you don't have a key yet, book the "LLM Service" on the KISSKI site with the same
  email as your Academic Cloud account. The key arrives by email.

**2. Create the virtual environment and install packages** (in this folder, in the VS Code terminal)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**3. Add your keys**
```bash
cp .env.example .env
open -e .env        # paste both keys after the "=", save, close
```
Never commit `.env`. It's already in `.gitignore`.

**4. Test every connection**
```bash
python check_setup.py
```
You want five ✅ and "Everything works". If a step shows ❌, the message says what to fix
(wrong key, no OpenRouter credit, a GWDG model id that changed, …).

**5. Run the demo**
```bash
streamlit run app.py
```
It opens in your browser with two tabs: **"How it works"** walks through one task step by
step (good for explaining the mechanism); **"Speed & reliability showdown"** fires a batch of
tasks at once (good for the class's "wow" moment - see script below).

## Demo script (about 4 minutes)

**Tab 1 - How it works** (explain the mechanism, ~2 min)
1. *"What is the capital of Peru?"* System 1 picks `quick_fact` almost instantly. Open the
   "Under the hood" steps and point at step 2's raw JSON: real numbers, no words generated.
2. *Palindrome function* → `coding`. Show the probability bars: that's a real distribution,
   not a guess.
3. *Bat-and-ball puzzle* → `reasoning`, with a high **Noul** "needs reasoning" value.
4. *"Is it better?"* (vague): low confidence, so the gatekeeper **escalates** to the big model.
5. Drag the threshold slider up and re-run: more tasks escalate. That's the cost/quality dial
   a company would tune.

**Tab 2 - Speed & reliability showdown** (the punchline, ~1 min)
6. Hit **🔀 Shuffle** for a fresh random batch (or edit the list yourself, or let a
   classmate type their own), then **🏎️ Fire!** and just let it run - System 1's side fills
   up almost instantly while the LLM router's table fills in one row at a time, visibly slower.
7. Point at the scoreboard: a real speed multiple (typically 20-30×), how many of the LLM's
   8 answers actually came back usable (it's rarely all 8), and the output tokens it burned
   generating a decision - exactly the number a real deployment would be billed for.
8. The punchline: System 1 can do this because it never writes a word - it answers typed
   questions with probabilities in one pass. That's also why it can't produce broken output
   the way a text-generating model can.

## Run CLM-8B locally (free alternative to Jev)

[CLM-8B](https://github.com/Contrastive-LM/CLM) (Stanford + NVIDIA, Apache 2.0) is a frozen
Qwen3-8B encoder plus two small contrastive heads. It answers the same typed questions as Jev,
for free, on your own Mac, with no internet call. Setup (tested on an M5, 24 GB RAM):

**1. llama.cpp** serves the Qwen3-8B encoder in embedding mode. No Homebrew/cmake needed —
download the prebuilt macOS binary:
```bash
mkdir -p .local/llama.cpp && curl -L -o /tmp/llama.tar.gz \
  https://github.com/ggml-org/llama.cpp/releases/download/b11912/llama-b11912-bin-macos-arm64.tar.gz
tar -xzf /tmp/llama.tar.gz -C .local/llama.cpp
xattr -dr com.apple.quarantine .local/llama.cpp   # macOS blocks unsigned downloads otherwise
```

**2. The model weights** (Q8_0 quant, ~8 GB — good fidelity, 24 GB RAM handles it easily):
```bash
mkdir -p models && curl -L -o models/Qwen3-8B-Q8_0.gguf \
  https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q8_0.gguf
```

**3. Start the encoder** (`--pooling last` matches how CLM's heads were trained — they read the
last-token hidden state, not a mean-pooled one):
```bash
.local/llama.cpp/llama-b11912/llama-server -m models/Qwen3-8B-Q8_0.gguf \
  --embeddings --pooling last --port 8090
```

**4. Install `clm-serve`.** `contrastive-lm`'s `pyproject.toml` lists `vllm` as a hard dependency,
but vllm has never published a macOS wheel (checked against PyPI directly) and only matters for
its *in-process* encoder mode, which we don't use — we point it at llama.cpp instead via
`--emb-url`. So install everything except vllm:
```bash
pip install --no-deps contrastive-lm
pip install numpy requests torch fastapi uvicorn pyarrow
```

**5. Start CLM**, pointed at the llama.cpp encoder from step 3:
```bash
clm-serve --port 8700 --emb-url http://127.0.0.1:8090/v1/embeddings --no-ui
```

**6. Switch the demo over** — in `.env`, set:
```
SYSTEM1_BACKEND=clm
```
Then `python check_setup.py` — step 4/5 now calls CLM instead of Jev, for free.

Note: CLM's README implies its HTTP API infers a question's type (`Choice`/`Noul`/`Score`) from
the shape of `criteria`. In practice the live server replies `422 unknown question type None`
without an explicit `"type"` field — same requirement as Jev. Found by calling the real endpoint
and reading the response, not by trusting the docs; see `_routing_questions()` in `brains.py`.

## Deploy to Streamlit Community Cloud (share a link)

Free hosting at https://share.streamlit.io. Two things only matter here: **CLM-8B can't be
used** (it needs your Mac's `llama-server`/`clm-serve` on localhost, which a cloud server
can't reach - `brains.py` falls back to Jev automatically, but set it explicitly below so
every request isn't wasting time on a failed connection first), and **secrets go through
Streamlit's own system**, not your `.env` file (which isn't pushed to GitHub at all).

1. **Push this folder to a GitHub repo** (this folder isn't a git repo yet):
   ```bash
   git init && git add . && git commit -m "Hybrid Brain demo"
   ```
   Then create an empty repo at https://github.com/new (don't initialize it with a README),
   and push:
   ```bash
   git remote add origin https://github.com/<you>/<repo-name>.git
   git branch -M main && git push -u origin main
   ```
2. Go to https://share.streamlit.io → **New app** → pick that repo, branch `main`,
   main file `app.py`.
3. Before (or right after) it deploys, open **Settings → Secrets** and paste in the contents
   of `.streamlit/secrets.toml.example`, filled in with your real keys, with
   `SYSTEM1_BACKEND = "jev"`.
4. Deploy. You get a `*.streamlit.app` link - that's what you send your classmates.

**Before you send the link around:** your OpenRouter key has real credit on it, and anyone
with the link can click "Route it!" or "Fire!" as many times as they want. A few friends
poking at it costs fractions of a dollar, but if the link spreads further than you intended,
nothing in this app stops repeated use. Keep an eye on your OpenRouter balance, and revoke/
replace the key in Streamlit's Secrets if you ever want to cut it off without redeploying.

## Honest notes (good for Q&A)
- Jev's call goes over the internet; CLM's doesn't, which is most of its latency advantage here —
  a fair speed comparison would also need CLM served from a GPU, not a MacBook CPU.
- GWDG has rate limits (check the `x-ratelimit-*` headers). Each click uses 2 GWDG calls.
- Jev's `confidence` is calibrated by training (RLCD). CLM's is a softmax over contrastive
  similarity scores — calibrated differently, worth comparing side by side. The LLM's
  "confidence" (right column) is just text it wrote, not calibrated at all.
- `llama-server` and `clm-serve` keep running in the background once started; stop them with
  `pkill -f llama-server` / `pkill -f clm-serve` when you're done.
