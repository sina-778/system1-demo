"""
app.py - the live demo.   Start with:  streamlit run app.py

Two tabs, two different jobs:
  Tab 1 "How it works" - one task, walked through step by step. Good for explaining
         the mechanism: System 1 (Jev or local CLM-8B) answers typed questions with
         probabilities, no text generated; System 2 (an LLM) has to write its decision
         as text instead, which can come out broken.
  Tab 2 "Speed & reliability showdown" - a batch of tasks fired at once. Good for the
         "wow" moment: System 1 finishes the whole batch almost instantly, System 2
         visibly grinds through it one at a time and sometimes breaks.

Everything here is a real network call - nothing is simulated.
"""
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st

import brains

st.set_page_config(page_title="System 1 vs System 2 router", layout="wide")
st.title("🧠 Hybrid Brain: System 1 routes, System 2 answers")

BACKEND = brains.SYSTEM1_BACKEND.upper()

if "history" not in st.session_state:
    st.session_state.history = []

threshold = st.sidebar.slider("Confidence threshold (gatekeeper)", 0.0, 1.0, 0.70, 0.05)
st.sidebar.caption(f"Below this, the task escalates to **{brains.ESCALATION_MODEL}**.")

tab1, tab2 = st.tabs([
    "How it works (one task)",
    "Speed & reliability showdown (many tasks)",
])

# ============================================================================
# TAB 1 - one task, explained step by step.
# ============================================================================
with tab1:
    EXAMPLES = [
        "What is the capital of Peru?",
        "Write a Python function that checks whether a string is a palindrome.",
        "A bat and a ball cost 1.10 euros. The bat costs 1 euro more than the ball. "
        "What does the ball cost?",
        "Give me three tips for a 5-minute class presentation.",
        "Is it better?",  # deliberately vague -> should get low confidence
    ]
    example = st.radio("Example tasks", EXAMPLES, index=None, horizontal=False)
    task = st.text_area("Task", value=example or "", height=90)
    go = st.button("Route it!", type="primary")

    if go and task.strip():
        # Run both routers at the same time so neither waits for the other.
        with ThreadPoolExecutor() as pool:
            jev_future = pool.submit(brains.system1_route, task)
            llm_future = pool.submit(brains.llm_route, task)
            jev, llm = jev_future.result(), llm_future.result()

        model, escalated = brains.pick_model(jev, threshold)

        # --- "Under the hood": the exact sequence of real network calls, with the real JSON. ---
        # Consecutive type="step" statuses render as one connected timeline.
        # Collapsed by default (expanded=False) - click any step to open it. That keeps
        # the page approachable for first-time visitors instead of dumping raw JSON on them.
        st.subheader("Under the hood", icon=":material/manufacturing:", anchor=False)
        st.caption("Five steps, click to expand any of them - every box is the literal "
                   "request/response of a real API call, nothing is simulated.")
        if note := jev.get("fallback_note"):
            st.info(note, icon=":material/info:")

        with st.status("1 · Your task is sent, as plain text, to both routers at the same time",
                        type="step", state="complete"):
            st.write(f"*“{task}”*")

        with st.status(f"2 · {BACKEND} answers 3 typed questions in ONE pass - it writes no "
                        "words, only probabilities", type="step", state="complete"):
            c1, c2 = st.columns(2)
            c1.caption(f"Sent to {BACKEND}")
            c1.json(jev["request"])
            c2.caption("Received back")
            c2.json(jev["raw_response"])

        with st.status(f"3 · For comparison, {brains.ROUTER_LLM} is asked the same thing, but "
                        "has to generate the decision as text", type="step", state="complete"):
            c1, c2 = st.columns(2)
            c1.caption(f"Sent to {brains.ROUTER_LLM}")
            c1.json(llm["request"])
            c2.caption("Received back (raw text, not structured)")
            c2.code(llm["raw"], language="json")
            if not llm["valid"]:
                c2.caption(":material/error: Could not be parsed as the requested decision JSON.")

        with st.status(f"4 · Gatekeeper check: is {BACKEND}'s confidence ({jev['confidence']:.2f}) "
                        f"≥ your threshold ({threshold:.2f})?",
                        type="step", state="complete"):
            if escalated:
                st.write(f"**No** → {BACKEND} isn't trusted here, so the task **escalates** "
                         f"to the biggest model instead of the specialist it picked.")
            else:
                st.write(f"**Yes** → {BACKEND} is trusted, so the task goes straight to the "
                         f"specialist it picked - no escalation needed.")

        with st.status(f"5 · Routing to **{model}**"
                        + (" (the escalation model)" if escalated else " (the chosen specialist)"),
                        type="step", state="complete"):
            pass  # a step with no content closes the timeline cleanly

        st.divider()

        # --- Side-by-side comparison: the friendly summary view of the same two calls above. ---
        st.subheader("Side-by-side comparison", icon=":material/compare_arrows:", anchor=False)
        left, right = st.columns(2)
        with left:
            st.markdown(f":material/bolt: **System 1: {BACKEND}**")
            st.metric("Decision time", f"{jev['ms']:.0f} ms")
            st.write(f"**Choice →** `{jev['label']}` (confidence {jev['confidence']:.2f})")
            st.bar_chart(jev["probabilities"], horizontal=True, height=180,
                         alt=f"Probability {jev['label']} should handle this task, vs. "
                             "the other specialists")
            st.write(f"**Noul** needs reasoning: {jev['needs_reasoning']:.2f}  ·  "
                     f"**Score** difficulty: {jev['difficulty']:.2f} / 2")
            cost = f"${jev['cost_usd']}" if jev["cost_usd"] else "free (local)"
            st.caption(f"cost {cost} · {jev['served_by']}")
        with right:
            st.markdown(f":material/hourglass: **System 2: {brains.ROUTER_LLM} as router**")
            st.metric("Decision time", f"{llm['ms']:.0f} ms",
                      delta=f"{llm['ms'] - jev['ms']:+.0f} ms vs {BACKEND}", delta_color="inverse")
            if llm["valid"]:
                st.write(f"**Decision →** `{llm['label']}` (self-reported confidence {llm['confidence']})")
            else:
                st.error("Could not use this output as a decision (broken JSON or unknown label).")
            st.caption(f"{llm['output_tokens']} output tokens generated just to make a decision")

        st.divider()
        st.subheader(f"Final answer, streamed live from {model}", icon=":material/chat:", anchor=False)
        st.write_stream(brains.answer_stream(model, task, jev["difficulty"]))

        st.session_state.history.append({
            "task": task[:50], "jev_choice": jev["label"], "jev_conf": round(jev["confidence"], 2),
            "jev_ms": round(jev["ms"]), "llm_choice": llm["label"], "llm_valid": llm["valid"],
            "llm_ms": round(llm["ms"]), "answered_by": model,
        })
        st.subheader("Session history", icon=":material/history:", anchor=False)
        st.dataframe(st.session_state.history, width="stretch")

# ============================================================================
# TAB 2 - the "wow" moment: fire a batch of tasks and watch the two systems race.
# ============================================================================
with tab2:
    BATCH_SIZE = 8
    MAX_BATCH = 10  # hard cap - this app is meant to be shared, keep each run short & cheap
    TASK_POOL = [
        "What is the capital of Peru?", "What is the capital of Japan?",
        "Write a function that reverses a list.", "Write a function that checks if a number is prime.",
        "Write a regex that matches an email address.", "What's 15% of 80?", "What's 7 times 8?",
        "Translate 'good morning' into French.", "Give a synonym for 'happy'.",
        "What year did the Berlin Wall fall?", "Who wrote Romeo and Juliet?",
        "What's the boiling point of water in Celsius?", "What's the chemical symbol for gold?",
        "Name the largest planet in the solar system.", "How many continents are there?",
        "Convert 10 kilometers to miles.", "What does HTTP stand for?",
        "A train leaves at 3pm going 60km/h. How far has it gone by 5pm?",
        "Give three tips for better sleep.", "Explain recursion in one sentence.",
        "Is it better?",  # deliberately vague - a good one to see fail or get escalated
        "Write a one-line Python lambda that doubles a number.",
    ]

    def _reshuffle():
        st.session_state.batch_text = "\n".join(random.sample(TASK_POOL, BATCH_SIZE))

    if "batch_text" not in st.session_state:
        _reshuffle()

    st.write(
        f"{BACKEND} doesn't generate text, so it can answer a whole batch of tasks in "
        "parallel. The LLM router has to write out an actual decision for each one, so it "
        "answers one at a time - and its text can come back broken."
    )
    st.text_area(f"Tasks to fire (one per line, up to {MAX_BATCH} - edit freely, or shuffle "
                 "for a new random set)", key="batch_text", height=180)
    col_shuffle, col_fire = st.columns([1, 2], vertical_alignment="bottom")
    col_shuffle.button("🔀 Shuffle", on_click=_reshuffle)
    fire = col_fire.button("🏎️ Fire!", type="primary")

    if fire:
        BATCH = [line.strip() for line in st.session_state.batch_text.splitlines() if line.strip()]
        if len(BATCH) > MAX_BATCH:
            st.caption(f":material/info: Using the first {MAX_BATCH} lines (kept short so this "
                       "stays quick and cheap to run for everyone).")
            BATCH = BATCH[:MAX_BATCH]
        if len(BATCH) < 2:
            st.warning("Add at least 2 tasks (one per line) to run the showdown.")
            st.stop()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f":material/bolt: **{BACKEND} (System 1) - all at once**")
            s1_bar = st.progress(0.0, text=f"0 / {len(BATCH)} decisions made")
            s1_rows = st.empty()
        with col2:
            st.markdown(f":material/hourglass: **{brains.ROUTER_LLM} (System 2) - one at a time**")
            s2_bar = st.progress(0.0, text=f"0 / {len(BATCH)} decisions made")
            s2_rows = st.empty()

        # System 1: every question is just a lookup of probabilities, so firing all of
        # them in parallel is cheap and realistic - this is the whole point of System 1.
        results1 = []
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(BATCH)) as pool:
            futures = {pool.submit(brains.system1_route, t): t for t in BATCH}
            for fut in as_completed(futures):
                t = futures[fut]
                r = fut.result()
                results1.append({"task": t[:40], "choice": r["label"], "ms": round(r["ms"])})
                s1_bar.progress(len(results1) / len(BATCH),
                                text=f"{len(results1)} / {len(BATCH)} decisions made")
                s1_rows.dataframe(results1, width="stretch", height=240)
        s1_total_ms = (time.perf_counter() - t0) * 1000

        # System 2: writing text is real, metered compute, and GWDG rate-limits it - so
        # tasks go one at a time, just like a real deployment would have to throttle them.
        results2 = []
        t0 = time.perf_counter()
        for t in BATCH:
            r = brains.llm_route(t)
            results2.append({"task": t[:40], "choice": r["label"] or "⚠️ broken",
                              "ms": round(r["ms"]), "output_tokens": r["output_tokens"]})
            s2_bar.progress(len(results2) / len(BATCH),
                            text=f"{len(results2)} / {len(BATCH)} decisions made")
            s2_rows.dataframe(results2, width="stretch", height=240)
        s2_total_ms = (time.perf_counter() - t0) * 1000

        broken = sum(1 for r in results2 if r["choice"] == "⚠️ broken")
        tokens_burned = sum(r["output_tokens"] or 0 for r in results2)

        st.divider()
        st.subheader("The scoreboard", icon=":material/scoreboard:", anchor=False)
        m1, m2, m3 = st.columns(3)
        m1.metric("Speed", f"{s2_total_ms / s1_total_ms:.0f}× faster",
                  help=f"{BACKEND}: {s1_total_ms:.0f} ms total · LLM router: {s2_total_ms:.0f} ms total")
        m2.metric("Reliability", f"{len(BATCH) - broken} / {len(BATCH)} usable",
                  help=f"{BACKEND} can't produce broken output - it never writes text. "
                       f"The LLM router's text came back unusable {broken} time(s) this run.")
        m3.metric("Tokens burned just to decide", f"{tokens_burned} vs. 0",
                  help=f"The LLM router generated {tokens_burned} output tokens total - tokens "
                       f"a real deployment pays for - just to produce a decision. {BACKEND} "
                       "generated zero, because it answers with numbers, not words.")
        st.caption(
            f"{BACKEND} handled all {len(BATCH)} tasks **in parallel** in "
            f"{s1_total_ms:.0f} ms total. The LLM router took {s2_total_ms:.0f} ms, "
            f"**one task at a time**, because writing text is real work a GPU has to do - "
            f"that's also why production systems rate-limit it."
        )
