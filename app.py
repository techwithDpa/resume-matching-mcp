"""Streamlit chat UI for the matching agent.

  streamlit run app.py

Left: chat. Sidebar: current requirements, shortlist with verdicts, the state
machine diagram, and the latest report. Each assistant turn has an expander
showing the reasoning trace (graph nodes + tool calls) that produced it.
"""

from __future__ import annotations

import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import streamlit as st

import llm
from matching_agent import MatchingSession

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIAGRAM = os.path.join(BASE_DIR, "docs", "state_machine.png")
MCP_DIAGRAM = os.path.join(BASE_DIR, "docs", "mcp_architecture.png")
VERDICT_ICON = {"hire": "🟢", "maybe": "🟡", "no-hire": "🔴"}

st.set_page_config(page_title="Resume Matching Agent", page_icon="🧭", layout="wide")


def new_session() -> None:
    st.session_state.session = MatchingSession()
    st.session_state.chat = []   # [{"role", "content", "trace"}]


if "session" not in st.session_state:
    new_session()
session: MatchingSession = st.session_state.session

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🧭 Matching Agent")
    st.caption(f"LLM: {'offline heuristics' if llm.is_offline() else llm.model_name()}")
    st.caption("Files + database served by 2 MCP servers (stdio, JSON-RPC 2.0)")
    if st.button("New conversation", width="stretch"):
        new_session()
        st.rerun()
    state = session.state() if session.started else {}
    req = state.get("requirements")
    if req:
        st.subheader(f"Requirements v{req.get('version', 1)}")
        st.markdown(f"**Role:** {req.get('role_title', '—')}  \n"
                    f"**Must:** {', '.join(req.get('must_have', [])) or '—'}  \n"
                    f"**Nice:** {', '.join(req.get('nice_to_have', [])) or '—'}  \n"
                    f"**Min years:** {req.get('min_years', 0)}")
    if state.get("shortlist"):
        st.subheader("Shortlist")
        st.caption(f"Round 1: {state.get('candidates_scored')} resumes → top {len(state['shortlist'])}")
        st.dataframe(
            [{"#": c["rank"], "candidate": c["candidate_name"], "final": c["final_score"],
              "fit": c["fit_score"],
              "verdict": f"{VERDICT_ICON[c['verdict']['decision']]} {c['verdict']['decision']}"}
             for c in state["shortlist"]],
            hide_index=True, width="stretch")
    if state.get("report_md"):
        with st.expander(f"📄 Report ({state.get('report_path')})"):
            st.markdown(state["report_md"])
    if os.path.isfile(DIAGRAM):
        with st.expander("State machine"):
            st.image(DIAGRAM)
    if os.path.isfile(MCP_DIAGRAM):
        with st.expander("MCP architecture"):
            st.image(MCP_DIAGRAM)

# ------------------------------------------------------------------ chat
st.header("Resume Matching Agent")
if not st.session_state.chat:
    st.info("Start with a JD file such as `jobs/jd_backend_python.txt`, or ask "
            "*\"Find me candidates with React and 3+ years experience\"*. Then compare, "
            "ask *why* someone ranked higher, refine requirements, or say **done**.")

for turn in st.session_state.chat:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("trace"):
            with st.expander("Reasoning trace"):
                st.code("\n".join(turn["trace"]), language=None)

if session.finished:
    st.success("Session ended. Start a new conversation from the sidebar.")
elif prompt := st.chat_input("Message the agent"):
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        trace, replies = [], []
        with st.status("Working…", expanded=True) as status:
            for ev in session.send(prompt):
                if ev["type"] == "trace":
                    line = f"[{ev['node']}] {ev['text']}"
                    trace.append(line)
                    st.write(line)
                elif ev["type"] == "message":
                    replies.append(ev["content"])
                elif ev["type"] == "end":
                    replies.append("Session ended. 👋")
            status.update(label="Done", state="complete", expanded=False)
        content = "\n\n---\n\n".join(replies) or "_(no reply)_"
        st.markdown(content)
    st.session_state.chat.append({"role": "assistant", "content": content, "trace": trace})
    st.rerun()
