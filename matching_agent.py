"""LangGraph resume matching agent (Milestone 3).

State machine:

  START -> parse_jd -> extract_requirements -> search_resumes (round 1)
        -> rank_candidates (round 2 deep analysis + round 3 hire/no-hire)
        -> generate_report -> human_feedback --+--> END
                                  ^            |
                                  |            +--> refine_requirements -> search_resumes
                                  |            +--> parse_jd (new search)
                                  +-- chat_agent <-+  (questions: compare / why / interview Qs / files)

human_feedback pauses the graph with `interrupt()`; the CLI / Streamlit UI
resumes it with the user's next message (`Command(resume=...)`). A MemorySaver
checkpointer keeps the full state per conversation thread.

Usage (programmatic):
    session = MatchingSession()
    for event in session.send("jobs/jd_ml_engineer.txt"):
        print(event)
    for event in session.send("Compare the top 3 matches side by side"):
        ...
"""

from __future__ import annotations

import json
import operator
import os
import re
import uuid
from datetime import datetime
from typing import Annotated, Iterator, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

import agent_tools as T
import llm
import mcp_bridge

ROUND1_TOP_K = 10           # initial screen: top 10 of all resumes
MAX_TOOL_STEPS = 8          # tool-calling iterations per chat turn
HISTORY_WINDOW = 12         # past messages given to the chat agent


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]   # conversation history
    query: str                       # latest user input
    jd_text: str
    jd_source: str                   # file path or "chat"
    requirements: dict               # current must-have / nice-to-have / min_years
    requirements_history: list[dict] # every version, for explaining re-ranks
    requirement_changes: list[str]   # what the last refinement changed
    candidates_scored: int
    round1: list[dict]               # initial screen (top 10)
    round2: dict                     # candidate_id -> deep analysis
    round3: dict                     # candidate_id -> hire / maybe / no-hire
    shortlist: list[dict]            # merged, ranked view with reasoning
    previous_shortlist: list[dict]   # ranking before the last refinement
    rank_changes: list[dict]
    report_md: str
    report_path: str
    intent: str                      # refine | new_search | question | end
    trace: Annotated[list[str], operator.add]   # reasoning trace (node + tool log)


def _emit(node: str, text: str, lines: list[str]) -> None:
    """Record a trace line and stream it live to the UI (custom stream mode)."""
    lines.append(f"[{node}] {text}")
    try:
        get_stream_writer()({"node": node, "text": text})
    except Exception:   # called outside a graph run (e.g. unit tests)
        pass


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
_JD_REF = re.compile(r"((?:[\w./-]+/)?jd_[\w-]+(?:\.txt)?|[\w./-]+\.(?:txt|pdf|docx))")


def parse_jd(state: AgentState) -> dict:
    """Load the JD from a referenced file, or treat the message itself as the JD
    (natural-language searches like 'Find me React devs with 3+ years')."""
    lines: list[str] = []
    q = state["query"].strip()
    jd_text, source = q, "chat"
    m = _JD_REF.search(q)
    if m:
        ref = m.group(1)
        for rel in (ref, os.path.join("jobs", ref), os.path.join("jobs", ref + ".txt")):
            res = mcp_bridge.call("read_file", filepath=rel)
            if isinstance(res, dict) and res.get("success"):
                jd_text, source = res["content"], rel
                break
    _emit("parse_jd", f"JD source: {source} ({len(jd_text.split())} words)", lines)
    return {"jd_text": jd_text, "jd_source": source, "previous_shortlist": [],
            "requirements_history": [], "requirement_changes": [], "rank_changes": [],
            "trace": lines}


def extract_requirements_node(state: AgentState) -> dict:
    lines: list[str] = []
    req = T.extract_requirements(state["jd_text"])
    req["version"] = 1
    _emit("extract_requirements",
          f"[{req['source']}] must-have={req['must_have']} nice-to-have={req['nice_to_have']} "
          f"min_years={req['min_years']}", lines)
    return {"requirements": req, "requirements_history": [req], "trace": lines}


def search_resumes(state: AgentState) -> dict:
    """Round 1 — initial screen over every indexed resume (hybrid RAG score)."""
    lines: list[str] = []
    out = T.round1_search(state["requirements"], state.get("jd_text", ""), top_k=ROUND1_TOP_K)
    r1 = out["top_matches"]
    qualified = sum(r["meets_requirements"] for r in r1)
    _emit("search_resumes",
          f"Round 1: scored {out['candidates_scored']} resumes -> top {len(r1)} "
          f"({qualified} meet every must-have + years)", lines)
    return {"round1": r1, "candidates_scored": out["candidates_scored"], "trace": lines}


def rank_candidates(state: AgentState) -> dict:
    """Round 2 (deep per-candidate analysis) + round 3 (hire / no-hire)."""
    lines: list[str] = []
    req, r1 = state["requirements"], state["round1"]
    _emit("rank_candidates", f"Round 2: deep analysis of {len(r1)} candidates", lines)
    analyses = T.analyze_candidates(r1, req)
    src = {a["source"] for a in analyses.values()}
    _emit("rank_candidates", f"Round 2 done ({'/'.join(sorted(src))})", lines)
    verdicts = T.final_verdicts(r1, analyses, req)
    counts = {d: sum(v["decision"] == d for v in verdicts.values()) for d in ("hire", "maybe", "no-hire")}
    _emit("rank_candidates", f"Round 3: verdicts {counts}", lines)

    shortlist = []
    for r in r1:
        cid = r["candidate_id"]
        a = analyses[cid]
        shortlist.append({**r, "round1_score": r["match_score"], "fit_score": a["fit_score"],
                          "final_score": T.final_score(r["match_score"], a["fit_score"]),
                          "analysis": a, "verdict": verdicts[cid]})
    # final ranking: qualified first, then the round-3 recommendation, then score
    order = {"hire": 2, "maybe": 1, "no-hire": 0}
    shortlist.sort(key=lambda c: (c["meets_requirements"], order[c["verdict"]["decision"]],
                                  c["final_score"]), reverse=True)
    for i, c in enumerate(shortlist, 1):
        c["rank"] = i

    changes = rank_diff(state.get("previous_shortlist") or [], shortlist,
                        _prev_requirements(state), req)
    if changes:
        moved = sum(1 for c in changes if c["change"] != "same")
        _emit("rank_candidates", f"Re-ranked: {moved} candidate(s) moved vs previous ranking", lines)
    return {"round2": analyses, "round3": verdicts, "shortlist": shortlist,
            "rank_changes": changes, "trace": lines}


def generate_report(state: AgentState) -> dict:
    lines: list[str] = []
    md = render_report(state)
    req = state["requirements"]
    slug = re.sub(r"[^a-z0-9]+", "_", req.get("role_title", "search").lower()).strip("_")[:40]
    rel = (f"reports/{slug}_v{req.get('version', 1)}_"
          f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    mcp_bridge.call("write_file", filepath=rel, content=md)
    _emit("generate_report", f"Match report written to {rel} (via MCP filesystem server)", lines)
    return {"report_md": md, "report_path": rel,
            "messages": [AIMessage(content=chat_summary(state, rel))], "trace": lines}


def human_feedback(state: AgentState) -> dict:
    """Pause for the user's next message, then classify what they want."""
    lines: list[str] = []
    user_msg = interrupt({"prompt": "Ask a question, refine the requirements, start a new search, or say 'done'.",
                          "report_path": state.get("report_path")})
    intent = classify_intent(str(user_msg), state)
    _emit("human_feedback", f"intent={intent}", lines)
    return {"messages": [HumanMessage(content=str(user_msg))], "query": str(user_msg),
            "intent": intent, "trace": lines}


class RefinedRequirements(BaseModel):
    role_title: str
    role_summary: str
    must_have: list[str]
    nice_to_have: list[str]
    min_years: int
    changes: list[str] = Field(description="Each change made, e.g. 'Kubernetes moved to must-have'")


def refine_requirements(state: AgentState) -> dict:
    """Apply a mid-conversation change to the criteria; remember the old
    ranking so the next report can explain movements."""
    lines: list[str] = []
    old = state["requirements"]
    try:
        out = llm.structured(RefinedRequirements, [
            SystemMessage(content=(
                "Update the hiring requirements according to the recruiter's instruction. "
                "Change only what they asked for and list each change. Keep skills atomic and "
                f"prefer these canonical names: {', '.join(T.SKILL_VOCAB)}.")),
            HumanMessage(content=f"CURRENT\n{json.dumps({k: old[k] for k in ('role_title', 'role_summary', 'must_have', 'nice_to_have', 'min_years')})}"
                                 f"\n\nINSTRUCTION\n{state['query']}"),
        ], effort="low")
        new = out.model_dump()
        changes = new.pop("changes")
        new["source"] = "llm"
    except Exception:
        new, changes = heuristic_refine(old, state["query"])
    new["must_have"] = T._canon_list(new["must_have"])
    new["nice_to_have"] = [s for s in T._canon_list(new["nice_to_have"])
                           if s.lower() not in {m.lower() for m in new["must_have"]}]
    new["version"] = old.get("version", 1) + 1
    changes = changes or describe_requirement_diff(old, new)
    _emit("refine_requirements", f"v{new['version']}: " + "; ".join(changes), lines)
    return {"requirements": new,
            "requirements_history": (state.get("requirements_history") or []) + [new],
            "requirement_changes": changes,
            "previous_shortlist": state.get("shortlist", []), "trace": lines}


CHAT_SYSTEM = """You are a recruiting assistant working over a resume database \
(100 resumes in ./resumes, job descriptions in ./jobs, reports in ./reports).

You already ran a 3-round screen for the current role. Use the tools to answer \
the recruiter: compare candidates, explain rankings with concrete score \
components, generate interview questions, read resumes, and save files when asked. \
Ground every claim in tool output; never invent resume facts. Refer to candidates \
by name and rank. Be concise; use markdown tables for comparisons.

File access (read_file, list_files, write_file, search_in_file, batch_process, \
watch_directory) is served by an MCP filesystem server, not read directly - call \
those tools exactly like any other. A second MCP server exposes the ground-truth \
candidate/job database (query_candidates, get_candidate_record, list_roles, \
get_job_posting) for questions the live index can't answer, e.g. "how many devops \
candidates are there" or "what's the ground-truth education for X".

Candidate tools accept a rank ("#2"), a full name, or just a first name, and \
search all 100 resumes, not only the shortlist. If a name is not in the shortlist \
(e.g. "Jane"), call the tool with it anyway: it picks the best-scoring match, \
scores that person on the fly and says so in its "notes". Report that note.

CURRENT REQUIREMENTS (v{version}): {requirements}

CURRENT SHORTLIST:
{shortlist}"""


def chat_agent(state: AgentState) -> dict:
    """Answer a question with a tool-calling loop over the agent's tools."""
    lines: list[str] = []
    ctx = {k: state.get(k) or ([] if k != "requirements" else {}) for k in
           ("requirements", "shortlist", "rank_changes", "requirement_changes", "requirements_history")}
    tools = T.make_tools(ctx)
    by_name = {t.name: t for t in tools}
    if llm.is_offline():
        answer = offline_answer(state["query"], ctx, lines, state["messages"])
        return {"messages": [AIMessage(content=answer)], "trace": lines}

    req = ctx["requirements"]
    system = CHAT_SYSTEM.format(
        version=req.get("version", 1),
        requirements=json.dumps({k: req.get(k) for k in ("role_title", "must_have", "nice_to_have", "min_years")}),
        shortlist="\n".join(f"#{c['rank']} {c['candidate_name']} ({c['candidate_id']}) final={c['final_score']} "
                            f"{c['verdict']['decision']}" for c in ctx["shortlist"]))
    history = [m for m in state["messages"] if isinstance(m, (HumanMessage, AIMessage))][-HISTORY_WINDOW:]
    msgs: list[dict] = [{"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
                        for m in history]
    while msgs and msgs[0]["role"] != "user":   # Messages API: first turn must be the user
        msgs.pop(0)
    answer = ""
    for _ in range(MAX_TOOL_STEPS):
        try:
            resp = llm.chat(system, msgs, T.tool_specs(tools))
        except Exception as e:   # API error: degrade to the offline path
            _emit("chat_agent", f"LLM error ({type(e).__name__}); using offline tools", lines)
            answer = offline_answer(state["query"], ctx, lines, state["messages"])
            break
        for b in resp.content:
            if b.type == "thinking" and b.thinking.strip():
                _emit("chat_agent", "thinking: " + " ".join(b.thinking.split())[:160], lines)
        if resp.stop_reason == "refusal":
            _emit("chat_agent", "model declined; using offline tools", lines)
            answer = offline_answer(state["query"], ctx, lines, state["messages"])
            break
        msgs.append({"role": "assistant", "content": resp.content})   # keep thinking + tool_use blocks
        calls = [b for b in resp.content if b.type == "tool_use"]
        if not calls:
            answer = "\n\n".join(b.text for b in resp.content if b.type == "text").strip()
            break
        results = []
        for call in calls:
            _emit("chat_agent", f"tool {call.name}({_short(call.input)})", lines)
            try:
                result, is_error = by_name[call.name].invoke(call.input), False
            except Exception as e:
                result, is_error = {"error": f"{type(e).__name__}: {e}"}, True
            results.append({"type": "tool_result", "tool_use_id": call.id, "is_error": is_error,
                            "content": json.dumps(result, default=str)[:12000]})
        msgs.append({"role": "user", "content": results})   # all results in one message
    else:
        answer = "I hit the tool-step limit before finishing; please narrow the question."
    return {"messages": [AIMessage(content=answer)], "trace": lines}


def _short(args: dict, limit: int = 90) -> str:
    s = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return s if len(s) <= limit else s[:limit] + "..."


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
_END_WORDS = {"done", "quit", "exit", "bye", "end", "stop", "q", "that's all", "thats all"}
_NEW_SEARCH = re.compile(r"^\s*(find|search|look for|show me candidates|new search|get me)\b|\bjd_[\w-]+|jobs/", re.I)
_REFINE = re.compile(
    r"\b(must[- ]?have|nice[- ]?to[- ]?have|drop|remove|relax|require[sd]?|requirement|"
    r"instead of|at least \d+|\d+\s*\+?\s*years|no longer|don'?t need|doesn'?t need|"
    r"make \w+(?: \w+)? (?:a )?(?:must|mandatory|optional|required)|add \w+ (?:as|to)|prioriti[sz]e|weight)\b", re.I)
_QUESTION = re.compile(r"\b(why|compare|explain|interview|question|tell me|who|what|which|how|show|"
                       r"save|write|read|list|summari[sz]e|details?)\b|\?", re.I)


class Intent(BaseModel):
    intent: str = Field(description="One of: refine, new_search, question, end")


def classify_intent(text: str, state: AgentState) -> str:
    """Rules for the clear cases; LLM for the ambiguous rest."""
    t = text.strip().lower().rstrip(".!")
    if t in _END_WORDS or t.startswith(("thanks, that's all", "no more", "we're done")):
        return "end"
    if _NEW_SEARCH.search(text):
        return "new_search"
    if _REFINE.search(text) and not re.match(r"^\s*(why|what|how|which|who)\b", text, re.I):
        return "refine"
    if _QUESTION.search(text):
        return "question"
    try:
        out = llm.structured(Intent, [
            SystemMessage(content=("Classify the recruiter's message. refine = change hiring criteria "
                                   "and re-rank; new_search = a different role/search; question = ask "
                                   "about candidates or files; end = finish the session.")),
            HumanMessage(content=text)], effort="low")
        if out.intent in {"refine", "new_search", "question", "end"}:
            return out.intent
    except Exception:
        pass
    return "question"


def route_feedback(state: AgentState) -> str:
    return state.get("intent", "question")


# --------------------------------------------------------------------------- #
# Explainability helpers
# --------------------------------------------------------------------------- #
def _prev_requirements(state: AgentState) -> dict:
    hist = state.get("requirements_history") or []
    return hist[-2] if len(hist) >= 2 else {}


def describe_requirement_diff(old: dict, new: dict) -> list[str]:
    out = []
    om, on = ({s.lower(): s for s in old.get(k, [])} for k in ("must_have", "nice_to_have"))
    nm, nn = ({s.lower(): s for s in new.get(k, [])} for k in ("must_have", "nice_to_have"))
    for s in nm:
        out.append(f"{nm[s]} moved nice-to-have → must-have" if s in on else
                   f"{nm[s]} added as must-have" if s not in om else None)
    for s in nn:
        out.append(f"{nn[s]} moved must-have → nice-to-have" if s in om else
                   f"{nn[s]} added as nice-to-have" if s not in on else None)
    out += [f"{om[s]} removed from must-have" for s in om if s not in nm and s not in nn]
    out += [f"{on[s]} removed from nice-to-have" for s in on if s not in nn and s not in nm]
    out = [o for o in out if o]
    if old.get("min_years") != new.get("min_years"):
        out.append(f"min years {old.get('min_years')} -> {new.get('min_years')}")
    return out or ["no effective change"]


def rank_diff(prev: list[dict], new: list[dict], old_req: dict, new_req: dict) -> list[dict]:
    """Per-candidate rank movement with a requirement-based reason."""
    if not prev:
        return []
    prev_rank = {c["candidate_id"]: c["rank"] for c in prev}
    prev_score = {c["candidate_id"]: c["final_score"] for c in prev}
    added_must = [s for s in new_req.get("must_have", []) if s.lower() not in {x.lower() for x in old_req.get("must_have", [])}]
    dropped = [s for s in old_req.get("must_have", []) + old_req.get("nice_to_have", [])
               if s.lower() not in {x.lower() for x in new_req.get("must_have", []) + new_req.get("nice_to_have", [])}]
    added_nice = [s for s in new_req.get("nice_to_have", []) if s.lower() not in {x.lower() for x in old_req.get("nice_to_have", []) + old_req.get("must_have", [])}]
    out = []
    for c in new:
        have = {s.lower() for s in c.get("skills", [])}
        reasons: list[tuple[str, int]] = []   # (text, +1 helps / -1 hurts)
        for s in added_must:
            reasons.append((f"has newly required {s}", 1) if s.lower() in have
                           else (f"lacks newly required {s}", -1))
        for s in added_nice:
            if s.lower() in have:
                reasons.append((f"has new nice-to-have {s}", 1))
        for s in dropped:
            reasons.append((f"loses credit for {s}", -1) if s.lower() in have
                           else (f"no longer penalised for missing {s}", 1))
        if old_req.get("min_years") != new_req.get("min_years"):
            ok = c["experience_years"] >= new_req.get("min_years", 0)
            reasons.append((f"{c['experience_years']} yrs {'meets' if ok else 'below'} new "
                            f"{new_req.get('min_years')}+ bar", 1 if ok else -1))
        old_rank = prev_rank.get(c["candidate_id"])
        if old_rank is None:
            change = "new"
        elif old_rank > c["rank"]:
            change = "up"
        elif old_rank < c["rank"]:
            change = "down"
        else:
            change = "same"
        # cite only the reasons that point the way the candidate actually moved
        want = {"up": 1, "new": 1, "down": -1}.get(change)
        cited = [t for t, sign in reasons if want is None or sign == want]
        if not cited:
            cited = (["candidates above were penalised more"] if change == "up" else
                     ["overtaken by candidates who gained more"] if change == "down" else
                     ["score shift from re-weighted retrieval"])
        out.append({"candidate_id": c["candidate_id"], "name": c["candidate_name"],
                    "old_rank": old_rank, "new_rank": c["rank"], "change": change,
                    "old_score": prev_score.get(c["candidate_id"]), "new_score": c["final_score"],
                    "reason": "; ".join(cited[:3])})
    new_ids = {c["candidate_id"] for c in new}
    for c in prev:
        if c["candidate_id"] not in new_ids:
            have = {s.lower() for s in c.get("skills", [])}
            why = [f"lacks newly required {s}" for s in added_must if s.lower() not in have]
            out.append({"candidate_id": c["candidate_id"], "name": c["candidate_name"],
                        "old_rank": c["rank"], "new_rank": None, "change": "dropped",
                        "old_score": c["final_score"], "new_score": None,
                        "reason": "; ".join(why) or "outscored after re-ranking"})
    return out


_ARROW = {"up": "↑", "down": "↓", "same": "=", "new": "★ new", "dropped": "✗ out"}


def _rank_change_label(rc: dict) -> str:
    if rc["change"] in ("up", "down"):
        return f"{_ARROW[rc['change']]}{abs(rc['old_rank'] - rc['new_rank'])}"
    return _ARROW[rc["change"]]


def render_report(state: AgentState) -> str:
    req, sl = state["requirements"], state["shortlist"]
    counts = {d: sum(c["verdict"]["decision"] == d for c in sl) for d in ("hire", "maybe", "no-hire")}
    changes = {rc["candidate_id"]: rc for rc in state.get("rank_changes") or []}
    out = [f"# Match report: {req.get('role_title', 'Candidate search')}",
           f"_Generated {datetime.now():%Y-%m-%d %H:%M} · requirements v{req.get('version', 1)} · "
           f"JD source: {state.get('jd_source', 'chat')}_", "",
           "## Requirements",
           f"- **Must have:** {', '.join(req.get('must_have', [])) or '—'}",
           f"- **Nice to have:** {', '.join(req.get('nice_to_have', [])) or '—'}",
           f"- **Minimum experience:** {req.get('min_years', 0)} years", ""]
    if state.get("requirement_changes") and req.get("version", 1) > 1:
        out += ["### Changes in this version", *[f"- {c}" for c in state["requirement_changes"]], ""]
    out += ["## Screening funnel",
            f"1. **Round 1 — initial screen:** {state.get('candidates_scored', '?')} resumes scored "
            f"(hybrid semantic + keyword), top {len(sl)} kept",
            f"2. **Round 2 — deep analysis:** strengths, gaps and evidence for each of the {len(sl)}",
            f"3. **Round 3 — final call:** {counts['hire']} hire · {counts['maybe']} maybe · "
            f"{counts['no-hire']} no-hire", "",
            "## Ranking",
            "| # | Candidate | Final | Retrieval | Fit | Years | Missing must-have | Verdict |" +
            (" Δ |" if changes else ""),
            "|---|---|---|---|---|---|---|---|" + ("---|" if changes else "")]
    for c in sl:
        row = (f"| {c['rank']} | {c['candidate_name']} | **{c['final_score']}** | {c['round1_score']} | "
               f"{c['fit_score']} | {c['experience_years']} | {', '.join(c['missing_must_have']) or '—'} | "
               f"{c['verdict']['decision']} |")
        if changes:
            row += f" {_rank_change_label(changes[c['candidate_id']])} |" if c["candidate_id"] in changes else " |"
        out.append(row)
    out.append("")
    if changes:
        out += ["## Ranking changes"]
        for rc in sorted(changes.values(), key=lambda r: (r["new_rank"] is None, r["new_rank"] or 0)):
            if rc["change"] == "same":
                continue
            where = (f"#{rc['old_rank']} → #{rc['new_rank']}" if rc["old_rank"] and rc["new_rank"]
                     else f"new at #{rc['new_rank']}" if rc["new_rank"] else f"#{rc['old_rank']} → out of top {len(sl)}")
            out.append(f"- **{rc['name']}** {_rank_change_label(rc)} ({where}): {rc['reason']}")
        out.append("")
    out += ["## Candidate details"]
    for c in sl:
        a, v = c["analysis"], c["verdict"]
        out += [f"### {c['rank']}. {c['candidate_name']} — {v['decision'].upper()} "
                f"({v['confidence']} confidence)",
                f"`{c['resume_path']}` · {c['experience_years']} yrs · final {c['final_score']} "
                f"(retrieval {c['round1_score']} = semantic {c['semantic_score']} / keyword {c['keyword_score']}; "
                f"fit {c['fit_score']})", "",
                f"**Verdict:** {v['rationale']}", "",
                f"**Summary:** {a['summary']}", "",
                "**Strengths**", *[f"- {s}" for s in a["strengths"] or ["—"]], "",
                "**Gaps**", *[f"- {g}" for g in a["gaps"] or ["None against the stated requirements"]], ""]
        if a.get("evidence"):
            out += ["**Evidence**", *[f"> {e}" for e in a["evidence"][:3]], ""]
        if a.get("improvement_suggestions") and (T.BORDERLINE[0] <= c["fit_score"] <= T.BORDERLINE[1]
                                                  or v["decision"] == "maybe"):
            out += ["**Improvement suggestions (borderline)**",
                    *[f"- {s}" for s in a["improvement_suggestions"]], ""]
    return "\n".join(out)


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0].rstrip(",;") + "…"


def chat_summary(state: AgentState, report_path: str) -> str:
    req, sl = state["requirements"], state["shortlist"]
    counts = {d: sum(c["verdict"]["decision"] == d for c in sl) for d in ("hire", "maybe", "no-hire")}
    out = []
    if req.get("version", 1) > 1:
        out += [f"**Requirements updated (v{req['version']}):** " + "; ".join(state.get("requirement_changes") or []), ""]
    out += [f"**{req.get('role_title', 'Search')}** — must: {', '.join(req.get('must_have', [])) or '—'}; "
            f"nice: {', '.join(req.get('nice_to_have', [])) or '—'}; {req.get('min_years', 0)}+ yrs", "",
            f"Screened {state.get('candidates_scored', '?')} resumes → top {len(sl)} → "
            f"{counts['hire']} hire / {counts['maybe']} maybe / {counts['no-hire']} no-hire", "",
            "| # | Candidate | Final | Verdict | Key reason |", "|---|---|---|---|---|"]
    for c in sl[:5]:
        out.append(f"| {c['rank']} | {c['candidate_name']} | {c['final_score']} | {c['verdict']['decision']} | "
                   f"{_clip(c['verdict']['rationale'], 110)} |")
    moves = [rc for rc in state.get("rank_changes") or [] if rc["change"] not in ("same",)]
    if moves:
        out += ["", "**Ranking changes:**"]
        for rc in moves[:6]:
            out.append(f"- {rc['name']} {_rank_change_label(rc)}: {rc['reason']}")
    out += ["", f"Full report: `{report_path}`"]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Offline fallbacks (no LLM): rule-based tool dispatch
# --------------------------------------------------------------------------- #
def heuristic_refine(old: dict, text: str) -> tuple[dict, list[str]]:
    new = {**old, "must_have": list(old["must_have"]), "nice_to_have": list(old["nice_to_have"]),
           "source": "heuristic"}
    low = text.lower()
    clauses = re.split(r"[,;]|\band\b|\bbut\b", low)
    for cl in clauses:
        skills = T.extract_jd_skills(cl)
        if not skills:
            continue
        if re.search(r"\b(drop|remove|don'?t need|no longer|without|ignore)\b", cl):
            new["must_have"] = [s for s in new["must_have"] if s not in skills]
            new["nice_to_have"] = [s for s in new["nice_to_have"] if s not in skills]
        elif re.search(r"nice[- ]?to[- ]?have|optional|bonus|plus", cl):
            new["must_have"] = [s for s in new["must_have"] if s not in skills]
            new["nice_to_have"] += [s for s in skills if s not in new["nice_to_have"]]
        else:
            new["nice_to_have"] = [s for s in new["nice_to_have"] if s not in skills]
            new["must_have"] += [s for s in skills if s not in new["must_have"]]
    m = re.search(r"(\d+)\s*\+?\s*years", low)
    if m:
        new["min_years"] = int(m.group(1))
    return new, describe_requirement_diff(old, new)


def _mentioned_candidates(text: str, ranked: list[str]) -> list[str]:
    reg = T.registry()
    low = text.lower()
    found: list[tuple[int, str]] = []
    for cid, c in reg.items():
        i = low.find(c["name"].lower())
        if i >= 0:
            found.append((i, cid))
    if not found:   # bare first names: prefer the shortlist, else the whole pool
        firsts = {c["name"].split()[0].lower() for c in reg.values()}
        for m in re.finditer(r"\b[a-z]+\b", low):
            if m.group(0) in firsts:
                try:
                    found.append((m.start(), T.resolve_candidate(m.group(0), ranked)))
                except ValueError:   # ambiguous: tools pick the best-scoring match
                    found.append((m.start(), m.group(0)))
    for m in re.finditer(r"#(\d+)", text):
        i = int(m.group(1)) - 1
        if 0 <= i < len(ranked):
            found.append((m.start(), ranked[i]))
    seen, out = set(), []
    for _, cid in sorted(found):
        if cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def offline_answer(query: str, ctx: dict, lines: list[str], history: list | None = None) -> str:
    ranked = [c["candidate_id"] for c in ctx["shortlist"]]
    low = query.lower()
    mentioned = _mentioned_candidates(query, ranked)
    if re.search(r"\b(save|write)\b", low):
        m = re.search(r"([\w./-]+\.(?:md|txt))", query)
        path = m.group(1) if m else "reports/notes.md"
        prev = [msg for msg in (history or []) if isinstance(msg, AIMessage)]
        content = prev[-1].content if prev else ""
        _emit("chat_agent", f"tool write_file(filepath={path!r}) via MCP filesystem server", lines)
        res = mcp_bridge.call("write_file", filepath=path, content=content)
        return (f"Saved {res['bytes_written']} bytes to `{path}`." if res.get("success")
                else f"Could not save: {res.get('error')}")
    if "why" in low and ("change" in low or "moved" in low) and ctx.get("rank_changes"):
        _emit("chat_agent", "tool get_rank_changes()", lines)
        moves = [rc for rc in ctx["rank_changes"] if rc["change"] != "same"]
        return ("**Requirement changes:** " + "; ".join(ctx.get("requirement_changes", [])) + "\n\n" +
                "\n".join(f"- {rc['name']} {_rank_change_label(rc)}: {rc['reason']}" for rc in moves))
    if "compare" in low or "side by side" in low:
        m = re.search(r"top\s*(\d+)", low)
        ids = mentioned if len(mentioned) >= 2 else ranked[: int(m.group(1)) if m else 3]
        _emit("chat_agent", f"tool compare_candidates({ids})", lines)
        res = T.compare_candidates(ids, ctx)
        return f"**Head-to-head** (leader: {res['leader']})\n\n{res['table_md']}"
    if "why" in low and len(mentioned) >= 2:
        _emit("chat_agent", f"tool explain_ranking({mentioned[0]}, {mentioned[1]})", lines)
        res = T.explain_ranking(mentioned[0], mentioned[1], ctx)
        if "error" in res:
            return res["error"]
        notes = "".join(f"\n> {n}" for n in res.get("notes", []))
        hi, lo, comp = res["higher_ranked"], res["lower_ranked"], res["score_components"]
        rows = "\n".join(f"| {k} | {v['higher']} | {v['lower']} | {v['delta']:+} |" for k, v in comp.items())
        diffs = "\n".join(f"- {d}" for d in res["key_differences"]) or "- (no requirement-level differences)"
        rank = lambda r: f"#{r}" if r else "not shortlisted"
        return (f"**{hi['name']}** ({rank(hi['rank'])}) ranks above **{lo['name']}** ({rank(lo['rank'])}).\n\n"
                f"| component | {hi['name']} | {lo['name']} | Δ |\n|---|---|---|---|\n{rows}\n\n"
                f"Key differences:\n{diffs}\n\n_{res['scoring_formula']}_{notes}")
    if "interview" in low or "question" in low:
        cid = mentioned[0] if mentioned else ranked[0]
        _emit("chat_agent", f"tool generate_interview_questions({cid})", lines)
        res = T.generate_interview_questions(cid, ctx)
        qs = "\n".join(f"{i}. {q['question']}  _({q['purpose']})_" for i, q in enumerate(res["questions"], 1))
        return f"**Screening questions for {res['name']}**\n\n{qs}"
    if any(w in low for w in ("strength", "gap", "detail", "tell me about", "improve", "borderline")):
        targets = mentioned or [c["candidate_id"] for c in ctx["shortlist"]
                                if c["verdict"]["decision"] == "maybe"][:3] or ranked[:1]
        parts = []
        for cid in targets:
            _emit("chat_agent", f"tool get_candidate_analysis({cid})", lines)
            p = T.profile(T.resolve_for(cid, ctx)[0], ctx)
            a = p["analysis"]
            parts.append(f"**{p['candidate_name']}** (#{p.get('rank') or '-'}, fit {a['fit_score']})\n"
                         f"- Strengths: {'; '.join(a['strengths']) or '—'}\n- Gaps: {'; '.join(a['gaps']) or '—'}\n"
                         f"- Suggestions: {'; '.join(a['improvement_suggestions']) or '—'}")
        return "\n\n".join(parts)
    _emit("chat_agent", "tool get_shortlist()", lines)
    view = T.shortlist_view(ctx["shortlist"])
    return "Current shortlist:\n" + "\n".join(
        f"{c['rank']}. {c['name']} — {c['final_score']} ({c['decision']})" for c in view)


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("parse_jd", parse_jd)
    g.add_node("extract_requirements", extract_requirements_node)
    g.add_node("search_resumes", search_resumes)
    g.add_node("rank_candidates", rank_candidates)
    g.add_node("generate_report", generate_report)
    g.add_node("human_feedback", human_feedback)
    g.add_node("refine_requirements", refine_requirements)
    g.add_node("chat_agent", chat_agent)

    g.add_edge(START, "parse_jd")
    g.add_edge("parse_jd", "extract_requirements")
    g.add_edge("extract_requirements", "search_resumes")
    g.add_edge("search_resumes", "rank_candidates")
    g.add_edge("rank_candidates", "generate_report")
    g.add_edge("generate_report", "human_feedback")
    g.add_conditional_edges("human_feedback", route_feedback, {
        "refine": "refine_requirements", "new_search": "parse_jd",
        "question": "chat_agent", "end": END})
    g.add_edge("refine_requirements", "search_resumes")
    g.add_edge("chat_agent", "human_feedback")
    return g.compile(checkpointer=checkpointer or MemorySaver())


class MatchingSession:
    """One conversation thread. `send()` yields UI events as the graph runs:
       {"type": "trace", "node", "text"}   live reasoning / tool log
       {"type": "node", "node"}            a node finished
       {"type": "message", "content"}      assistant reply
       {"type": "waiting"} / {"type": "end"}"""

    def __init__(self, thread_id: str | None = None):
        self.graph = build_graph()
        self.config = {"configurable": {"thread_id": thread_id or uuid.uuid4().hex},
                       "recursion_limit": 50}
        self.started = False
        self.finished = False

    def state(self) -> dict:
        return self.graph.get_state(self.config).values

    def send(self, text: str) -> Iterator[dict]:
        if self.finished:
            yield {"type": "end"}
            return
        payload = Command(resume=text) if self.started else {
            "query": text, "messages": [HumanMessage(content=text)]}
        self.started = True
        for mode, chunk in self.graph.stream(payload, self.config, stream_mode=["updates", "custom"]):
            if mode == "custom":
                yield {"type": "trace", **chunk}
                continue
            for node, update in chunk.items():
                if node == "__interrupt__":
                    yield {"type": "waiting"}
                    continue
                yield {"type": "node", "node": node}
                for m in (update or {}).get("messages", []):
                    if isinstance(m, AIMessage):
                        yield {"type": "message", "content": m.content}
        if not self.graph.get_state(self.config).next:
            self.finished = True
            yield {"type": "end"}


if __name__ == "__main__":
    import sys

    s = MatchingSession()
    for ev in s.send(" ".join(sys.argv[1:]) or "jobs/jd_ml_engineer.txt"):
        if ev["type"] == "trace":
            print(f"  · [{ev['node']}] {ev['text']}")
        elif ev["type"] == "message":
            print(ev["content"])
