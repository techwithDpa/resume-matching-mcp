"""Graph-level tests: routing, multi-round screening, refinement, explanations.
Runs the real LangGraph with offline heuristics (no LLM)."""

import os

import pytest

import agent_tools
import matching_agent as A
from matching_agent import MatchingSession, classify_intent

FLOWS = os.path.join(agent_tools.BASE_DIR, "flows")


def drive(session, text):
    return list(session.send(text))


@pytest.mark.parametrize("text,intent", [
    ("done", "end"),
    ("Find me candidates with React and 3+ years experience", "new_search"),
    ("jobs/jd_ml_engineer.txt", "new_search"),
    ("Drop Go and make Terraform a must-have", "refine"),
    ("Also require at least 8 years of experience", "refine"),
    ("Compare the top 3 matches side by side", "question"),
    ("Why did John rank higher than Jane?", "question"),
    ("Generate interview questions for the #1 candidate", "question"),
])
def test_intent_routing(text, intent):
    assert classify_intent(text, {}) == intent


def test_full_screen_produces_three_rounds_and_report():
    s = MatchingSession()
    events = drive(s, "jobs/jd_ml_engineer.txt")
    nodes = [e["node"] for e in events if e["type"] == "node"]
    assert nodes == ["parse_jd", "extract_requirements", "search_resumes",
                     "rank_candidates", "generate_report"]
    assert events[-1]["type"] == "waiting"          # paused at human_feedback
    st = s.state()
    assert st["candidates_scored"] == 100
    assert len(st["round1"]) == len(st["round2"]) == len(st["round3"]) == 10
    assert [c["rank"] for c in st["shortlist"]] == list(range(1, 11))
    assert all(v["decision"] in {"hire", "maybe", "no-hire"} for v in st["round3"].values())
    assert os.path.isfile(os.path.join(agent_tools.BASE_DIR, st["report_path"]))
    md = st["report_md"]
    for section in ("## Requirements", "## Screening funnel", "## Ranking", "**Strengths**", "**Gaps**"):
        assert section in md


def test_refinement_reranks_and_explains_changes():
    s = MatchingSession()
    drive(s, "jobs/jd_platform_engineer.txt")
    before = [c["candidate_id"] for c in s.state()["shortlist"]]
    events = drive(s, "Drop Go and make Terraform a must-have")
    st = s.state()
    assert st["requirements"]["version"] == 2
    assert "Terraform" in st["requirements"]["must_have"] and "Go" not in st["requirements"]["must_have"]
    assert [c["candidate_id"] for c in st["shortlist"]] != before
    moved = [rc for rc in st["rank_changes"] if rc["change"] != "same"]
    assert moved and all(rc["reason"] for rc in moved)
    reply = next(e["content"] for e in events if e["type"] == "message")
    assert "Ranking changes" in reply and "## Ranking changes" in st["report_md"]


def test_compare_why_questions_and_save(tmp_path):
    s = MatchingSession()
    drive(s, "jobs/jd_backend_python.txt")
    msg = lambda evs: next(e["content"] for e in evs if e["type"] == "message")
    assert "| Final score |" in msg(drive(s, "Compare the top 3 matches side by side"))
    why = msg(drive(s, "Why did #1 rank higher than #3?"))
    assert "ranks above" in why and "final_score" in why
    why_names = msg(drive(s, "Why did John rank higher than Jane?"))
    assert "Jane" in why_names
    qs = msg(drive(s, "Generate interview questions for the #1 candidate"))
    assert "Screening questions" in qs
    out = "reports/test_questions.md"
    saved = msg(drive(s, f"Save those questions to {out}"))
    assert "Saved" in saved
    path = os.path.join(agent_tools.BASE_DIR, out)
    assert "Screening questions" in open(path).read()
    os.remove(path)


@pytest.mark.parametrize("flow", sorted(os.listdir(FLOWS)))
def test_scripted_flows_run_to_end(flow):
    s = MatchingSession()
    lines = [l.strip() for l in open(os.path.join(FLOWS, flow)) if l.strip() and not l.startswith("#")]
    for line in lines:
        if line.startswith("!"):   # a shell directive (see chat_cli.py) - not a chat turn
            continue
        events = drive(s, line)
        assert any(e["type"] in ("message", "end") for e in events), line
    assert s.finished


@pytest.mark.live
def test_live_llm_requirements_and_chat():
    s = MatchingSession()
    drive(s, "Find me candidates with React and 3+ years experience")
    st = s.state()
    assert st["requirements"]["source"] == "llm"
    assert "React" in st["requirements"]["must_have"]
    reply = next(e["content"] for e in drive(s, "Compare the top 3 matches side by side")
                 if e["type"] == "message")
    assert "|" in reply
