"""Tools for the resume matching agent.

Two layers:
  1. Plain-Python functions (requirement extraction, candidate analysis,
     comparison, interview questions, ranking explanations). The LangGraph
     nodes in matching_agent.py call these directly.
  2. `make_tools(ctx)` wraps them — plus every tool from the two MCP servers
     (filesystem_mcp_server.py, resume_db_mcp_server.py) and the Milestone 2
     RAG search — as tools for the conversational (tool-calling) node. `ctx`
     is a snapshot of the agent state, so tools can see the current
     requirements and shortlist without mutating them.

All file I/O (reading resumes, writing reports) goes through the filesystem
MCP server via mcp_bridge.call(), not direct fs_tools calls — Milestone 4
replaces the Milestone 1-3 direct file tools with MCP as the sole file-access
path for the agent. fs_tools.py itself now lives inside the MCP servers only.

Every LLM-backed function has a deterministic heuristic fallback, used when the
LLM is unavailable (offline mode / quota) or returns something unparsable.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

import llm
import mcp_bridge
from job_matcher import JobMatcher, extract_jd_skills
from resume_rag import KNOWN_SKILLS, _display_skill

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(BASE_DIR, "reports")
BORDERLINE = (50, 70)   # fit scores in this band get improvement suggestions

SKILL_VOCAB = sorted({_display_skill(s) for s in KNOWN_SKILLS})


# --------------------------------------------------------------------------- #
# Structured-output schemas
# --------------------------------------------------------------------------- #
class Requirements(BaseModel):
    role_title: str = Field(description="Short job title, e.g. 'Senior Backend Engineer'")
    role_summary: str = Field(description="One-sentence summary of the role")
    must_have: list[str] = Field(description="Non-negotiable skills")
    nice_to_have: list[str] = Field(description="Preferred / bonus skills")
    min_years: int = Field(description="Minimum years of experience, 0 if unstated")


class CandidateAnalysis(BaseModel):
    fit_score: int = Field(description="0-100 overall fit for the role")
    summary: str = Field(description="One or two sentences on overall fit")
    strengths: list[str]
    gaps: list[str]
    evidence: list[str] = Field(description="Short quotes/facts from the resume backing the score")
    improvement_suggestions: list[str] = Field(
        description="What would make this candidate a stronger fit; empty if clearly strong or clearly unfit")


class Verdict(BaseModel):
    candidate_id: str
    decision: Literal["hire", "maybe", "no-hire"]
    confidence: Literal["high", "medium", "low"]
    rationale: str


class Verdicts(BaseModel):
    verdicts: list[Verdict]


class InterviewQuestion(BaseModel):
    question: str
    purpose: str = Field(description="What this question probes (a strength to verify or a gap to test)")


class InterviewQuestions(BaseModel):
    questions: list[InterviewQuestion]


# --------------------------------------------------------------------------- #
# Candidate registry + lookup
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def get_matcher() -> JobMatcher:
    return JobMatcher()


@lru_cache(maxsize=1)
def registry() -> dict[str, dict]:
    """candidate_id (resume filename stem) -> {candidate_id, name, resume_path, skills, experience_years}."""
    got = get_matcher().col.get(include=["metadatas"])
    reg: dict[str, dict] = {}
    for m in got["metadatas"]:
        cid = candidate_id_for(m["resume_path"])
        reg.setdefault(cid, {
            "candidate_id": cid,
            "name": m["candidate_name"],
            "resume_path": m["resume_path"],
            "skills": [s.strip() for s in m.get("skills_joined", "").split(",") if s.strip()],
            "experience_years": m.get("experience_years", 0),
        })
    return reg


def candidate_id_for(resume_path: str) -> str:
    return os.path.splitext(os.path.basename(resume_path))[0]


def _norm(s: str) -> str:
    return re.sub(r"[\s_]+", " ", s.strip().lower())


def resolve_candidate(ref: str, ranked_ids: list[str] | None = None) -> str:
    """Resolve a candidate reference to a candidate_id.

    Accepts an id ("John_Doe"), a full or partial name ("john doe", "John"),
    or a rank in the current shortlist ("#1", "2"). Ambiguous first names are
    resolved in favour of the best-ranked shortlisted candidate.
    """
    ranked_ids = ranked_ids or []
    reg = registry()
    r = ref.strip().lstrip("#")
    if r.isdigit() and ranked_ids:
        i = int(r) - 1
        if 0 <= i < len(ranked_ids):
            return ranked_ids[i]
        raise ValueError(f"rank {r} is outside the shortlist (1-{len(ranked_ids)})")
    key = _norm(r)
    for cid, c in reg.items():
        if _norm(cid) == key or _norm(c["name"]) == key:
            return cid
    tokens = key.split()
    partial = [cid for cid, c in reg.items()
               if all(t in _norm(c["name"]).split() for t in tokens)]
    if not partial:
        raise ValueError(f"no candidate matches '{ref}'")
    if len(partial) == 1:
        return partial[0]
    in_shortlist = [cid for cid in ranked_ids if cid in partial]
    if in_shortlist:
        return in_shortlist[0]
    names = ", ".join(reg[c]["name"] for c in sorted(partial))
    raise ValueError(f"'{ref}' is ambiguous: {names}")


def resolve_for(ref: str, ctx: dict) -> tuple[str, str | None]:
    """resolve_candidate against the current shortlist; when a name is
    ambiguous outside it (e.g. 'Jane' with three Janes in the pool), pick the
    best-scoring match for the current requirements and say so."""
    ranked = [c["candidate_id"] for c in ctx.get("shortlist", [])]
    try:
        return resolve_candidate(ref, ranked), None
    except ValueError as e:
        if "ambiguous" not in str(e):
            raise
    key = _norm(ref.strip().lstrip("#")).split()
    options = [cid for cid, c in registry().items() if all(t in _norm(c["name"]).split() for t in key)]
    req = ctx.get("requirements", {})
    best = max(options, key=lambda cid: score_candidate(cid, req)["match_score"])
    names = ", ".join(registry()[c]["name"] for c in sorted(options))
    return best, (f"'{ref}' matches {len(options)} candidates ({names}); using "
                  f"{registry()[best]['name']}, the best-scoring for this role")


def resume_text(candidate_id: str) -> str:
    """One resume's text, via the filesystem MCP server's read_file tool."""
    c = registry()[candidate_id]
    result = mcp_bridge.call("read_file", filepath=c["resume_path"])
    return result.get("content") or "" if isinstance(result, dict) else ""


def resume_texts_batch(candidate_ids: list[str]) -> dict[str, str]:
    """Every listed candidate's resume text, in ONE MCP batch_process call
    instead of one read_file round trip per candidate (used by Round 2, which
    needs the full text of every shortlisted candidate at once)."""
    reg = registry()
    paths = [reg[cid]["resume_path"] for cid in candidate_ids]
    out = mcp_bridge.call("batch_process", filepaths=paths, operation="read")
    by_path = {r["filepath"]: r for r in out.get("results", [])} if isinstance(out, dict) else {}
    texts = {}
    for cid in candidate_ids:
        r = by_path.get(reg[cid]["resume_path"], {})
        texts[cid] = r.get("content") or "" if r.get("success") else ""
    return texts


def canonical_skill(skill: str) -> str:
    """Map a skill name onto the indexed vocabulary when possible (so keyword
    matching against resume metadata works), else keep it as given."""
    low = skill.strip().lower()
    aliases = {"k8s": "kubernetes", "golang": "go", "js": "javascript", "ts": "typescript",
               "node": "node.js", "nodejs": "node.js", "reactjs": "react", "react.js": "react",
               "ml": "machine learning", "rest": "rest apis", "rest api": "rest apis",
               "postgres": "postgresql", "sklearn": "scikit-learn", "ci-cd": "ci/cd"}
    low = aliases.get(low, low)
    return _display_skill(low) if low in KNOWN_SKILLS else skill.strip()


def _canon_list(skills: list[str]) -> list[str]:
    out: list[str] = []
    for s in skills:
        c = canonical_skill(s)
        if c and c.lower() not in {o.lower() for o in out}:
            out.append(c)
    return out


# --------------------------------------------------------------------------- #
# extract_requirements
# --------------------------------------------------------------------------- #
_MUST_CUES = ("must", "required", "require", "requires", "need", "strong", "with")


def heuristic_requirements(jd: str) -> dict:
    """Rule-based fallback: skills in 'must/required' sentences are must-have,
    the rest nice-to-have; min years from an 'N+ years' phrase."""
    lines = [l for l in jd.strip().splitlines() if l.strip()]
    title = lines[0].strip() if len(lines) > 1 else "Ad-hoc candidate search"
    body = "\n".join(lines[1:]) if len(lines) > 1 else jd
    sentences = re.split(r"(?<=[.!?;])\s+|\n", body)
    must, nice = [], []
    for sent in sentences:
        skills = extract_jd_skills(sent)
        low = sent.lower()
        is_nice = any(k in low for k in ("plus", "preferred", "nice", "bonus", "valued"))
        if not is_nice and any(k in low for k in _MUST_CUES):
            must.extend(s for s in skills if s not in must)   # must cues win over earlier mentions
        else:
            nice.extend(s for s in skills if s not in nice)
    nice = [s for s in nice if s not in must]
    if not must:
        must, nice = nice, []
    m = re.search(r"(\d+)\s*\+?\s*(?:years|yrs)", jd.lower())
    return {
        "role_title": title,
        "role_summary": " ".join(jd.split())[:160],
        "must_have": must,
        "nice_to_have": nice,
        "min_years": int(m.group(1)) if m else 0,
        "source": "heuristic",
    }


def extract_requirements(jd: str) -> dict:
    """Parse a JD (or a natural-language search request) into must-have vs
    nice-to-have skills and minimum years."""
    try:
        req = llm.structured(Requirements, [
            SystemMessage(content=(
                "You extract hiring requirements from a job description or a recruiter's "
                "search request. Separate non-negotiable must-have skills from nice-to-have "
                "ones. Keep skills atomic (one technology each). Only skills the text states "
                "as required are must-have. When the text offers alternatives ('X or Y'), "
                "neither is must-have: list them as nice-to-have. If the request is terse (e.g. 'React devs with 3+ "
                "years'), add 2-4 skills that commonly accompany the role as nice-to-have so "
                "candidates can be differentiated. Prefer these canonical skill names when "
                f"they apply: {', '.join(SKILL_VOCAB)}.")),
            HumanMessage(content=jd),
        ], effort="low")
        out = req.model_dump()
        out["source"] = "llm"
    except Exception:
        out = heuristic_requirements(jd)
    out["must_have"] = _canon_list(out["must_have"])
    out["nice_to_have"] = [s for s in _canon_list(out["nice_to_have"])
                           if s.lower() not in {m.lower() for m in out["must_have"]}]
    out["min_years"] = max(0, int(out.get("min_years") or 0))
    return out


def requirements_query(req: dict, jd_text: str = "") -> str:
    """Text used for semantic retrieval: the role plus the current skill lists.
    Built from the requirements (not the raw JD) so refinements move results."""
    parts = [req.get("role_title", ""), req.get("role_summary", "")]
    if req.get("must_have"):
        parts.append("Must have: " + ", ".join(req["must_have"]) + ".")
    if req.get("nice_to_have"):
        parts.append("Nice to have: " + ", ".join(req["nice_to_have"]) + ".")
    if req.get("min_years"):
        parts.append(f"{req['min_years']}+ years of experience.")
    return " ".join(p for p in parts if p) or jd_text


# --------------------------------------------------------------------------- #
# Round 1: retrieval (RAG search from Milestone 2)
# --------------------------------------------------------------------------- #
def round1_search(req: dict, jd_text: str = "", top_k: int = 10) -> dict:
    """Score every indexed resume with the hybrid semantic+keyword matcher and
    keep the top_k (qualified candidates first)."""
    m = get_matcher()
    out = m.match(requirements_query(req, jd_text), top_k=top_k,
                  must_have=req.get("must_have", []), min_years=req.get("min_years", 0),
                  skills=req.get("must_have", []) + req.get("nice_to_have", []),
                  pool=m.col.count())
    for r in out["top_matches"]:
        r["candidate_id"] = candidate_id_for(r["resume_path"])
    return out


def score_candidate(candidate_id: str, req: dict) -> dict:
    """Round-1-style score for one specific candidate (e.g. someone the user
    asks about who is not in the shortlist)."""
    m = get_matcher()
    c = registry()[candidate_id]
    q = m.index.embed([requirements_query(req)])[0]
    got = m.col.get(where={"resume_path": c["resume_path"]}, include=["embeddings"])
    sim = max((sum(a * b for a, b in zip(q, e)) for e in got["embeddings"]), default=0.0)
    wanted = [s.lower() for s in req.get("must_have", []) + req.get("nice_to_have", [])]
    have = {s.lower() for s in c["skills"]}
    matched = [s for s in wanted if s in have]
    kw = len(matched) / len(wanted) if wanted else 0.0
    missing = [s for s in req.get("must_have", []) if s.lower() not in have]
    return {
        "candidate_id": candidate_id,
        "candidate_name": c["name"],
        "resume_path": c["resume_path"],
        "match_score": round(100 * (0.6 * max(0.0, sim) + 0.4 * kw)),
        "semantic_score": round(100 * max(0.0, sim)),
        "keyword_score": round(100 * kw),
        "experience_years": c["experience_years"],
        "skills": c["skills"],
        "matched_skills": [_display_skill(s) if s in KNOWN_SKILLS else s for s in matched],
        "missing_must_have": missing,
        "meets_requirements": not missing and c["experience_years"] >= req.get("min_years", 0),
    }


# --------------------------------------------------------------------------- #
# Round 2: deep analysis
# --------------------------------------------------------------------------- #
def heuristic_analysis(r1: dict, req: dict) -> dict:
    have = {s.lower() for s in r1.get("skills", [])}
    must = req.get("must_have", [])
    nice = req.get("nice_to_have", [])
    years, min_years = r1.get("experience_years", 0), req.get("min_years", 0)
    must_hit = [s for s in must if s.lower() in have]
    nice_hit = [s for s in nice if s.lower() in have]
    strengths = ([f"Has must-have skills: {', '.join(must_hit)}"] if must_hit else []) + \
                ([f"Bonus skills: {', '.join(nice_hit)}"] if nice_hit else [])
    if years >= min_years and years:
        strengths.append(f"{years} years of experience (needs {min_years}+)")
    gaps = [f"Missing must-have: {s}" for s in must if s.lower() not in have]
    gaps += [f"No evidence of {s}" for s in nice if s.lower() not in have]
    if years < min_years:
        gaps.append(f"Only {years} years of experience (needs {min_years}+)")
    must_cov = len(must_hit) / len(must) if must else 1.0
    nice_cov = len(nice_hit) / len(nice) if nice else 1.0
    years_fit = min(1.0, years / min_years) if min_years else 1.0
    fit = round(100 * (0.55 * must_cov + 0.15 * nice_cov + 0.15 * years_fit
                       + 0.15 * r1.get("semantic_score", 0) / 100))
    suggestions = []
    if BORDERLINE[0] <= fit <= BORDERLINE[1] or (fit > BORDERLINE[1] and gaps):
        suggestions = [f"Build demonstrable experience with {g.split(': ')[-1].replace('No evidence of ', '')}"
                       for g in gaps[:3]]
    return {
        "fit_score": fit,
        "summary": f"Covers {len(must_hit)}/{len(must)} must-have and {len(nice_hit)}/{len(nice)} nice-to-have skills.",
        "strengths": strengths,
        "gaps": gaps,
        "evidence": r1.get("relevant_excerpts", [])[:2],
        "improvement_suggestions": suggestions,
        "source": "heuristic",
    }


def _analysis_messages(candidate_id: str, r1: dict, req: dict, text: str) -> list:
    return [
        SystemMessage(content=(
            "You are a senior technical recruiter doing a second-round deep screen. "
            "Judge the resume strictly against the requirements. Cite concrete resume "
            "facts as evidence. Do not invent experience the resume does not state. "
            "Calibrate fit_score so candidates are differentiated: weigh must-have coverage "
            "most, then depth (years and recency of hands-on use, seniority, measurable "
            "impact), then nice-to-haves. Reserve 90+ for clearly exceptional fits. "
            f"For borderline candidates (fit {BORDERLINE[0]}-{BORDERLINE[1]}) give specific, "
            "actionable improvement suggestions.")),
        HumanMessage(content=(
            f"REQUIREMENTS\n{json.dumps({k: req[k] for k in ('role_title', 'must_have', 'nice_to_have', 'min_years') if k in req})}\n\n"
            f"ROUND-1 RETRIEVAL SCORE: {r1.get('match_score')}/100 "
            f"(semantic {r1.get('semantic_score')}, keyword {r1.get('keyword_score')})\n\n"
            f"RESUME ({candidate_id})\n{text}")),
    ]


def analyze_candidates(r1_list: list[dict], req: dict) -> dict[str, dict]:
    """Round 2: one deep LLM analysis per shortlisted candidate (run concurrently),
    falling back to the heuristic per candidate on failure. All resume text is
    fetched in a single MCP batch_process call before any LLM call starts."""
    results: dict[str, dict] = {}
    texts = resume_texts_batch([r["candidate_id"] for r in r1_list])
    try:
        batches = [_analysis_messages(r["candidate_id"], r, req, texts[r["candidate_id"]])
                  for r in r1_list]
        outputs = llm.structured_batch(CandidateAnalysis, batches)
    except llm.LLMUnavailable:
        outputs = [None] * len(r1_list)
    for r, out in zip(r1_list, outputs):
        if isinstance(out, CandidateAnalysis):
            a = out.model_dump()
            a["fit_score"] = max(0, min(100, a["fit_score"]))
            a["source"] = "llm"
        else:
            a = heuristic_analysis(r, req)
        results[r["candidate_id"]] = a
    return results


# --------------------------------------------------------------------------- #
# Round 3: hire / no-hire
# --------------------------------------------------------------------------- #
def heuristic_verdict(cid: str, r1: dict, a: dict) -> dict:
    fit = a["fit_score"]
    if r1.get("meets_requirements") and fit >= 75:
        decision, conf = "hire", "high" if fit >= 85 else "medium"
    elif fit >= BORDERLINE[0]:
        decision, conf = "maybe", "medium"
    else:
        decision, conf = "no-hire", "high" if fit < 35 else "medium"
    why = a["gaps"][0] if a["gaps"] and decision != "hire" else (a["strengths"][0] if a["strengths"] else a["summary"])
    return {"candidate_id": cid, "decision": decision, "confidence": conf,
            "rationale": f"Fit {fit}/100. {why}.", "source": "heuristic"}


def final_verdicts(r1_list: list[dict], analyses: dict[str, dict], req: dict) -> dict[str, dict]:
    """Round 3: one LLM call that sees every round-2 summary side by side and
    issues calibrated hire / maybe / no-hire recommendations."""
    by_id = {r["candidate_id"]: r for r in r1_list}
    try:
        digest = [{"candidate_id": cid, "name": by_id[cid]["candidate_name"],
                   "years": by_id[cid]["experience_years"],
                   "meets_must_haves": by_id[cid]["meets_requirements"],
                   "fit_score": a["fit_score"], "strengths": a["strengths"], "gaps": a["gaps"]}
                  for cid, a in analyses.items()]
        out = llm.structured(Verdicts, [
            SystemMessage(content=(
                "You are the hiring panel making the final-round call. For EVERY candidate "
                "give hire / maybe / no-hire with a confidence and a one-to-two sentence "
                "rationale. Missing a must-have skill should normally mean no better than "
                "'maybe'. Be decisive and consistent across candidates.")),
            HumanMessage(content=f"REQUIREMENTS\n{json.dumps(req)}\n\nCANDIDATES\n{json.dumps(digest, indent=1)}"),
        ])
        verdicts = {v.candidate_id: {**v.model_dump(), "source": "llm"} for v in out.verdicts}
    except Exception:
        verdicts = {}
    for cid, a in analyses.items():
        if cid not in verdicts:
            verdicts[cid] = heuristic_verdict(cid, by_id[cid], a)
    return verdicts


# --------------------------------------------------------------------------- #
# Shortlist helpers
# --------------------------------------------------------------------------- #
def final_score(round1_score: int, fit_score: int) -> int:
    """Blend retrieval (round 1) and deep-analysis (round 2) scores."""
    return round(0.4 * round1_score + 0.6 * fit_score)


def find_in_shortlist(shortlist: list[dict], cid: str) -> dict | None:
    return next((c for c in shortlist if c["candidate_id"] == cid), None)


def profile(cid: str, ctx: dict) -> dict:
    """Shortlist entry for cid, or an on-the-fly profile for anyone else."""
    entry = find_in_shortlist(ctx.get("shortlist", []), cid)
    if entry:
        return entry
    req = ctx.get("requirements", {})
    r1 = score_candidate(cid, req)
    a = analyze_candidates([r1], req)[cid]
    return {**r1, "rank": None, "round1_score": r1["match_score"], "fit_score": a["fit_score"],
            "final_score": final_score(r1["match_score"], a["fit_score"]), "analysis": a,
            "verdict": heuristic_verdict(cid, r1, a)}


# --------------------------------------------------------------------------- #
# compare_candidates / explain_ranking / generate_interview_questions
# --------------------------------------------------------------------------- #
def compare_candidates(candidate_ids: list[str], ctx: dict) -> dict:
    """Head-to-head comparison: requirement coverage, scores, verdicts."""
    req = ctx.get("requirements", {})
    rows, errors, notes = [], [], []
    for ref in candidate_ids:
        try:
            cid, note = resolve_for(ref, ctx)
            p = profile(cid, ctx)
        except ValueError as e:
            errors.append(str(e))
            continue
        if note:
            notes.append(note)
        have = {s.lower() for s in p.get("skills", [])}
        rows.append({
            "candidate_id": p["candidate_id"], "name": p["candidate_name"], "rank": p.get("rank"),
            "experience_years": p["experience_years"],
            "final_score": p["final_score"], "round1_score": p["round1_score"],
            "fit_score": p["fit_score"], "decision": p["verdict"]["decision"],
            "must_have": {s: s.lower() in have for s in req.get("must_have", [])},
            "nice_to_have": {s: s.lower() in have for s in req.get("nice_to_have", [])},
            "strengths": p["analysis"]["strengths"][:3], "gaps": p["analysis"]["gaps"][:3],
        })
    table = _comparison_table(rows, req)
    best = max(rows, key=lambda r: r["final_score"])["name"] if rows else None
    return {"requirements": req, "candidates": rows, "table_md": table,
            "leader": best, "notes": notes, "errors": errors}


def _comparison_table(rows: list[dict], req: dict) -> str:
    if not rows:
        return ""
    head = "| | " + " | ".join(r["name"] for r in rows) + " |"
    sep = "|---|" + "---|" * len(rows)
    lines = [head, sep,
             "| Rank | " + " | ".join(str(r["rank"] or "-") for r in rows) + " |",
             "| Final score | " + " | ".join(str(r["final_score"]) for r in rows) + " |",
             "| Retrieval / Fit | " + " | ".join(f"{r['round1_score']} / {r['fit_score']}" for r in rows) + " |",
             "| Years | " + " | ".join(str(r["experience_years"]) for r in rows) + " |"]
    for s in req.get("must_have", []):
        lines.append(f"| {s} (must) | " + " | ".join("✅" if r["must_have"][s] else "❌" for r in rows) + " |")
    for s in req.get("nice_to_have", []):
        lines.append(f"| {s} (nice) | " + " | ".join("✅" if r["nice_to_have"][s] else "—" for r in rows) + " |")
    lines.append("| Verdict | " + " | ".join(r["decision"] for r in rows) + " |")
    return "\n".join(lines)


def explain_ranking(candidate_a: str, candidate_b: str, ctx: dict) -> dict:
    """Why does A rank above (or below) B? Score-component breakdown plus the
    concrete requirement differences that drive it."""
    req = ctx.get("requirements", {})
    try:
        (ida, note_a), (idb, note_b) = resolve_for(candidate_a, ctx), resolve_for(candidate_b, ctx)
        a, b = profile(ida, ctx), profile(idb, ctx)
    except ValueError as e:
        return {"error": str(e)}
    notes = [n for n in (note_a, note_b) if n]
    if b["final_score"] > a["final_score"]:
        a, b = b, a
    have_a = {s.lower() for s in a.get("skills", [])}
    have_b = {s.lower() for s in b.get("skills", [])}
    reasons = []
    for s in req.get("must_have", []):
        if (s.lower() in have_a) != (s.lower() in have_b):
            who = a if s.lower() in have_a else b
            reasons.append(f"{who['candidate_name']} has must-have {s}; the other does not")
    for s in req.get("nice_to_have", []):
        if (s.lower() in have_a) != (s.lower() in have_b):
            who = a if s.lower() in have_a else b
            reasons.append(f"{who['candidate_name']} has nice-to-have {s}")
    dy = a["experience_years"] - b["experience_years"]
    if dy:
        reasons.append(f"{a['candidate_name']} has {abs(dy)} {'more' if dy > 0 else 'fewer'} years of experience")
    components = {
        k: {"higher": a.get(k), "lower": b.get(k), "delta": (a.get(k) or 0) - (b.get(k) or 0)}
        for k in ("final_score", "round1_score", "semantic_score", "keyword_score", "fit_score")
    }
    return {
        "higher_ranked": {"name": a["candidate_name"], "rank": a.get("rank"),
                          "decision": a["verdict"]["decision"], "strengths": a["analysis"]["strengths"],
                          "gaps": a["analysis"]["gaps"]},
        "lower_ranked": {"name": b["candidate_name"], "rank": b.get("rank"),
                         "decision": b["verdict"]["decision"], "strengths": b["analysis"]["strengths"],
                         "gaps": b["analysis"]["gaps"]},
        "score_components": components,
        "scoring_formula": "final = 0.4 * round1 (0.6*semantic + 0.4*keyword) + 0.6 * round2 fit",
        "key_differences": reasons,
        "notes": notes + [f"{p['candidate_name']} is not in the current top {len(ctx.get('shortlist', []))}; "
                          f"scored on the fly" for p in (a, b) if p.get("rank") is None],
    }


def heuristic_questions(p: dict, req: dict) -> list[dict]:
    qs = []
    for s in [s for s in req.get("must_have", []) if s.lower() in {k.lower() for k in p.get("skills", [])}][:3]:
        qs.append({"question": f"Walk me through the most complex system you built with {s}. "
                               f"What trade-offs did you make?", "purpose": f"Verify depth in {s}"})
    for g in p["analysis"]["gaps"][:3]:
        topic = g.split(": ")[-1].replace("No evidence of ", "")
        qs.append({"question": f"Our role needs {topic}. What exposure have you had to it, "
                               f"and how would you ramp up in your first 60 days?",
                   "purpose": f"Probe gap: {topic}"})
    qs.append({"question": "Tell me about a production incident you owned end to end.",
               "purpose": "Ownership and judgement"})
    return qs


def generate_interview_questions(candidate_id: str, ctx: dict) -> dict:
    """Tailored screening questions: verify claimed strengths, probe gaps."""
    req = ctx.get("requirements", {})
    try:
        cid, _ = resolve_for(candidate_id, ctx)
    except ValueError as e:
        return {"error": str(e)}
    p = profile(cid, ctx)
    try:
        out = llm.structured(InterviewQuestions, [
            SystemMessage(content=(
                "Write 6-8 phone-screen questions for this candidate. Mix: questions that "
                "verify their claimed strengths with specifics, and questions that probe "
                "each gap against the requirements. No generic trivia.")),
            HumanMessage(content=(
                f"REQUIREMENTS\n{json.dumps(req)}\n\nANALYSIS\n{json.dumps(p['analysis'])}\n\n"
                f"RESUME\n{resume_text(cid)}")),
        ])
        questions, source = [q.model_dump() for q in out.questions], "llm"
    except Exception:
        questions, source = heuristic_questions(p, req), "heuristic"
    return {"candidate_id": cid, "name": p["candidate_name"], "questions": questions, "source": source}


# --------------------------------------------------------------------------- #
# Tool wrappers for the conversational node
# --------------------------------------------------------------------------- #
def shortlist_view(shortlist: list[dict]) -> list[dict]:
    return [{"rank": c["rank"], "candidate_id": c["candidate_id"], "name": c["candidate_name"],
             "final_score": c["final_score"], "decision": c["verdict"]["decision"],
             "experience_years": c["experience_years"],
             "missing_must_have": c.get("missing_must_have", [])} for c in shortlist]


def tool_specs(tools: list) -> list[dict]:
    """Tools (LangChain StructuredTool or mcp_bridge.MCPToolProxy) -> Claude
    Messages API tool definitions. A StructuredTool's args_schema is a pydantic
    model (auto-generated from the function signature); an MCPToolProxy's is
    already a raw JSON Schema dict, straight from the MCP server."""
    specs = []
    for t in tools:
        schema = t.args_schema
        if schema is None:
            schema = {"type": "object", "properties": {}}
        elif hasattr(schema, "model_json_schema"):
            schema = schema.model_json_schema()
        else:
            schema = dict(schema)
        schema.pop("title", None)
        specs.append({"name": t.name, "description": t.description, "input_schema": schema})
    return specs


def make_tools(ctx: dict) -> list:
    """Build the agent's tool set bound to a read-only snapshot of the state:
    Python tools (comparison, explanation, RAG search, requirement parsing)
    plus every tool from both MCP servers (file access, batch/watch, the
    candidate/job database) — see mcp_bridge.py. If the MCP servers can't be
    reached, chat still works for everything except file/database tools."""

    def rag_search(query: str, top_k: int = 5, must_have: list[str] | None = None, min_years: int = 0) -> dict:
        """Semantic + keyword search over all indexed resumes (does NOT change the current shortlist)."""
        out = get_matcher().match(query, top_k=top_k, must_have=_canon_list(must_have or []),
                                  min_years=min_years, pool=get_matcher().col.count())
        return {"candidates_scored": out["candidates_scored"],
                "matches": [{k: r[k] for k in ("candidate_name", "resume_path", "match_score",
                                               "experience_years", "matched_skills",
                                               "missing_must_have", "meets_requirements")}
                            for r in out["top_matches"]]}

    def extract_requirements_tool(jd: str) -> dict:
        """Parse a job description into must-have vs nice-to-have skills and minimum years."""
        return extract_requirements(jd)

    def get_shortlist() -> dict:
        """Current requirements and ranked shortlist (rank, id, score, verdict)."""
        return {"requirements": ctx.get("requirements", {}),
                "shortlist": shortlist_view(ctx.get("shortlist", []))}

    def get_candidate_analysis(candidate: str) -> dict:
        """Full round-2 analysis + round-3 verdict for one candidate (name, id, or rank like '#2')."""
        try:
            p = profile(resolve_for(candidate, ctx)[0], ctx)
        except ValueError as e:
            return {"error": str(e)}
        return {k: p.get(k) for k in ("rank", "candidate_id", "candidate_name", "final_score",
                                      "round1_score", "semantic_score", "keyword_score", "fit_score",
                                      "experience_years", "skills", "missing_must_have",
                                      "analysis", "verdict")}

    def get_rank_changes() -> dict:
        """How the ranking moved after the last requirement refinement, with per-candidate reasons."""
        return {"requirement_changes": ctx.get("requirement_changes", []),
                "rank_changes": ctx.get("rank_changes", []),
                "requirements_history": [{k: r.get(k) for k in ("version", "must_have", "nice_to_have", "min_years")}
                                         for r in ctx.get("requirements_history", [])]}

    def compare_candidates_tool(candidate_ids: list[str]) -> dict:
        """Head-to-head comparison of 2+ candidates (names, ids, or ranks like '1','2','3')."""
        return compare_candidates(candidate_ids, ctx)

    def explain_ranking_tool(candidate_a: str, candidate_b: str) -> dict:
        """Explain why one candidate ranks higher than another, with score components and key differences."""
        return explain_ranking(candidate_a, candidate_b, ctx)

    def generate_interview_questions_tool(candidate_id: str) -> dict:
        """Create tailored screening questions for one candidate (name, id, or rank)."""
        return generate_interview_questions(candidate_id, ctx)

    fns = [
        (rag_search, "rag_search"),
        (extract_requirements_tool, "extract_requirements"), (get_shortlist, "get_shortlist"),
        (get_candidate_analysis, "get_candidate_analysis"), (get_rank_changes, "get_rank_changes"),
        (compare_candidates_tool, "compare_candidates"), (explain_ranking_tool, "explain_ranking"),
        (generate_interview_questions_tool, "generate_interview_questions"),
    ]
    tools: list = [StructuredTool.from_function(fn, name=name) for fn, name in fns]
    try:
        tools += mcp_bridge.tool_proxies()
    except mcp_bridge.MCPUnavailable:
        pass   # chat still works without file/database tools; a direct call will say why
    return tools
