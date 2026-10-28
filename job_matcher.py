"""Job matching engine: hybrid semantic + keyword search over the resume index.

Given a job description, embed it, retrieve the most similar resume chunks,
aggregate to the candidate level, apply hybrid keyword boosting for critical
skills, filter by must-have requirements, score 0-100, and emit structured
matches with reasoning.

CLI:
  python job_matcher.py jobs/jd_ml_engineer.txt
  python job_matcher.py jobs/jd_ml_engineer.txt --must-have Python,"Machine Learning" --min-years 5
  python job_matcher.py "Senior Python engineer, 5+ years, ML and NLP" --top-k 10
"""

from __future__ import annotations

import argparse
import json
import os
import re

from resume_rag import ResumeIndex, KNOWN_SKILLS, _display_skill

TOP_K = 10
CANDIDATE_POOL = 60   # chunks pulled before aggregating to candidates

# Scoring weights (sum = 1.0): semantic similarity vs. critical-skill keyword overlap.
W_SEMANTIC = 0.6
W_KEYWORD = 0.4


def extract_jd_skills(jd_text: str) -> list[str]:
    """Known skills explicitly named in the JD (used for hybrid keyword match)."""
    low = jd_text.lower()
    found = [s for s in KNOWN_SKILLS
             if re.search(r"(?<![a-z0-9])" + re.escape(s) + r"(?![a-z0-9])", low)]
    return sorted({_display_skill(s) for s in found})


def _as_skill_set(joined: str) -> set[str]:
    return {s.strip().lower() for s in joined.split(",") if s.strip()}


class JobMatcher:
    def __init__(self, index: ResumeIndex | None = None):
        self.index = index or ResumeIndex()
        self.col = self.index.collection()

    def match(self, jd_text: str, top_k: int = TOP_K,
              must_have: list[str] | None = None,
              min_years: int = 0,
              w_semantic: float = W_SEMANTIC,
              w_keyword: float = W_KEYWORD,
              skills: list[str] | None = None,
              pool: int | None = None) -> dict:
        """Rank candidates for a JD.

        skills: keyword set for the hybrid score (defaults to skills named in
            the JD). The agent passes its must-have + nice-to-have list here so
            refined requirements change the ranking.
        pool: number of chunks to retrieve before aggregating. None keeps the
            Milestone 2 default; pass self.col.count() to score every resume.
        """
        jd_skills = skills if skills is not None else extract_jd_skills(jd_text)
        must_have = must_have or []
        must_lower = {m.strip().lower() for m in must_have}

        # --- semantic retrieval over chunks ---
        q_emb = self.index.embed([jd_text])
        if pool is None:
            pool = min(CANDIDATE_POOL, max(top_k * 4, 40))
        pool = max(1, min(pool, self.col.count()))
        res = self.col.query(
            query_embeddings=q_emb, n_results=pool,
            include=["documents", "metadatas", "distances"],
        )
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]

        # --- aggregate chunks -> candidates (best similarity + collected excerpts) ---
        candidates: dict[str, dict] = {}
        for doc, meta, dist in zip(docs, metas, dists):
            sim = 1.0 - dist  # cosine distance -> similarity
            name = meta["candidate_name"]
            c = candidates.setdefault(name, {
                "candidate_name": name,
                "resume_path": meta["resume_path"],
                "experience_years": meta.get("experience_years", 0),
                "education": meta.get("education", ""),
                "skills": sorted(_as_skill_set(meta.get("skills_joined", ""))),
                "best_sim": 0.0,
                "excerpts": [],
            })
            if sim > c["best_sim"]:
                c["best_sim"] = sim
            # keep the strongest excerpt per section
            c["excerpts"].append((sim, meta.get("section", ""), doc))

        # --- score, keyword overlap, must-have filter ---
        jd_skill_set = {s.lower() for s in jd_skills}
        results = []
        for c in candidates.values():
            cand_skills = {s.lower() for s in c["skills"]}

            matched = sorted(jd_skill_set & cand_skills)
            keyword_overlap = (len(matched) / len(jd_skill_set)) if jd_skill_set else 0.0

            missing_must = sorted(must_lower - cand_skills)
            meets_years = c["experience_years"] >= min_years

            score = round(100 * (w_semantic * max(0.0, c["best_sim"])
                                 + w_keyword * keyword_overlap))

            c["excerpts"].sort(key=lambda x: x[0], reverse=True)
            top_excerpts = [self._trim(doc) for _, sec, doc in c["excerpts"][:2]]
            matched_display = [_display_skill(s) for s in matched]

            results.append({
                "candidate_name": c["candidate_name"],
                "resume_path": c["resume_path"],
                "match_score": score,
                "semantic_score": round(100 * max(0.0, c["best_sim"])),
                "keyword_score": round(100 * keyword_overlap),
                "skills": [_display_skill(s) for s in c["skills"]],
                "education": c["education"],
                "experience_years": c["experience_years"],
                "matched_skills": matched_display,
                "missing_must_have": [_display_skill(s) for s in missing_must],
                "meets_requirements": (not missing_must) and meets_years,
                "relevant_excerpts": top_excerpts,
                "reasoning": self._reason(c, matched_display, missing_must,
                                          meets_years, min_years, score),
            })

        # must-have filter: candidates meeting all requirements rank first,
        # then by score. (Keeps near-misses visible but below qualified ones.)
        results.sort(key=lambda r: (r["meets_requirements"], r["match_score"]),
                     reverse=True)
        return {"job_description": jd_text.strip(),
                "requirements": {"must_have": must_have, "min_years": min_years},
                "candidates_scored": len(results),
                "top_matches": results[:top_k]}

    @staticmethod
    def _trim(text: str, limit: int = 220) -> str:
        t = " ".join(text.split())
        return t if len(t) <= limit else t[:limit].rsplit(" ", 1)[0] + "..."

    @staticmethod
    def _reason(c, matched, missing_must, meets_years, min_years, score) -> str:
        bits = []
        if matched:
            bits.append(f"Matches {len(matched)} required/related skill(s): "
                        f"{', '.join(matched[:6])}.")
        else:
            bits.append("No explicit skill overlap with the job description.")
        bits.append(f"{c['experience_years']} years of experience"
                    + (f" (meets {min_years}+ requirement)." if meets_years and min_years
                       else f" (below {min_years}+ requirement)." if min_years else "."))
        if missing_must:
            bits.append(f"Missing must-have: {', '.join(_display_skill(m) for m in missing_must)}.")
        bits.append(f"Semantic relevance {round(c['best_sim'] * 100)}/100; overall {score}/100.")
        return " ".join(bits)


def _read_jd(arg: str) -> str:
    if os.path.isfile(arg):
        with open(arg, encoding="utf-8") as f:
            return f.read()
    return arg


def main():
    ap = argparse.ArgumentParser(description="Match resumes to a job description.")
    ap.add_argument("jd", help="Path to a JD file, or the JD text itself.")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--must-have", default="",
                    help="Comma-separated must-have skills (e.g. 'Python,Machine Learning').")
    ap.add_argument("--min-years", type=int, default=0)
    args = ap.parse_args()

    must_have = [s.strip() for s in args.must_have.split(",") if s.strip()]
    matcher = JobMatcher()
    out = matcher.match(_read_jd(args.jd), top_k=args.top_k,
                        must_have=must_have, min_years=args.min_years)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
