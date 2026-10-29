"""Resume RAG: document processing, embedding, and vector storage.

Pipeline:
  load (fs_tools) -> section-aware chunk -> extract metadata -> embed (MiniLM)
  -> store in ChromaDB (one record per chunk, with per-resume metadata).

The index is the retrieval backend used by job_matcher.py.

CLI:
  python resume_rag.py build          # (re)build the index from ./resumes
  python resume_rag.py stats          # show what's indexed
"""

from __future__ import annotations

import os
import re
import sys

import chromadb
from sentence_transformers import SentenceTransformer

import fs_tools

EMBED_MODEL = "all-MiniLM-L6-v2"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chroma_db")
RESUME_DIR = os.path.join(BASE_DIR, "resumes")
COLLECTION = "resumes"

# Canonical skills we recognize for metadata + keyword matching. Lowercase ->
# display form. Keep broad so extraction works across roles.
KNOWN_SKILLS = {
    "python", "java", "javascript", "typescript", "go", "rust", "c++", "swift",
    "kotlin", "ruby", "scala", "r", "sql",
    "django", "flask", "fastapi", "react", "next.js", "node.js", "express",
    "redux", "graphql", "rest apis", "html", "css",
    "machine learning", "deep learning", "nlp", "pytorch", "tensorflow",
    "scikit-learn", "pandas", "numpy", "statistics", "mlops", "spark", "airflow",
    "kafka", "snowflake", "dbt", "etl", "a/b testing", "experimentation",
    "aws", "gcp", "azure", "docker", "kubernetes", "terraform", "ansible",
    "ci/cd", "prometheus", "grafana", "linux", "microservices", "redis",
    "postgresql", "mongodb", "celery", "grpc",
    "selenium", "cypress", "playwright", "pytest", "test automation",
    "api testing", "performance testing",
    "threat modeling", "penetration testing", "siem", "burp suite",
    "cryptography", "soc", "incident response", "slos",
    "ios", "android", "mobile", "react native", "flutter",
    "roadmapping", "analytics", "agile", "figma", "prioritization",
    "stakeholder management", "go-to-market",
}

SECTION_HEADINGS = ["summary", "objective", "skills", "experience",
                    "work experience", "education", "projects", "certifications"]


# --------------------------------------------------------------------------- #
# Loading + chunking
# --------------------------------------------------------------------------- #
def load_resume_text(path: str) -> str:
    """Load a resume to plain text via the Milestone 1 file tools."""
    result = fs_tools.read_file(path)
    if not result["success"]:
        raise RuntimeError(f"Could not read {path}: {result['error']}")
    return result["content"]


def split_sections(text: str) -> dict[str, str]:
    """Split a resume into sections keyed by heading.

    A line is treated as a heading if, stripped and lowercased, it matches one
    of SECTION_HEADINGS (resumes here use UPPERCASE headings, but we match
    case-insensitively). Text before the first heading goes under 'header'.
    """
    sections: dict[str, list[str]] = {"header": []}
    current = "header"
    for line in text.splitlines():
        key = line.strip().lower().rstrip(":")
        if key in SECTION_HEADINGS:
            current = key
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items() if "\n".join(v).strip()}


def chunk_resume(path: str, text: str) -> list[dict]:
    """Produce section-preserving chunks.

    One chunk per section (Education/Experience/Skills/etc. stay intact). The
    'header' (name/title/contact) is merged into the summary chunk for context.
    Long sections (e.g. Experience) are left whole — resumes are short enough
    that section granularity is the right retrieval unit.
    """
    sections = split_sections(text)
    header = sections.pop("header", "")
    chunks = []
    for name, body in sections.items():
        content = f"{header}\n\n{body}".strip() if name in ("summary", "objective") else body
        chunks.append({"section": name, "text": content})
    if not chunks:  # no recognized headings — fall back to one whole-doc chunk
        chunks.append({"section": "full", "text": text.strip()})
    return chunks


# --------------------------------------------------------------------------- #
# Metadata extraction
# --------------------------------------------------------------------------- #
def extract_name(text: str) -> str:
    """First non-empty line is the candidate name in our format."""
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return "Unknown"


def extract_skills(text: str) -> list[str]:
    """Find known skills mentioned anywhere in the resume (display-cased)."""
    low = text.lower()
    found = []
    for skill in KNOWN_SKILLS:
        # word-ish boundary match; escape regex metachars in the skill
        if re.search(r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])", low):
            found.append(skill)
    # Prefer a Skills section's own order/casing when present
    return sorted({_display_skill(s) for s in found})


def _display_skill(skill_lower: str) -> str:
    special = {
        "nlp": "NLP", "aws": "AWS", "gcp": "GCP", "sql": "SQL", "ci/cd": "CI/CD",
        "rest apis": "REST APIs", "mlops": "MLOps", "ios": "iOS",
        "a/b testing": "A/B Testing", "html": "HTML", "css": "CSS",
        "etl": "ETL", "siem": "SIEM", "soc": "SOC", "slos": "SLOs",
        "node.js": "Node.js", "next.js": "Next.js", "dbt": "dbt",
        "c++": "C++", "r": "R", "javascript": "JavaScript",
        "typescript": "TypeScript", "postgresql": "PostgreSQL",
        "mongodb": "MongoDB", "graphql": "GraphQL", "fastapi": "FastAPI",
        "pytorch": "PyTorch", "tensorflow": "TensorFlow", "scikit-learn": "scikit-learn",
    }
    if skill_lower in special:
        return special[skill_lower]
    return " ".join(w.capitalize() for w in skill_lower.split())


def extract_years(text: str) -> int:
    """Years of experience: prefer an explicit 'N years' phrase, else infer
    from the earliest 4-digit year in the Experience section."""
    m = re.search(r"(\d+)\+?\s*years?", text.lower())
    if m:
        return int(m.group(1))
    years = [int(y) for y in re.findall(r"\b(19|20)\d{2}\b", text)]
    if years:
        return max(0, 2026 - min(years))
    return 0


def extract_education(text: str) -> str:
    sections = split_sections(text)
    edu = sections.get("education", "")
    return edu.splitlines()[0].strip() if edu else ""


def extract_metadata(path: str, text: str) -> dict:
    skills = extract_skills(text)
    return {
        "resume_path": path,
        "candidate_name": extract_name(text),
        "skills": skills,
        "skills_joined": ", ".join(skills),   # Chroma metadata must be scalar
        "experience_years": extract_years(text),
        "education": extract_education(text),
    }


# --------------------------------------------------------------------------- #
# Index build / access
# --------------------------------------------------------------------------- #
class ResumeIndex:
    def __init__(self, db_path: str = DB_PATH, model_name: str = EMBED_MODEL):
        self.client = chromadb.PersistentClient(path=db_path)
        self.model = SentenceTransformer(model_name)

    def _collection(self, reset: bool = False):
        if reset:
            try:
                self.client.delete_collection(COLLECTION)
            except Exception:
                pass
        return self.client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    def embed(self, texts: list[str]):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def build(self, resume_dir: str = RESUME_DIR) -> dict:
        col = self._collection(reset=True)
        files = fs_tools.list_files(resume_dir)
        ids, docs, metas = [], [], []
        resume_count = 0
        for f in files:
            if f.get("extension") not in (".txt", ".pdf", ".docx"):
                continue
            path = f["path"]
            text = load_resume_text(path)
            meta = extract_metadata(os.path.join("resumes", f["name"]), text)
            for i, chunk in enumerate(chunk_resume(path, text)):
                ids.append(f"{f['name']}::{chunk['section']}::{i}")
                docs.append(chunk["text"])
                metas.append({**meta, "section": chunk["section"]})
            resume_count += 1
        embeddings = self.embed(docs)
        col.add(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
        return {"resumes": resume_count, "chunks": len(ids)}

    def collection(self):
        return self._collection(reset=False)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    idx = ResumeIndex()
    if cmd == "build":
        stats = idx.build()
        print(f"Indexed {stats['resumes']} resumes into {stats['chunks']} chunks "
              f"at {DB_PATH}")
    elif cmd == "stats":
        col = idx.collection()
        got = col.get(include=["metadatas"])
        names = sorted({m["candidate_name"] for m in got["metadatas"]})
        print(f"Collection '{COLLECTION}': {col.count()} chunks, "
              f"{len(names)} candidates")
        for n in names[:5]:
            print("  -", n)
    else:
        print(f"unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
