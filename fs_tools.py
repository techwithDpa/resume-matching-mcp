"""File system tools for an LLM file assistant.

Each tool returns a JSON-serializable structure (dict or list) so results can be
handed straight back to an LLM as a tool result. Errors are reported in the
return value rather than raised, so the model can read and react to them.

Supported resume formats for read_file / search_in_file: .txt, .pdf, .docx
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

# Optional format handlers. Imported lazily-friendly at module load; if a
# library is missing we degrade gracefully and report it per-file instead of
# crashing the whole module.
try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None

try:
    import docx  # python-docx
except ImportError:  # pragma: no cover
    docx = None

TEXT_EXTENSIONS = {".txt", ".md", ".text", ""}


def _file_metadata(filepath: str) -> dict:
    """Collect size and modified-time metadata for a file."""
    stat = os.stat(filepath)
    return {
        "name": os.path.basename(filepath),
        "path": os.path.abspath(filepath),
        "extension": os.path.splitext(filepath)[1].lower(),
        "size_bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def _extract_text(filepath: str, extension: str) -> str:
    """Extract plain text from a supported file based on its extension."""
    if extension == ".pdf":
        if PdfReader is None:
            raise RuntimeError("pypdf is not installed; cannot read PDF files")
        reader = PdfReader(filepath)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()

    if extension == ".docx":
        if docx is None:
            raise RuntimeError("python-docx is not installed; cannot read DOCX files")
        document = docx.Document(filepath)
        return "\n".join(p.text for p in document.paragraphs).strip()

    if extension in TEXT_EXTENSIONS:
        with open(filepath, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()

    raise ValueError(f"Unsupported file type: {extension or '(no extension)'}")


def read_file(filepath: str) -> dict:
    """Read a resume file (PDF, TXT, or DOCX) and return its text and metadata.

    Returns a dict:
        {
          "success": bool,
          "filepath": str,
          "content": str | None,     # extracted text when successful
          "metadata": dict | None,   # name, size_bytes, modified, extension
          "error": str | None,
        }
    """
    result = {
        "success": False,
        "filepath": filepath,
        "content": None,
        "metadata": None,
        "error": None,
    }

    if not os.path.exists(filepath):
        result["error"] = f"File not found: {filepath}"
        return result
    if not os.path.isfile(filepath):
        result["error"] = f"Not a file: {filepath}"
        return result

    extension = os.path.splitext(filepath)[1].lower()
    try:
        content = _extract_text(filepath, extension)
    except Exception as exc:  # report, don't crash
        result["metadata"] = _file_metadata(filepath)
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["success"] = True
    result["content"] = content
    result["metadata"] = _file_metadata(filepath)
    result["metadata"]["char_count"] = len(content)
    return result


def list_files(directory: str, extension: str | None = None) -> list:
    """List files in a directory, optionally filtered by extension.

    Args:
        directory: directory to list.
        extension: e.g. ".pdf" or "pdf" to filter; None lists every file.

    Returns a list of metadata dicts (name, path, extension, size_bytes,
    modified). On error, returns a single-element list containing an error dict:
        [{"success": False, "error": "..."}]
    """
    if not os.path.isdir(directory):
        return [{"success": False, "error": f"Directory not found: {directory}"}]

    normalized_ext = None
    if extension:
        normalized_ext = extension.lower()
        if not normalized_ext.startswith("."):
            normalized_ext = "." + normalized_ext

    files = []
    for entry in sorted(os.listdir(directory)):
        full_path = os.path.join(directory, entry)
        if not os.path.isfile(full_path):
            continue
        if normalized_ext and os.path.splitext(entry)[1].lower() != normalized_ext:
            continue
        files.append(_file_metadata(full_path))
    return files


def write_file(filepath: str, content: str) -> dict:
    """Write content to a file, creating parent directories as needed.

    Returns a dict:
        {"success": bool, "filepath": str, "bytes_written": int | None,
         "error": str | None}
    """
    result = {
        "success": False,
        "filepath": filepath,
        "bytes_written": None,
        "error": None,
    }
    try:
        parent = os.path.dirname(os.path.abspath(filepath))
        os.makedirs(parent, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as handle:
            handle.write(content)
        result["success"] = True
        result["bytes_written"] = len(content.encode("utf-8"))
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def search_in_file(filepath: str, keyword: str, context_chars: int = 60) -> dict:
    """Case-insensitive search for a keyword in a file's text content.

    Reads the file via read_file (so PDF/DOCX/TXT all work), then returns each
    match with surrounding context.

    Returns a dict:
        {
          "success": bool,
          "filepath": str,
          "keyword": str,
          "match_count": int,
          "matches": [
            {"line_number": int, "line": str, "context": str, "offset": int}
          ],
          "error": str | None,
        }
    """
    result = {
        "success": False,
        "filepath": filepath,
        "keyword": keyword,
        "match_count": 0,
        "matches": [],
        "error": None,
    }

    if not keyword:
        result["error"] = "keyword must be a non-empty string"
        return result

    read_result = read_file(filepath)
    if not read_result["success"]:
        result["error"] = read_result["error"]
        return result

    content = read_result["content"]
    lowered = content.lower()
    needle = keyword.lower()

    # Precompute line boundaries so each match can report its line number.
    lines = content.splitlines()
    line_starts = []
    running = 0
    for line in lines:
        line_starts.append(running)
        running += len(line) + 1  # +1 for the split newline

    def line_number_for(offset: int) -> int:
        lo, hi = 0, len(line_starts) - 1
        result_line = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if line_starts[mid] <= offset:
                result_line = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return result_line + 1  # 1-indexed

    start = 0
    while True:
        idx = lowered.find(needle, start)
        if idx == -1:
            break
        ctx_start = max(0, idx - context_chars)
        ctx_end = min(len(content), idx + len(keyword) + context_chars)
        ln = line_number_for(idx)
        result["matches"].append(
            {
                "line_number": ln,
                "line": lines[ln - 1].strip() if ln - 1 < len(lines) else "",
                "context": content[ctx_start:ctx_end].replace("\n", " ").strip(),
                "offset": idx,
            }
        )
        start = idx + len(needle)

    result["success"] = True
    result["match_count"] = len(result["matches"])
    return result
