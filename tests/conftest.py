"""Test setup: offline mode by default (deterministic heuristics, no API spend),
and make sure the Chroma index exists."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

LIVE = os.environ.get("LIVE") == "1"
if not LIVE:
    os.environ["MATCHING_AGENT_OFFLINE"] = "1"

import pytest  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "live: hits the real LLM (run with LIVE=1)")


def pytest_collection_modifyitems(config, items):
    if LIVE:
        return
    skip = pytest.mark.skip(reason="live LLM test; run with LIVE=1")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def index():
    from agent_tools import get_matcher
    from resume_rag import ResumeIndex

    if get_matcher().col.count() == 0:
        ResumeIndex().build()
        get_matcher.cache_clear()
    yield
