"""Unit tests for the agent tools (offline)."""

import agent_tools as T


def test_heuristic_requirements_split_must_vs_nice():
    jd = open(f"{T.BASE_DIR}/jobs/jd_platform_engineer.txt").read()
    req = T.extract_requirements(jd)
    assert set(req["must_have"]) == {"Kubernetes", "Go"}
    assert {"Terraform", "Prometheus", "AWS"} <= set(req["nice_to_have"])
    assert req["min_years"] == 5


def test_natural_language_query_requirements():
    req = T.extract_requirements("Find me candidates with React and 3+ years experience")
    assert req["must_have"] == ["React"]
    assert req["min_years"] == 3


def test_canonical_skill_aliases():
    assert T.canonical_skill("k8s") == "Kubernetes"
    assert T.canonical_skill("golang") == "Go"
    assert T.canonical_skill("Underwater Basket Weaving") == "Underwater Basket Weaving"


def test_round1_scores_all_100_resumes():
    req = T.extract_requirements(open(f"{T.BASE_DIR}/jobs/jd_backend_python.txt").read())
    out = T.round1_search(req, top_k=10)
    assert out["candidates_scored"] == 100
    assert len(out["top_matches"]) == 10
    flags = [m["meets_requirements"] for m in out["top_matches"]]
    assert flags == sorted(flags, reverse=True)   # qualified candidates first


def test_resolve_candidate_by_id_name_and_rank():
    assert T.resolve_candidate("John_Doe") == "John_Doe"
    assert T.resolve_candidate("john doe") == "John_Doe"
    assert T.resolve_candidate("#2", ["A", "John_Doe"]) == "John_Doe"
    # ambiguous first name resolves to the shortlisted one
    assert T.resolve_candidate("John", ["Jane_Smith", "John_Khan"]) == "John_Khan"


def test_resolve_for_picks_best_scoring_when_ambiguous():
    req = T.extract_requirements("Senior ML engineer, Python and Machine Learning, 5+ years")
    cid, note = T.resolve_for("Jane", {"requirements": req, "shortlist": []})
    assert cid.startswith("Jane_") and "matches 3 candidates" in note


def test_make_tools_exposes_required_tools():
    names = {t.name for t in T.make_tools({"requirements": {}, "shortlist": []})}
    assert {"read_file", "list_files", "write_file", "search_in_file", "rag_search",
            "extract_requirements", "compare_candidates", "generate_interview_questions",
            "explain_ranking"} <= names


def test_make_tools_includes_both_mcp_servers():
    """read_file etc. come from the filesystem MCP server; batch_process and
    watch_directory are its MCP-native extras; query_candidates etc. come from
    the second (bonus) resume_db MCP server."""
    names = {t.name for t in T.make_tools({"requirements": {}, "shortlist": []})}
    assert {"batch_process", "watch_directory"} <= names
    assert {"query_candidates", "get_candidate_record", "list_roles", "get_job_posting"} <= names


def test_tool_specs_handles_both_schema_shapes():
    """tool_specs() must build a valid input_schema for both a LangChain
    StructuredTool (pydantic args_schema) and an MCPToolProxy (raw dict
    args_schema) - this broke once already when they were assumed uniform."""
    specs = T.tool_specs(T.make_tools({"requirements": {}, "shortlist": []}))
    by_name = {s["name"]: s for s in specs}
    assert by_name["rag_search"]["input_schema"]["properties"]   # StructuredTool-sourced
    assert by_name["read_file"]["input_schema"]["properties"]    # MCPToolProxy-sourced
    assert by_name["read_file"]["input_schema"]["required"] == ["filepath"]


def test_rag_search_tool():
    tool = {t.name: t for t in T.make_tools({})}["rag_search"]
    out = tool.invoke({"query": "Kubernetes Terraform SRE", "top_k": 3, "must_have": ["k8s"]})
    assert out["candidates_scored"] == 100 and len(out["matches"]) == 3


def test_mcp_file_tools_round_trip():
    tools = {t.name: t for t in T.make_tools({"requirements": {}, "shortlist": []})}
    out = tools["read_file"].invoke({"filepath": "resumes/John_Doe.txt"})
    assert out["success"] is True and "John Doe" in out["content"]

    missing = tools["read_file"].invoke({"filepath": "resumes/NOPE.txt"})
    assert missing["success"] is False


def test_resume_text_and_batch_match_by_relative_path():
    """Regression guard: batch_process's per-file results must be keyed by the
    caller's original relative path, not the server's resolved absolute path -
    resume_texts_batch silently returned empty text for every candidate when
    this was broken."""
    single = T.resume_text("John_Doe")
    assert "John Doe" in single and len(single) > 100

    batch = T.resume_texts_batch(["John_Doe", "Jane_Smith"])
    assert set(batch) == {"John_Doe", "Jane_Smith"}
    assert all(len(t) > 100 for t in batch.values())
    assert batch["John_Doe"] == single
