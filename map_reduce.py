"""Map-reduce LLM analysis pipeline.

Map: analyze each session independently and emit structured findings.
Reduce: synthesize cross-session findings into a final report.
Interactive: answer follow-up questions with full context.
"""

from __future__ import annotations

import re
import textwrap

from openai import OpenAI

from config import (
    EXIT_CODE_RE,
    IMPORTANCE_HIGH,
    IMPORTANCE_LOW,
    IMPORTANCE_MEDIUM,
    INTERACTIVE_MAX_TOKENS,
    MAP_JSON_SCHEMA,
    MAP_MAX_TOKENS,
    MODEL,
    REDUCE_MAX_TOKENS,
    ROLE_DELEGATOR,
    ROLE_UNKNOWN,
    ROLE_EXPECTATIONS,
)
from fact_sheet import (
    MapResult,
    SessionFactSheet,
    _parse_map_response,
    build_fact_sheet,
    format_fact_sheet,
)
from llm_client import (
    _extract_content,
    _is_budget_error,
    _stream_response,
)
from parsing import LogEntry, Session, ToolCall, _extract_bash_command
from prefilter import PrefilterFindings, _format_session_prefilter_flags
from utils import _apply_text_compression, _normalize_paths, _truncate, _deduplicate_lines


# ── Iteration Importance Scoring ─────────────────────────────────────────────


def _score_iteration_importance(
    entry: LogEntry, session: Session, entry_index: int,
) -> str:
    """Score an iteration's diagnostic importance. Deterministic and auditable.

    HIGH:   errors, markers, write_file calls, first/last iteration
    MEDIUM: has tool calls but no errors, markers, or writes
    LOW:    no tool calls at all
    """
    is_first = entry_index == 0
    is_last = entry_index == len(session.entries) - 1

    if is_first or is_last:
        return IMPORTANCE_HIGH

    has_err = (entry.has_tool_errors if session.role == ROLE_DELEGATOR
               else entry.has_errors)
    if has_err:
        return IMPORTANCE_HIGH

    if entry.markers_found:
        return IMPORTANCE_HIGH

    if any(tc.tool_name == "write_file" for tc in entry.tool_calls):
        return IMPORTANCE_HIGH

    if entry.tool_calls:
        return IMPORTANCE_MEDIUM

    return IMPORTANCE_LOW


def _tool_call_signature(tc: ToolCall) -> tuple:
    """Generate a deduplication key for a tool call.

    Returns (tool_name, command_or_args_prefix, exit_code).
    """
    if tc.tool_name == "bash":
        cmd = _extract_bash_command(tc.args)
    else:
        cmd = tc.args[:200]
    exit_match = EXIT_CODE_RE.search(tc.output)
    exit_code = int(exit_match.group(1)) if exit_match else None
    return (tc.tool_name, cmd, exit_code)


def _is_successful_output(output: str) -> bool:
    """Check if a tool output indicates success (no error patterns detected)."""
    from config import ERROR_PATTERNS
    if any(p.search(output) for p in ERROR_PATTERNS):
        return False
    exit_match = EXIT_CODE_RE.search(output)
    if exit_match and int(exit_match.group(1)) != 0:
        return False
    return True


def _compress_tool_output(tc: ToolCall) -> str:
    """Compress tool output for successful operations to save tokens.

    - write_file success: replace with META annotation (content already
      visible in the Response section where the LLM generated the tool call).
    - bash exit code 0 without errors: keep first/last few lines only.
    - All others: return output unchanged.
    """
    if not tc.output:
        return ""

    # write_file success: content is already in the Response section
    if tc.tool_name == "write_file" and _is_successful_output(tc.output):
        path_match = re.search(r"path[=:]\s*(\S+)", tc.args)
        fname = path_match.group(1) if path_match else "file"
        return (f"[META: write_file returned success for {fname}. "
                f"File content is visible in the Response section above.]")

    # bash with exit code 0 and no error patterns: keep first/last lines
    if tc.tool_name == "bash" and _is_successful_output(tc.output):
        lines = tc.output.splitlines()
        if len(lines) <= 10:
            return tc.output  # Short enough, keep in full
        head = "\n".join(lines[:3])
        tail = "\n".join(lines[-3:])
        return (f"{head}\n"
                f"  [... {len(lines) - 6} lines omitted (exit code 0, no errors) ...]\n"
                f"{tail}")

    # Default: return as-is
    return tc.output


# ── Session Digest ───────────────────────────────────────────────────────────


def _build_session_digest(
    session: Session,
    project_root: str = "",
    is_clean: bool = False,
    fact_sheet: SessionFactSheet | None = None,
    skip_reasoning: bool = False,
) -> str:
    """Build a condensed text digest of a session for the LLM.

    Args:
        session: The session to digest.
        project_root: If set, long paths are replaced with $ROOT alias.
        is_clean: If True, return a minimal one-line summary (no detail).
        fact_sheet: If provided, prepend verified facts for the LLM.
        skip_reasoning: If True, omit all [REASONING] blocks to save tokens.
    """
    # Clean sessions: one-line summary — no diagnostic value in full detail
    if is_clean:
        return (f"SESSION {session.session_id} | {session.role} | "
                f"{session.total_iterations} iterations | "
                f"{session.outcome} | 0 errors | no issues detected")

    parts = []

    # Prepend fact sheet as ground truth for the LLM
    if fact_sheet:
        parts.append(format_fact_sheet(fact_sheet))
        parts.append("")

    # Path legend so the LLM knows what $ROOT means
    if project_root:
        parts.append(f"[Path legend: $ROOT = {project_root}]")

    parts.append(f"SESSION {session.session_id}")
    parts.append(f"Role: {session.role}")
    parts.append(f"Time: {session.start_time:%Y-%m-%d %H:%M:%S} → {session.end_time:%H:%M:%S}")
    parts.append(f"Iterations: {session.total_iterations}, Outcome: {session.outcome}")
    parts.append(f"Errors detected: {session.error_count}")
    parts.append("")

    # Track tool call signatures for cross-iteration deduplication.
    # Maps signature -> (first_iteration_number, compressed_output).
    seen_tool_calls: dict[tuple, tuple[int, str]] = {}

    for idx, entry in enumerate(session.entries):
        importance = _score_iteration_importance(entry, session, idx)

        # LOW importance: one-line summary — no diagnostic value in full detail
        if importance == IMPORTANCE_LOW:
            parts.append(
                f"--- Iteration {entry.iteration} ({entry.timestamp:%H:%M:%S}) "
                f"--- [routine — no errors, writes, or markers]"
            )
            if entry.response:
                response_oneline = entry.response[:200].replace("\n", " ")
                parts.append(f"[RESPONSE] {response_oneline}...")
            parts.append("")
            continue

        parts.append(f"--- Iteration {entry.iteration} ({entry.timestamp:%H:%M:%S}) ---")

        # Truncation limits based on importance
        max_reasoning = 2000 if importance == IMPORTANCE_HIGH else 800
        max_response = 2000 if importance == IMPORTANCE_HIGH else 500
        max_output = 1500 if importance == IMPORTANCE_HIGH else 500

        # Reasoning: gated — only include for error iterations and the
        # final iteration.  Clean mid-session iterations rarely have
        # diagnostically useful reasoning.
        if entry.reasoning and not skip_reasoning:
            is_last = entry is session.entries[-1]
            has_err = entry.has_tool_errors if session.role == ROLE_DELEGATOR else entry.has_errors
            if has_err or is_last:
                reasoning = _truncate(entry.reasoning, max_reasoning)
                reasoning = _apply_text_compression(reasoning, project_root)
                parts.append(f"[REASONING] {reasoning}")

        # Response: always included (contains actions taken)
        if entry.response:
            response = _truncate(entry.response, max_response)
            response = _apply_text_compression(response, project_root)
            parts.append(f"[RESPONSE] {response}")

        # Tool calls and outputs (with compression + deduplication)
        for tc in entry.tool_calls:
            args = tc.args[:200]
            if project_root:
                args = _normalize_paths(args, project_root)
            parts.append(f"[TOOL: {tc.tool_name}] args={args}")

            # Compute signature and compressed output for deduplication
            sig = _tool_call_signature(tc)
            compressed = _compress_tool_output(tc)

            if sig in seen_tool_calls and compressed:
                first_iter, first_output = seen_tool_calls[sig]
                if compressed == first_output:
                    exit_match = EXIT_CODE_RE.search(tc.output)
                    exit_str = (f" → exit {exit_match.group(1)}"
                                if exit_match else "")
                    parts.append(
                        f"[OUTPUT] [identical to iter {first_iter}"
                        f"{exit_str} — see above]"
                    )
                    continue

            # First occurrence or different output — record signature
            if sig not in seen_tool_calls:
                seen_tool_calls[sig] = (entry.iteration, compressed)

            # Normal output processing
            output = _truncate(compressed, max_output)
            output = _apply_text_compression(output, project_root)
            output = _deduplicate_lines(output)
            if output:
                parts.append(f"[OUTPUT] {output}")

        parts.append("")

    return "\n".join(parts)


# ── Map Phase ────────────────────────────────────────────────────────────────


def _build_map_prompt(
    session: Session, digest: str, role_context: str,
    prefilter_flags: str = "",
) -> str:
    """Build the map analysis prompt requesting JSON output."""
    # Compute error ratio hint to prevent "everything is fine" bias
    error_hint = ""
    if session.total_iterations > 1:
        err_count = session.error_count
        total = session.total_iterations
        if err_count > 0:
            ratio = err_count / total
            error_hint = (
                f"\n    IMPORTANT: {err_count} out of {total} iterations "
                f"({ratio:.0%}) in this session had errors. "
            )
            if ratio >= 0.5:
                error_hint += (
                    "This is a HIGH error ratio. Even if the final iteration was "
                    "clean, you MUST report the errors that occurred during the "
                    "session. 'Resolved by the end' still means problems happened — "
                    "report each distinct error as a finding."
                )
            elif ratio >= 0.25:
                error_hint += (
                    "Report each distinct error as a finding even if it was "
                    "eventually resolved."
                )

    flags_block = ""
    if prefilter_flags:
        flags_block = f"\n\n    {prefilter_flags}\n"

    return textwrap.dedent(f"""\
    You are analyzing logs from an AI coding agent system. This system uses a pipeline:
    Delegator → Coder → Reviewer → Integration Reviewer.

    {role_context}
    The session log below starts with VERIFIED FACTS — these are extracted
    deterministically from the raw logs and are ground truth.
    {error_hint}
    CRITICAL RULES:
    - Your analysis MUST NOT contradict the verified facts.
    - When citing commands, exit codes, or errors, use ONLY exact strings from the
      verified facts section. Do NOT invent or paraphrase commands.
    - Every finding MUST include an evidence_ref that points to a specific fact sheet
      entry, using the format: "iter=<N>/tool=<name>/exit=<code>".
    - For findings about markers or behavioral issues (no tool call), use the format:
      "iter=<N>/marker=<MARKER_NAME>" or "iter=<N>/tool=write_file".
    - If you cannot link a finding to a specific fact sheet entry, put it in "unknowns".
    - If a pattern or issue is not evidenced in the facts, do not claim it exists.
    - Set supported_by_fact_sheet to true ONLY if the finding directly matches a
      fact sheet entry.
    {flags_block}
    Respond with ONLY valid JSON matching this schema (no markdown, no prose):
    {MAP_JSON_SCHEMA}

    Finding type guide:
    - execution_error: real error in the session's own code/commands
    - review_finding: reviewer found a bug in code under review (expected, not a failure)
    - reviewer_error: reviewer's own command was broken
    - suspicious_validation: command claims success without real testing
    - untested_completion: TASK_COMPLETE with no real tests run
    - infrastructure_bug: sandbox, tool, or stop-sequence issue
    - behavioral_issue: LLM hallucination, loops, ignoring instructions
    - unresolved_error: error still present at final iteration
    - repeated_failure: same command fails repeatedly with no fix
    - write_without_test: file written but never tested

    If nothing went wrong AND no pre-filter flags were raised, return an empty
    findings array with severity "INFO".

    === SESSION LOG ===
    {digest}
    === END LOG ===
    """)


def _call_map_llm(
    client: OpenAI, prompt: str, session: Session,
    stream: bool, show_thinking: bool,
) -> str:
    """Call the LLM for map analysis and return raw text."""
    role = session.role
    if stream:
        print(f"\n  ── Session {session.session_id} [{role}] "
              + "─" * max(0, 50 - len(role) - len(str(session.session_id)))
              + "\n", flush=True)
        return _stream_response(
            client, [{"role": "user", "content": prompt}],
            MAP_MAX_TOKENS, temperature=0.3, show_thinking=show_thinking,
        )
    else:
        print(f"  Analyzing session {session.session_id} [{role}]...",
              end=" ", flush=True)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAP_MAX_TOKENS,
            temperature=0.3,
        )
        result = _extract_content(response)
        print("done.")
        return result


def map_analyze_session(
    client: OpenAI, session: Session, project_root: str = "",
    fact_sheet: SessionFactSheet | None = None,
    prefilter_findings: PrefilterFindings | None = None,
    stream: bool = False, show_thinking: bool = True,
) -> tuple[str, MapResult | None]:
    """Map phase: analyze a single session with the LLM.

    Returns (prose_text, MapResult_or_None). The prose is for display and
    the reducer; the MapResult is for structured verification.
    """
    if fact_sheet is None:
        fact_sheet = build_fact_sheet(session)
    digest = _build_session_digest(
        session, project_root=project_root, fact_sheet=fact_sheet,
    )

    role = session.role
    role_info = ROLE_EXPECTATIONS.get(role, ROLE_EXPECTATIONS[ROLE_UNKNOWN])
    role_context = (
        f"This session's role is: {role}\n"
        f"Role description: {role_info['description']}\n"
        f"Expected iterations: {role_info['normal_iterations']}\n"
        f"Expected markers: {role_info['expected_markers']}\n"
        f"Error interpretation: {role_info['error_guidance']}\n"
    )

    # Build pre-filter behavioral flags for this specific session
    prefilter_flags = ""
    if prefilter_findings is not None:
        prefilter_flags = _format_session_prefilter_flags(
            session.session_id, prefilter_findings,
        )

    prompt = _build_map_prompt(session, digest, role_context, prefilter_flags)
    raw_text = _call_map_llm(client, prompt, session, stream, show_thinking)

    # Budget exhaustion: retry with progressively smaller context
    if _is_budget_error(raw_text):
        print(f"    [budget retry 1] Retrying session {session.session_id} "
              f"without reasoning...")
        digest_lite = _build_session_digest(
            session, project_root=project_root, fact_sheet=fact_sheet,
            skip_reasoning=True,
        )
        prompt = _build_map_prompt(session, digest_lite, role_context, prefilter_flags)
        raw_text = _call_map_llm(client, prompt, session, stream, show_thinking)

    if _is_budget_error(raw_text):
        print(f"    [budget retry 2] Retrying session {session.session_id} "
              f"with fact sheet only...")
        fact_only_digest = format_fact_sheet(fact_sheet)
        prompt = _build_map_prompt(session, fact_only_digest, role_context, prefilter_flags)
        raw_text = _call_map_llm(client, prompt, session, stream, show_thinking)

    if _is_budget_error(raw_text):
        print(f"    [budget exhausted] Session {session.session_id}")
        return f"## Session {session.session_id}\n{raw_text}\n", None

    # Try to parse JSON from the response
    map_result = _parse_map_response(raw_text, session)

    if map_result is not None:
        prose = map_result.to_prose(session.session_id)
        return prose + "\n", map_result

    # JSON parse failed — retry once with explicit instruction
    print(f"    [retry] JSON parse failed for session {session.session_id}, retrying...")
    retry_prompt = (
        prompt
        + "\n\nIMPORTANT: Your previous response was not valid JSON. "
        "Respond with ONLY the JSON object, no markdown fences, no prose before or after."
    )
    raw_text = _call_map_llm(client, retry_prompt, session, stream, show_thinking)
    map_result = _parse_map_response(raw_text, session)

    if map_result is not None:
        prose = map_result.to_prose(session.session_id)
        return prose + "\n", map_result

    # Both attempts failed — fall back to prose (graceful degradation)
    print(f"    [fallback] Using raw prose for session {session.session_id}")
    return f"## Session {session.session_id}\n{raw_text}\n", None


# ── Reduce Phase ─────────────────────────────────────────────────────────────


def _build_reduce_prompt(
    combined: str, question: str | None, context_block: str,
    findings_block: str,
) -> str:
    """Build the reducer prompt."""
    if question:
        focus = f"""The user specifically wants to know: "{question}"
Focus your analysis on answering this question.\n\n"""
    else:
        focus = ""

    if context_block:
        context_instruction = (
            "\n\nYou also have access to the source code that implements the agent system. "
            "Source files include line numbers (e.g. '  42 | def foo():').\n"
            "IMPORTANT RULES for citing source code:\n"
            "- ONLY cite line numbers that you can see in the numbered source code below.\n"
            "- If a file was truncated, do NOT guess line numbers for the missing parts.\n"
            "- If you cannot find the relevant code, say so rather than fabricating a reference.\n"
            "- Reference the function/class name along with the line number.\n"
        )
    else:
        context_instruction = ""

    if findings_block:
        findings_instruction = (
            "\n\nYou have been given AUTOMATED PRE-FILTER FINDINGS below. These are "
            "computed programmatically and are ground truth. Your report MUST:\n"
            "- Use the exact error counts from the findings (do not recount yourself).\n"
            "- Include ALL detected patterns from the findings in your sections.\n"
            "- Not contradict the findings (e.g., do not say a session was clean if findings say it had errors).\n"
            "\nThe session analyses below have been evidence-verified. All findings "
            "have evidence references linking them to deterministic fact sheet entries. "
            "Findings that could not be verified have been removed.\n"
        )
    else:
        findings_instruction = ""

    return textwrap.dedent(f"""\
    You are reviewing analysis reports from multiple AI coding agent sessions.
    {focus}Based on all session analyses below, produce a FINAL REPORT with these sections:

    1. **Verified Facts** — ONLY statements directly supported by the pre-filter findings
       and verified session analyses. Sub-sections:
       - Overall Statistics (from pre-filter — use exact numbers, do not recount)
       - Per-Session Outcomes (role, outcome, error counts from fact sheets)
       - Confirmed Recurring Patterns (only patterns detected by pre-filter)
       - Worst Sessions (ranked by verified error counts)

    2. **Hypotheses / Likely Root Causes** — Interpretations and suspected causes.
       Each hypothesis MUST include:
       - Confidence: HIGH (>80%), MEDIUM (50-80%), or LOW (<50%)
       - Supporting evidence (reference specific sessions and findings)
       - Any counter-evidence
       {context_instruction}

    3. **Recommendations** — Specific, actionable changes to improve the system.
       Link each recommendation to a specific verified fact or hypothesis.
       Only recommend changes feasible within the existing architecture.
       Do not recommend "reward model changes" or "fine-tuning" for a local GGUF model.

    Be concrete and specific. Reference session numbers.
    {findings_instruction}
    {findings_block}

    === SESSION ANALYSES ===
    {combined}
    === END ===

    {context_block}
    """)


def _call_reduce_llm(
    client: OpenAI, prompt: str,
    stream: bool, show_thinking: bool,
) -> str:
    """Call the LLM for reduce synthesis and return raw text."""
    if stream:
        return _stream_response(
            client, [{"role": "user", "content": prompt}],
            REDUCE_MAX_TOKENS, temperature=0.3, show_thinking=show_thinking,
        )
    else:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=REDUCE_MAX_TOKENS,
            temperature=0.3,
        )
        return _extract_content(response)


def reduce_synthesize(
    client: OpenAI,
    session_analyses: list[str],
    question: str | None = None,
    context_block: str = "",
    findings_block: str = "",
    stream: bool = False,
    show_thinking: bool = True,
) -> str:
    """Reduce phase: synthesize all session analyses into a final report."""
    combined = "\n\n".join(session_analyses)

    if stream:
        print("\nSynthesizing cross-session analysis...\n", flush=True)
        print("=" * 70)
        print("  FINAL ANALYSIS REPORT")
        print("=" * 70 + "\n", flush=True)
    else:
        print("\nSynthesizing cross-session analysis...", flush=True)

    prompt = _build_reduce_prompt(combined, question, context_block, findings_block)
    result = _call_reduce_llm(client, prompt, stream, show_thinking)

    # Budget exhaustion retry 1: drop source context
    if _is_budget_error(result) and context_block:
        print("  [budget retry 1] Retrying reduce without source context...")
        prompt = _build_reduce_prompt(combined, question, "", findings_block)
        result = _call_reduce_llm(client, prompt, stream, show_thinking)

    # Budget exhaustion retry 2: truncate session analyses
    if _is_budget_error(result):
        print("  [budget retry 2] Retrying reduce with truncated analyses...")
        short = "\n\n".join(a[:500] + "..." for a in session_analyses)
        prompt = _build_reduce_prompt(short, question, "", findings_block)
        result = _call_reduce_llm(client, prompt, stream, show_thinking)

    return result


# ── Interactive Mode ─────────────────────────────────────────────────────────


def interactive_mode(
    client: OpenAI,
    sessions: list[Session],
    session_analyses: list[str],
    context_block: str = "",
    findings_block: str = "",
    stream: bool = False,
    show_thinking: bool = True,
):
    """Interactive follow-up questions about the logs."""
    combined_analyses = "\n\n".join(session_analyses)

    print("\n" + "="*70)
    print("  INTERACTIVE MODE — Ask questions about the logs (type 'quit' to exit)")
    if context_block:
        print("  Source code context is loaded — you can ask about specific code.")
    print("="*70 + "\n")

    system_content = textwrap.dedent("""\
        You are a log analysis expert for an AI coding agent system.
        You have analyzed all the session logs and have the analysis reports available.
        Answer the user's questions about what went wrong, patterns, root causes, etc.
        Be specific — reference session numbers, iterations, and exact errors when possible.
        If the user asks about a specific session, you can reference the detailed analysis.""")

    if context_block:
        system_content += (
            "\n\nYou also have access to the agent system's source code. "
            "When the user asks about root causes, reference specific files, functions, "
            "and code patterns. For prompt-related issues check prompts.py; for tool "
            "parsing issues check xml_parser.py; for execution issues check agent.py "
            "and orchestrator.py; for sandbox issues check handlers/bash.py."
        )

    user_context = ""
    if findings_block:
        user_context += f"{findings_block}\n\n"
    user_context += f"Here are the analysis reports for all sessions:\n\n{combined_analyses}"
    if context_block:
        user_context += f"\n\n{context_block}"

    context_messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_context},
        {
            "role": "assistant",
            "content": "I've reviewed all session analyses"
                       + (" and the agent source code" if context_block else "")
                       + ". What would you like to know?",
        },
    ]

    while True:
        try:
            question = input("\n[You] > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not question or question.lower() in ("quit", "exit", "q"):
            break

        # If user asks about a specific session, inject its raw digest
        session_match = re.search(r"session\s*(\d+)", question, re.IGNORECASE)
        extra_context = ""
        if session_match:
            sid = int(session_match.group(1))
            matching = [s for s in sessions if s.session_id == sid]
            if matching:
                extra_context = (
                    f"\n\n[Full log digest for session {sid} is attached below]\n"
                    + _build_session_digest(matching[0])
                )

        context_messages.append({
            "role": "user",
            "content": question + extra_context,
        })

        if stream:
            print(flush=True)
            answer = _stream_response(
                client, context_messages,
                INTERACTIVE_MAX_TOKENS, temperature=0.3,
                show_thinking=show_thinking,
            )
        else:
            response = client.chat.completions.create(
                model=MODEL,
                messages=context_messages,
                max_tokens=INTERACTIVE_MAX_TOKENS,
                temperature=0.3,
            )
            answer = _extract_content(response)
            print(f"\n{answer}")
        context_messages.append({"role": "assistant", "content": answer})
