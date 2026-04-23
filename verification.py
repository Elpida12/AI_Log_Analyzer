"""Cross-check LLM analysis claims against deterministic fact sheets."""

from __future__ import annotations

import re
from pathlib import Path

from config import _CMD_PREFIXES, _MARKDOWN_NOISE_RE
from fact_sheet import MapFinding, MapResult, SessionFactSheet


def _normalize_for_comparison(text: str) -> str:
    """Normalize text for fuzzy comparison: lowercase, collapse whitespace."""
    return re.sub(r'\s+', ' ', text.lower().strip())


def _collapse_quote_whitespace(text: str) -> str:
    """Remove whitespace immediately after opening quotes and before closing quotes.

    Multiline python3 -c commands have ``"\\nfrom ...`` which normalizes to
    ``" from ...`` — but LLM citations omit that space (``"from ...``).
    Collapsing it lets both forms match.
    """
    text = re.sub(r'(["\'])\s+', r'\1', text)
    text = re.sub(r'\s+(["\'])', r'\1', text)
    return text


def _command_matches(cited: str, actual: str) -> bool:
    """Check if a cited command matches an actual command (fuzzy)."""
    cited_n = _normalize_for_comparison(cited)
    actual_n = _normalize_for_comparison(actual)
    if cited_n == actual_n:
        return True
    # Also try with whitespace adjacent to quotes collapsed, since multiline
    # python3 -c commands have "\nfrom..." → " from..." after normalization
    # but citations write "from..." (no space after the quote).
    cited_cq = _collapse_quote_whitespace(cited_n)
    actual_cq = _collapse_quote_whitespace(actual_n)
    if cited_cq == actual_cq:
        return True
    # Ellipsis wildcard: `python3 -c "..."` matches any command with that prefix
    # Check both startswith (simple commands) and substring (compound commands
    # like `cd /path && python3 -c "..."` where the cited part is a suffix).
    if '...' in cited_n:
        prefix = cited_n.split('...')[0].strip().rstrip('"\'')
        if prefix and (actual_n.startswith(prefix) or prefix in actual_n):
            return True
        # Retry with quote-whitespace collapsed
        prefix_cq = _collapse_quote_whitespace(prefix)
        if prefix_cq and (actual_cq.startswith(prefix_cq) or prefix_cq in actual_cq):
            return True
    # Substring match (cited might be truncated)
    if len(cited_n) > 10 and (cited_n in actual_n or actual_n in cited_n):
        return True
    # Starts-with match for truncated citations
    if len(cited_n) > 15 and actual_n.startswith(cited_n[:15]):
        return True
    return False


def _looks_like_command(text: str) -> bool:
    """Check if backtick-quoted text looks like a bash command.

    Requires a command prefix followed by optional version digits/dots,
    then a space or end-of-string. Rejects non-command references like
    Docker image names ('python:3.10-slim') where a colon follows.
    """
    text_lower = text.lower()
    for prefix in _CMD_PREFIXES:
        if text_lower.startswith(prefix):
            # Prefixes ending with space (e.g., "# ") already include word boundary
            if prefix.endswith(' '):
                return True
            rest = text_lower[len(prefix):]
            # Skip optional version suffix (e.g., '3' in python3, '3.10' in python3.10)
            i = 0
            while i < len(rest) and (rest[i].isdigit() or rest[i] == '.'):
                i += 1
            # After prefix+version, must be space/end (not colon, dash, etc.)
            if i == len(rest) or rest[i] in (' ', '\t', '\n'):
                return True
    return False


def _verify_evidence_ref(ref: str, fact_sheet: SessionFactSheet) -> bool:
    """Check if an evidence_ref string matches a fact sheet entry.

    Supported ref formats:
      - "iter=<N>/tool=<name>/exit=<code>" — tool call evidence
      - "iter=<N>/marker=<MARKER_NAME>" — marker-based evidence
      - "iter=<N>/tool=write_file" — file write evidence
    """
    if not ref:
        return False
    parts = {}
    for segment in ref.split("/"):
        if "=" in segment:
            k, v = segment.split("=", 1)
            parts[k.strip()] = v.strip()

    ref_iter = parts.get("iter")
    ref_tool = parts.get("tool")
    ref_exit = parts.get("exit")
    ref_marker = parts.get("marker")

    # Marker-based evidence: check against fact_sheet.markers
    if ref_marker is not None:
        marker_name = ref_marker if ref_marker.startswith("[") else f"[{ref_marker}]"
        for m in fact_sheet.markers:
            iter_match = ref_iter is None or str(m["iter"]) == ref_iter
            name_match = m["marker"] == marker_name
            if iter_match and name_match:
                return True
        # Also match if the marker string appears in the outcome
        if ref_marker.replace("[", "").replace("]", "") in fact_sheet.outcome:
            return True
        return False

    # Non-tool-call evidence for reviewer sessions: if the ref names a role
    # instead of a real tool (e.g. "tool=reviewer"), validate that the
    # iteration exists in the session.
    non_tool_names = {"reviewer", "integration_reviewer", "delegator", "coder"}
    if ref_tool and ref_tool in non_tool_names:
        if ref_iter is not None:
            return int(ref_iter) <= fact_sheet.iterations - 1
        return True  # role ref without specific iter — accept

    # Tool call evidence: check against fact_sheet.tool_calls
    for tc in fact_sheet.tool_calls:
        iter_match = ref_iter is None or str(tc["iter"]) == ref_iter
        tool_match = ref_tool is None or tc["tool"] == ref_tool
        exit_match = True
        if ref_exit is not None and ref_exit != "null":
            tc_exit = tc.get("exit_code")
            exit_match = tc_exit is not None and str(tc_exit) == ref_exit
        if iter_match and tool_match and exit_match:
            return True

    # Fallback: if the session has no tool calls (e.g. reviewer with prose only)
    # and the ref just points at an iteration, validate the iteration exists
    if not fact_sheet.tool_calls and ref_iter is not None:
        return int(ref_iter) <= fact_sheet.iterations - 1

    return False


def verify_map_result(
    result: MapResult, fact_sheet: SessionFactSheet,
) -> tuple[MapResult, dict[str, int], list[str]]:
    """Verify a structured MapResult against the fact sheet.

    Drops contradicted findings (hard gate). Returns the filtered MapResult,
    verification stats, and annotation strings.
    """
    stats: dict[str, int] = {"verified": 0, "contradicted": 0, "unverifiable": 0}
    annotations: list[str] = []
    verified_findings: list[MapFinding] = []

    for f in result.findings:
        issues: list[str] = []

        # Check 1: evidence_ref must match a fact sheet entry
        if f.evidence_ref:
            if _verify_evidence_ref(f.evidence_ref, fact_sheet):
                stats["verified"] += 1
            else:
                issues.append(f"evidence_ref '{f.evidence_ref}' not found in fact sheet")
        else:
            issues.append("missing evidence_ref")

        # Check 2: iteration must be within session range
        if f.iteration < 0 or f.iteration > max(tc["iter"] for tc in fact_sheet.tool_calls) if fact_sheet.tool_calls else False:
            issues.append(f"iteration {f.iteration} out of range")

        # Check 3: command_exact should appear in fact sheet commands
        if f.command_exact:
            cmd_found = any(
                _command_matches(f.command_exact, cmd)
                for cmd in fact_sheet.all_commands
            )
            if not cmd_found:
                # Also check raw args
                cmd_found = any(
                    _normalize_for_comparison(f.command_exact) in
                    _normalize_for_comparison(tc["args_raw"])
                    for tc in fact_sheet.tool_calls
                )
            if cmd_found:
                stats["verified"] += 1
            else:
                # Reviewer sessions may reference commands from the reviewed session
                if fact_sheet.role in ("reviewer", "integration_reviewer"):
                    stats["unverifiable"] += 1
                else:
                    issues.append(f"command not found: `{f.command_exact[:80]}`")

        # Check 4: exit_code should exist in fact sheet
        if f.exit_code is not None:
            actual_codes = fact_sheet.all_exit_codes | {0}
            if f.exit_code in actual_codes:
                stats["verified"] += 1
            else:
                issues.append(f"exit code {f.exit_code} not in session "
                              f"(actual: {sorted(actual_codes)})")

        # Check 5: claimed unresolved error vs fact sheet final_iteration_clean
        if f.type == "unresolved_error" and fact_sheet.final_iteration_clean:
            issues.append("claimed unresolved error but final iteration is clean")

        # Check 6: write_without_test — verify the file was actually written
        if f.type == "write_without_test" and f.command_exact:
            file_found = any(
                f.command_exact in fw["path"] or Path(f.command_exact).name in fw["path"]
                for fw in fact_sheet.files_written
            )
            if file_found:
                stats["verified"] += 1
            else:
                issues.append(f"claimed untested write for '{f.command_exact}' "
                              f"but file not in fact sheet writes")

        # Check 7: repeated_failure — verify the command actually failed multiple times
        if f.type == "repeated_failure" and f.command_exact:
            fail_count = sum(
                1 for tc in fact_sheet.tool_calls
                if tc["has_error"]
                and f.command_exact in (tc.get("command", "") or tc.get("args_raw", ""))
            )
            if fail_count >= 2:
                stats["verified"] += 1
            elif fail_count == 1:
                issues.append(f"claimed repeated failure but command only failed once")
            # fail_count == 0 already caught by command check

        # Check 8: "never tested" — verify no bash command references the file
        if f.type == "untested_completion":
            has_bash = any(tc["tool"] == "bash" for tc in fact_sheet.tool_calls)
            if has_bash:
                issues.append("claimed untested completion but bash commands exist")

        # Decision: keep or drop
        if issues:
            stats["contradicted"] += len(issues)
            for issue in issues:
                annotations.append(
                    f"[DROPPED] Session {fact_sheet.session_id}, "
                    f"{f.type} iter {f.iteration}: {issue}"
                )
            # Move to unknowns instead of silently dropping
            result.unknowns.append(
                f"[dropped finding: {f.type} iter {f.iteration}] "
                + "; ".join(issues)
            )
        else:
            verified_findings.append(f)

    # Check severity consistency
    if result.severity == "CRITICAL" and not verified_findings:
        annotations.append(
            f"[DOWNGRADED] Session {fact_sheet.session_id}: "
            f"severity was CRITICAL but all findings were dropped"
        )
        result.severity = "INFO"

    result.findings = verified_findings
    return result, stats, annotations


def verify_session_analysis(
    analysis: str, fact_sheet: SessionFactSheet,
) -> tuple[str, dict[str, int]]:
    """Cross-check an LLM session analysis against its fact sheet.

    Scans the analysis text for factual claims (commands, exit codes,
    error messages, patterns) and verifies each against deterministic data.

    Returns:
        (annotated_analysis, {"verified": N, "contradicted": N, "unverifiable": N},
         annotations_list)
    """
    stats: dict[str, int] = {"verified": 0, "contradicted": 0, "unverifiable": 0}
    annotations: list[str] = []

    # === Check 1: Cited bash commands ===
    # Detect negation context: command cited as NOT having been run
    _negation_re = re.compile(
        r'(?:did(?:n.t| not)|never|without|failed to|should have|'
        r'skipped|omitted|no .{0,20})\s*'
        r'(?:run|ran|execut|running|executing|test|call|invoke)',
        re.IGNORECASE,
    )
    for match in re.finditer(r'`([^`]{8,150})`', analysis):
        cited_stripped = match.group(1).strip()
        if not _looks_like_command(cited_stripped):
            continue

        # Check if the citation is in a negative context (e.g., "did not run `cmd`")
        preceding_start = max(0, match.start() - 100)
        preceding_text = analysis[preceding_start:match.start()]
        if _negation_re.search(preceding_text):
            # Command cited as not having been run — skip verification
            continue

        found = any(
            _command_matches(cited_stripped, cmd)
            for cmd in fact_sheet.all_commands
        )
        # Also check raw args as fallback
        if not found:
            found = any(
                _normalize_for_comparison(cited_stripped) in
                _normalize_for_comparison(tc["args_raw"])
                for tc in fact_sheet.tool_calls
            )
        if found:
            stats["verified"] += 1
        else:
            # Reviewer/integration sessions naturally reference commands from
            # the session under review — don't mark as contradicted.
            is_reviewer = fact_sheet.role in ("reviewer", "integration_reviewer")
            if is_reviewer:
                annotations.append(
                    f"[UNVERIFIED] Cited command may be from reviewed session "
                    f"{fact_sheet.session_id}: `{cited_stripped[:80]}`"
                )
                stats["unverifiable"] += 1
            else:
                annotations.append(
                    f"[CONTRADICTED] Cited command not found in session "
                    f"{fact_sheet.session_id}: `{cited_stripped[:80]}`"
                )
                stats["contradicted"] += 1

    # === Check 2: Exit code claims ===
    exit_claims = re.findall(
        r'exit\s+(?:code[:\s]+)?(\d+)', analysis, re.IGNORECASE
    )
    actual_codes = fact_sheet.all_exit_codes | {0}
    for code_str in exit_claims:
        code = int(code_str)
        if code in actual_codes:
            stats["verified"] += 1
        else:
            annotations.append(
                f"[CONTRADICTED] Claimed exit code {code} not found in session "
                f"{fact_sheet.session_id} (actual: {sorted(actual_codes)})"
            )
            stats["contradicted"] += 1

    # === Check 3: Pattern claims ===
    # "command=" prefix claim
    if re.search(r'command[=].*prefix', analysis, re.IGNORECASE):
        has_prefix = any(
            "'command'" in tc["args_raw"] and
            re.search(r"'command':\s*'command[: >]", tc["args_raw"])
            for tc in fact_sheet.tool_calls
        )
        if has_prefix:
            stats["verified"] += 1
        else:
            annotations.append(
                f"[CONTRADICTED] Claimed 'command=' prefix pattern in session "
                f"{fact_sheet.session_id} but no tool call args contain this"
            )
            stats["contradicted"] += 1

    # === Check 4: Error message citations ===
    # Only match single-line backtick-delimited text (inline code spans) to
    # avoid extracting narrative prose / markdown headings that happen to
    # contain error keywords like "Syntax error persisted".
    error_cites = re.findall(
        r'`((?:[^`\n]*(?:Error|Exception|Traceback|failed|errno|SyntaxError)[^`\n]*))`',
        analysis, re.IGNORECASE,
    )
    # Markdown formatting inside a citation means it's prose, not an error msg
    for err_cite in error_cites:
        if len(err_cite) < 8 or len(err_cite) > 200:
            continue
        if _MARKDOWN_NOISE_RE.search(err_cite):
            continue
        cite_words = {w.lower() for w in err_cite.split() if len(w) > 3}
        if not cite_words:
            continue
        actual_lower = fact_sheet.all_error_text.lower()
        # Also check full tool outputs
        all_outputs = " ".join(tc.get("error_snippet", "") for tc in fact_sheet.tool_calls)
        combined_actual = (actual_lower + " " + all_outputs).lower()
        match_count = sum(1 for w in cite_words if w in combined_actual)
        if match_count >= max(1, len(cite_words) * 0.4):
            stats["verified"] += 1
        else:
            annotations.append(
                f"[UNVERIFIED] Cited error text not found in session "
                f"{fact_sheet.session_id}: '{err_cite[:80]}'"
            )
            stats["unverifiable"] += 1

    # === Build annotated output ===
    summary = (
        f"[VERIFICATION: {stats['verified']} verified, "
        f"{stats['contradicted']} contradicted, "
        f"{stats['unverifiable']} unverifiable]"
    )
    if annotations:
        annotation_block = "\n".join(annotations)
        annotated = f"{analysis}\n\n{summary}\n{annotation_block}"
    else:
        annotated = f"{analysis}\n\n{summary}"

    return annotated, stats, annotations


def verify_final_report(
    report: str, fact_sheets: dict[int, SessionFactSheet],
    findings=None,
) -> str:
    """Cross-check the final synthesized report against all fact sheets.

    Looks for cross-session claims (e.g., "Session X had Y") and verifies
    them against the appropriate session's fact sheet and pre-filter findings.

    Returns annotation text to append to the report (empty if all clean).
    """
    annotations: list[str] = []

    # Find "Session N" references paired with factual assertions
    # Pattern: "Session <N>" near an exit code, command, or pattern claim
    for match in re.finditer(
        r'[Ss]ession\s+(\d+)[^.]*?(?:exit\s+(?:code\s+)?(\d+)|`([^`]{8,120})`)',
        report,
    ):
        sid = int(match.group(1))
        if sid not in fact_sheets:
            continue
        fs = fact_sheets[sid]

        # Check exit code claim for this session
        if match.group(2):
            code = int(match.group(2))
            actual_codes = fs.all_exit_codes | {0}
            if code not in actual_codes:
                annotations.append(
                    f"[CONTRADICTED] Session {sid}: claimed exit code {code}, "
                    f"actual codes: {sorted(actual_codes)}"
                )

        # Check command claim for this session
        if match.group(3):
            cited_cmd = match.group(3)
            if _looks_like_command(cited_cmd):
                # Check if the citation is in a negative context
                preceding_start = max(0, match.start() - 100)
                preceding_text = report[preceding_start:match.start()]
                _neg_re = re.compile(
                    r'(?:did(?:n.t| not)|never|without|failed to|should have|'
                    r'skipped|omitted|no .{0,20}|[Cc]laimed to)\s*'
                    r'(?:run|ran|execut|running|executing|test|call|invoke)',
                    re.IGNORECASE,
                )
                if _neg_re.search(preceding_text):
                    continue

                found = any(
                    _command_matches(cited_cmd, cmd)
                    for cmd in fs.all_commands
                )
                if not found:
                    # Also check raw args
                    found = any(
                        _normalize_for_comparison(cited_cmd) in
                        _normalize_for_comparison(tc["args_raw"])
                        for tc in fs.tool_calls
                    )
                if not found:
                    annotations.append(
                        f"[CONTRADICTED] Session {sid}: cited command not found: "
                        f"`{cited_cmd[:80]}`"
                    )

    # Cross-session claim verification (requires pre-filter findings)
    if findings is not None:
        # Check "N sessions had errors" type claims
        for match in re.finditer(
            r'(\d+)\s+(?:out of \d+ )?sessions?\s+(?:had|with|showed|exhibited)\s+'
            r'(?:execution\s+)?errors?',
            report, re.IGNORECASE,
        ):
            claimed = int(match.group(1))
            actual = len(findings.execution_error_ids)
            if claimed != actual:
                annotations.append(
                    f"[CONTRADICTED] Claimed {claimed} sessions with errors, "
                    f"actual: {actual}"
                )

        # Check claimed session outcome (e.g., "Session 3 passed review")
        for match in re.finditer(
            r'[Ss]ession\s+(\d+)\s+.*?(?:passed|PASS|pass)',
            report,
        ):
            sid = int(match.group(1))
            if sid in fact_sheets:
                outcome = fact_sheets[sid].outcome
                if "FAIL" in outcome or "BLOCKED" in outcome:
                    annotations.append(
                        f"[CONTRADICTED] Session {sid}: claimed pass but "
                        f"actual outcome: {outcome}"
                    )

        # Check claimed role for a session
        for match in re.finditer(
            r'[Ss]ession\s+(\d+)\s*[\[(]?\s*(delegator|coder|reviewer|'
            r'integration.reviewer)',
            report, re.IGNORECASE,
        ):
            sid = int(match.group(1))
            claimed_role = match.group(2).lower().replace(" ", "_")
            if sid in fact_sheets:
                actual_role = fact_sheets[sid].role
                if claimed_role != actual_role:
                    annotations.append(
                        f"[CONTRADICTED] Session {sid}: claimed role "
                        f"'{claimed_role}', actual: '{actual_role}'"
                    )

        # Check "N files modified" claims per session
        for match in re.finditer(
            r'[Ss]ession\s+(\d+)[^.]*?(\d+)\s+files?\s+'
            r'(?:modified|written|created)',
            report, re.IGNORECASE,
        ):
            sid = int(match.group(1))
            claimed_count = int(match.group(2))
            if sid in fact_sheets:
                actual_count = len(fact_sheets[sid].files_written)
                if claimed_count != actual_count:
                    annotations.append(
                        f"[CONTRADICTED] Session {sid}: claimed {claimed_count} "
                        f"files modified, actual: {actual_count}"
                    )

    if not annotations:
        return ""

    lines = [
        "",
        "─" * 70,
        "  CLAIM VERIFICATION (final report)",
        "─" * 70,
    ]
    for a in annotations:
        lines.append(f"  {a}")
    lines.append("")
    return "\n".join(lines)
