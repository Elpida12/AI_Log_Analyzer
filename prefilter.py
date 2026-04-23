"""Deterministic pre-filter detection heuristics."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import config
from config import (
    _BOGUS_SUCCESS_RE,
    _EXCEPTION_LINE_RE,
    ERROR_PATTERNS,
    EXIT_CODE_RE,
    ROLE_CODER,
    ROLE_DELEGATOR,
    ROLE_INTEGRATION,
    ROLE_REVIEWER,
    ROLE_UNKNOWN,
    ROLE_EXPECTATIONS,
)
from parsing import (
    Session,
    _classify_tool_failure,
    _extract_bash_command,
    _extract_write_file_content,
    _extract_write_file_path,
)


# ── Shared Check Helpers ──────────────────────────────────────────────────────


def _check_bogus_validation(cmd: str) -> str:
    """Check if a bash command looks like a bogus validation.

    Returns a short reason string if suspicious, empty string if OK.
    """
    # Pattern 1: python3 -c with only imports and hardcoded success prints
    if re.search(r"python3?\s+-c\s+", cmd):
        # Check if a print() call contains a success-claiming keyword
        print_match = re.search(r"print\s*\(['\"]([^'\"]*)['\"]", cmd)
        if print_match and _BOGUS_SUCCESS_RE.search(print_match.group(1)):
            # Extract inline code and strip imports + prints
            code_match = re.search(r"python3?\s+-c\s+['\"](.+)", cmd, re.DOTALL)
            if code_match:
                code = code_match.group(1).rstrip("'\"")
                stripped = code
                stripped = re.sub(r"(?:from\s+\S+\s+)?import\s+[^;\"'\n]+", "", stripped)
                stripped = re.sub(r"print\s*\([^)]*\)", "", stripped)
                stripped = re.sub(r"[;\s]", "", stripped)
                if not stripped:
                    return "Hardcoded success — only imports and prints, no real test"

    # Pattern 2: exit code masking via trailing echo (cmd; echo "success")
    masking_match = re.search(r";\s*echo\s+['\"]([^'\"]*)['\"]", cmd)
    if masking_match and _BOGUS_SUCCESS_RE.search(masking_match.group(1)):
        return "Exit code masking — echo with success message after semicolon"

    # Pattern 3: file existence check claimed as content validation
    if re.search(r"(?:ls\b|test\s+-[fde])", cmd):
        echo_match = re.search(r"&&\s*echo\s+['\"]([^'\"]*)['\"]", cmd)
        if echo_match:
            msg = echo_match.group(1)
            if _BOGUS_SUCCESS_RE.search(msg) and re.search(
                r"(?i)\b(content|correct|data|output|valid)\b", msg
            ):
                return "File existence check claimed as content validation"

    # Pattern 4: py_compile treated as functional validation
    if "py_compile" in cmd:
        echo_match = re.search(r"&&\s*echo\s+['\"]([^'\"]*)['\"]", cmd)
        if echo_match and re.search(
            r"(?i)\b(correct|works|functional|valid|good|ok)\b",
            echo_match.group(1),
        ):
            return "Syntax check (py_compile) claimed as functional validation"

    return ""


def _check_stop_seq_truncation(content: str) -> str:
    """Detect if write_file content has an unbalanced quote on its last line.

    The stop sequence ``\\n</tool>`` can consume a trailing closing quote
    when the quote and newline share a BPE token (a known issue already
    handled by bash.py's ``_fix_unbalanced_quotes``).  For write_file,
    this silently corrupts file content — the LLM generates the correct
    closing quote, but it gets stripped before the tool sees it.

    Gated behind ``config.DETECT_AI_CODER_BUGS`` — this pattern is specific
    to the AI_Coder agent system.

    Returns a short reason string if suspicious, empty string if OK.
    """
    if not config.DETECT_AI_CODER_BUGS:
        return ""
    if not content:
        return ""
    last_line = content.rstrip('\n').rsplit('\n', 1)[-1]
    # Count unescaped double and single quotes on the last line
    for q in ('"', "'"):
        count = last_line.count(q) - last_line.count(f'\\{q}')
        if count % 2 != 0:
            return (
                f"Unbalanced {q} on last line of write_file content — "
                f"likely stop-sequence truncation (see bash.py "
                f"_fix_unbalanced_quotes for the known issue)"
            )
    return ""


# ── Detectors ─────────────────────────────────────────────────────────────────


def _detect_suspicious_validations(
    sessions: list[Session],
) -> dict[int, list[dict]]:
    """Detect bash commands that claim to validate something but don't.

    Returns session_id -> list of {iter, command, reason} for affected sessions.
    """
    results: dict[int, list[dict]] = {}

    for session in sessions:
        suspicious: list[dict] = []
        for entry in session.entries:
            for tc in entry.tool_calls:
                if tc.tool_name != "bash":
                    continue
                # Only inspect commands that "succeeded"
                exit_match = EXIT_CODE_RE.search(tc.output)
                if not exit_match or int(exit_match.group(1)) != 0:
                    continue
                cmd = _extract_bash_command(tc.args)
                if not cmd:
                    continue
                reason = _check_bogus_validation(cmd)
                if reason:
                    suspicious.append({
                        "iter": entry.iteration,
                        "command": cmd[:200],
                        "reason": reason,
                    })
        if suspicious:
            results[session.session_id] = suspicious

    return results


def _detect_unresolved_exits(
    sessions: list[Session],
) -> dict[int, list[dict]]:
    """Detect sessions where the final iteration still has non-zero exit codes.

    Returns session_id -> list of {iter, command, exit_code, error_snippet}.
    """
    results: dict[int, list[dict]] = {}
    for session in sessions:
        if not session.entries:
            continue
        final = session.entries[-1]
        unresolved: list[dict] = []
        for tc in final.tool_calls:
            exit_match = EXIT_CODE_RE.search(tc.output)
            if exit_match and int(exit_match.group(1)) != 0:
                cmd = _extract_bash_command(tc.args) if tc.tool_name == "bash" else ""
                snippet = ""
                for line in tc.output.splitlines():
                    if _EXCEPTION_LINE_RE.match(line):
                        snippet = line.strip()[:200]
                        break
                if not snippet:
                    for line in tc.output.splitlines():
                        if any(p.search(line) for p in ERROR_PATTERNS):
                            snippet = line.strip()[:200]
                            break
                unresolved.append({
                    "iter": final.iteration,
                    "command": cmd[:200],
                    "exit_code": int(exit_match.group(1)),
                    "error_snippet": snippet,
                })
        if unresolved:
            results[session.session_id] = unresolved
    return results


def _detect_write_without_test(
    sessions: list[Session],
) -> dict[int, list[dict]]:
    """Detect write_file calls not followed by any bash command referencing the file.

    Returns session_id -> list of {iter_written, path, tested}.
    """
    results: dict[int, list[dict]] = {}
    for session in sessions:
        writes: list[tuple[int, str]] = []  # (iteration, filename)
        all_cmds_after: dict[str, list[str]] = {}  # filename -> [cmds after write]

        for entry in session.entries:
            for tc in entry.tool_calls:
                if tc.tool_name == "write_file":
                    path = _extract_write_file_path(tc.args)
                    if path and path != "?":
                        fname = Path(path).name
                        writes.append((entry.iteration, path))
                        all_cmds_after[fname] = []
                elif tc.tool_name == "bash":
                    cmd = _extract_bash_command(tc.args)
                    if cmd:
                        for fname in all_cmds_after:
                            all_cmds_after[fname].append(cmd)

        untested = []
        for it, path in writes:
            fname = Path(path).name
            cmds = all_cmds_after.get(fname, [])
            tested = any(fname in c for c in cmds)
            if not tested:
                untested.append({
                    "iter_written": it,
                    "path": path,
                    "tested": False,
                })
        if untested:
            results[session.session_id] = untested
    return results


def _detect_repeated_failing_command(
    sessions: list[Session],
) -> dict[int, list[dict]]:
    """Detect identical commands failing repeatedly with no code change between them.

    Returns session_id -> list of {command, exit_code, iterations, count}.
    """
    results: dict[int, list[dict]] = {}
    for session in sessions:
        # Track (command, exit_code) -> list of iterations
        fail_sequences: dict[tuple[str, int], list[int]] = {}
        # Track iterations with write_file calls (code changes)
        write_iters: set[int] = set()

        for entry in session.entries:
            for tc in entry.tool_calls:
                if tc.tool_name == "write_file":
                    write_iters.add(entry.iteration)
                elif tc.tool_name == "bash":
                    exit_match = EXIT_CODE_RE.search(tc.output)
                    if exit_match and int(exit_match.group(1)) != 0:
                        cmd = _extract_bash_command(tc.args)
                        if cmd:
                            key = (cmd.strip(), int(exit_match.group(1)))
                            if key not in fail_sequences:
                                fail_sequences[key] = []
                            fail_sequences[key].append(entry.iteration)

        flagged = []
        for (cmd, code), iters in fail_sequences.items():
            if len(iters) < 2:
                continue
            # Check for consecutive failures with no write_file in between
            consecutive_runs: list[list[int]] = [[iters[0]]]
            for i in range(1, len(iters)):
                # Check if any write_file happened between iters[i-1] and iters[i]
                intervening_writes = any(
                    iters[i - 1] < w < iters[i] for w in write_iters
                )
                if intervening_writes:
                    consecutive_runs.append([iters[i]])
                else:
                    consecutive_runs[-1].append(iters[i])
            for run in consecutive_runs:
                if len(run) >= 2:
                    flagged.append({
                        "command": cmd[:200],
                        "exit_code": code,
                        "iterations": run,
                        "count": len(run),
                    })
        if flagged:
            results[session.session_id] = flagged
    return results


def _detect_marker_outcome_mismatches(
    sessions: list[Session],
) -> dict[int, list[str]]:
    """Detect sessions with markers that contradict their role or each other.

    Returns session_id -> list of reason strings.
    """
    # Which markers are valid for each role
    valid_markers: dict[str, set[str]] = {
        ROLE_CODER: {"[TASK_COMPLETE]", "[OBSERVATIONS]"},
        ROLE_REVIEWER: {"[REVIEW_PASS]", "[REVIEW_FAIL]", "[REVIEW_BLOCKED]"},
        ROLE_INTEGRATION: {
            "[INTEGRATION_PASS]", "[INTEGRATION_FAIL]", "[INTEGRATION_BLOCKED]",
        },
        ROLE_DELEGATOR: set(),  # delegators should not emit markers
    }

    # Contradictory marker pairs
    contradictions = [
        ("[REVIEW_PASS]", "[REVIEW_FAIL]"),
        ("[INTEGRATION_PASS]", "[INTEGRATION_FAIL]"),
    ]

    results: dict[int, list[str]] = {}
    for session in sessions:
        issues: list[str] = []
        markers = set(session.all_markers)
        role = session.role

        # Check for wrong-role markers
        allowed = valid_markers.get(role)
        if allowed is not None:
            wrong = markers - allowed
            # Exclude markers used for role detection itself
            if wrong:
                issues.append(
                    f"Role '{role}' emitted unexpected markers: {sorted(wrong)}"
                )

        # Check for contradictory markers
        for a, b in contradictions:
            if a in markers and b in markers:
                issues.append(f"Contradictory markers in same session: {a} and {b}")

        if issues:
            results[session.session_id] = issues
    return results


def _detect_command_prefix_pattern(sessions: list[Session]) -> dict[int, dict[str, int]]:
    """Detect the 'command' prefix hallucination in bash tool calls.

    Gated behind ``config.DETECT_AI_CODER_BUGS`` — this hallucination pattern
    is specific to the AI_Coder agent system.

    Returns a dict of session_id -> {variant: count} for affected sessions.
    """
    if not config.DETECT_AI_CODER_BUGS:
        return {}
    prefix_re = re.compile(
        r"^'command':\s*'(command[: >]|command(?=[a-zA-Z]))"
    )
    results: dict[int, dict[str, int]] = {}

    for s in sessions:
        variants: dict[str, int] = {}
        for e in s.entries:
            for tc in e.tool_calls:
                # Check the Args field for command prefix in the actual command value
                args_str = tc.args
                # Match patterns like: 'command': 'command python3 ...'
                # or 'command': 'command: python3 ...'
                m = re.search(
                    r"'command':\s*'(command[: >]|command(?=[a-zA-Z]))",
                    args_str,
                )
                if m:
                    variant = m.group(1).rstrip()
                    variants[variant] = variants.get(variant, 0) + 1
        if variants:
            results[s.session_id] = variants

    return results


# ── Prefilter Findings ────────────────────────────────────────────────────────


@dataclass
class PrefilterFindings:
    """Structured output from the pre-filter pass."""
    total_sessions: int
    total_iterations: int
    problem_session_ids: list[int]
    clean_session_ids: list[int]
    per_session: list[dict]           # [{id, errors, failed_tools, outcome, markers, error_class}]
    command_prefix: dict[int, dict[str, int]]  # session_id -> {variant: count}
    total_command_prefix_count: int
    execution_error_ids: list[int]    # Coder/delegator sessions with real errors
    review_finding_ids: list[int]     # Reviewer sessions — errors are findings from testing
    reviewer_error_ids: list[int]     # Reviewer sessions where the reviewer itself broke
    suspicious_validation_ids: list[int] = field(default_factory=list)
    suspicious_validations: dict = field(default_factory=dict)  # session_id -> [{iter, command, reason}]
    untested_completion_ids: list[int] = field(default_factory=list)  # coder sessions with TASK_COMPLETE but no bash calls
    stop_seq_truncation_ids: list[int] = field(default_factory=list)  # sessions with write_file truncation
    stop_seq_truncations: dict = field(default_factory=dict)  # session_id -> [{iter, path, reason}]
    unresolved_exit_ids: list[int] = field(default_factory=list)
    unresolved_exits: dict = field(default_factory=dict)  # session_id -> [{iter, command, exit_code, error_snippet}]
    write_without_test_ids: list[int] = field(default_factory=list)
    write_without_tests: dict = field(default_factory=dict)  # session_id -> [{iter_written, path, tested}]
    repeated_fail_ids: list[int] = field(default_factory=list)
    repeated_fails: dict = field(default_factory=dict)  # session_id -> [{command, exit_code, iterations, count}]
    marker_mismatch_ids: list[int] = field(default_factory=list)
    marker_mismatches: dict = field(default_factory=dict)  # session_id -> [reason_strings]


def compute_prefilter_findings(sessions: list[Session]) -> PrefilterFindings:
    """Run all pre-filter checks and return structured findings."""
    problem_ids = []
    clean_ids = []
    execution_error_ids = []
    review_finding_ids = []
    reviewer_error_ids = []
    per_session = []

    for s in sessions:
        role = s.role
        # Delegator responses are JSON task plans — only check tool outputs
        # to avoid false positives from natural language like "handle errors".
        if role == ROLE_DELEGATOR:
            error_count = sum(1 for e in s.entries if e.has_tool_errors)
        else:
            error_count = sum(1 for e in s.entries if e.has_errors)
        failed_tools = []
        for e in s.entries:
            for tc in e.tool_calls:
                match = EXIT_CODE_RE.search(tc.output)
                if match and int(match.group(1)) != 0:
                    classification = _classify_tool_failure(
                        tc.tool_name, tc.args, int(match.group(1)),
                        tc.output, role,
                    )
                    failed_tools.append((tc.tool_name, tc.args[:80],
                                         int(match.group(1)), classification))

        has_problems = error_count > 0 or len(failed_tools) > 0

        # Classify session errors by role
        if not has_problems:
            error_class = "clean"
            clean_ids.append(s.session_id)
        elif role in (ROLE_REVIEWER, ROLE_INTEGRATION):
            # Check if any tool failures are the reviewer's own mistakes
            has_reviewer_mistakes = any(
                cls == "reviewer_error" for _, _, _, cls in failed_tools
            )
            # Reviewer sessions naturally discuss errors in the code they review,
            # so error-pattern matches in response text are expected, not failures.
            if has_reviewer_mistakes:
                error_class = "reviewer_error"
                reviewer_error_ids.append(s.session_id)
            else:
                error_class = "review_finding"
                review_finding_ids.append(s.session_id)
            problem_ids.append(s.session_id)
        else:
            error_class = "execution_error"
            execution_error_ids.append(s.session_id)
            problem_ids.append(s.session_id)

        per_session.append({
            "id": s.session_id,
            "role": role,
            "errors": error_count,
            "failed_tools": failed_tools,
            "outcome": s.outcome,
            "markers": s.all_markers,
            "iterations": s.total_iterations,
            "error_class": error_class,
        })

    cmd_prefix = _detect_command_prefix_pattern(sessions)
    total_prefix = sum(sum(v.values()) for v in cmd_prefix.values())

    # Detect bogus validations (commands that claim success without testing)
    suspicious = _detect_suspicious_validations(sessions)
    suspicious_ids = sorted(suspicious.keys())

    # Detect coder sessions that declared TASK_COMPLETE without running any bash commands
    untested_ids = []
    for s in sessions:
        if s.role not in (ROLE_CODER,):
            continue
        if "TASK_COMPLETE" not in s.outcome:
            continue
        has_bash = any(
            tc.tool_name == "bash"
            for e in s.entries
            for tc in e.tool_calls
        )
        if not has_bash:
            untested_ids.append(s.session_id)

    # Detect stop-sequence truncation in write_file calls
    trunc_map: dict[int, list[dict]] = {}
    for s in sessions:
        hits = []
        for entry in s.entries:
            for tc in entry.tool_calls:
                if tc.tool_name != "write_file":
                    continue
                file_content = _extract_write_file_content(tc.args)
                if not file_content:
                    continue
                reason = _check_stop_seq_truncation(file_content)
                if reason:
                    hits.append({
                        "iter": entry.iteration,
                        "path": _extract_write_file_path(tc.args),
                        "reason": reason,
                    })
        if hits:
            trunc_map[s.session_id] = hits
    trunc_ids = sorted(trunc_map.keys())

    # Detect unresolved non-zero exits at final iteration
    unresolved = _detect_unresolved_exits(sessions)
    unresolved_ids = sorted(unresolved.keys())

    # Detect write_file calls with no subsequent test referencing the file
    write_no_test = _detect_write_without_test(sessions)
    write_no_test_ids = sorted(write_no_test.keys())

    # Detect repeated identical failing commands with no code change between them
    repeated = _detect_repeated_failing_command(sessions)
    repeated_ids = sorted(repeated.keys())

    # Detect marker/role mismatches
    marker_mm = _detect_marker_outcome_mismatches(sessions)
    marker_mm_ids = sorted(marker_mm.keys())

    # Sessions with suspicious validations, untested completions, or new
    # deterministic detections should not be auto-skipped
    for sid in (suspicious_ids + untested_ids + unresolved_ids
                + write_no_test_ids + repeated_ids + marker_mm_ids):
        if sid in clean_ids:
            clean_ids.remove(sid)

    return PrefilterFindings(
        total_sessions=len(sessions),
        total_iterations=sum(s.total_iterations for s in sessions),
        problem_session_ids=problem_ids,
        clean_session_ids=clean_ids,
        per_session=per_session,
        command_prefix=cmd_prefix,
        total_command_prefix_count=total_prefix,
        execution_error_ids=execution_error_ids,
        review_finding_ids=review_finding_ids,
        reviewer_error_ids=reviewer_error_ids,
        suspicious_validation_ids=suspicious_ids,
        suspicious_validations=suspicious,
        untested_completion_ids=untested_ids,
        stop_seq_truncation_ids=trunc_ids,
        stop_seq_truncations=trunc_map,
        unresolved_exit_ids=unresolved_ids,
        unresolved_exits=unresolved,
        write_without_test_ids=write_no_test_ids,
        write_without_tests=write_no_test,
        repeated_fail_ids=repeated_ids,
        repeated_fails=repeated,
        marker_mismatch_ids=marker_mm_ids,
        marker_mismatches=marker_mm,
    )


def prefilter_report(sessions: list[Session], findings: PrefilterFindings) -> str:
    """Generate a quick text report of issues found without using the LLM."""
    lines = []
    lines.append(f"{'='*70}")
    lines.append(f"  PRE-FILTER REPORT — {findings.total_sessions} sessions, "
                 f"{findings.total_iterations} total iterations")
    lines.append(f"{'='*70}\n")

    for ps in findings.per_session:
        s_id = ps["id"]
        sess = [s for s in sessions if s.session_id == s_id][0]
        role_tag = f"  [{ps['role']}]" if ps["role"] != ROLE_UNKNOWN else ""
        lines.append(f"Session {s_id}  [{sess.start_time:%H:%M:%S} → {sess.end_time:%H:%M:%S}]  "
                     f"{ps['iterations']} iters  |  outcome: {ps['outcome']}{role_tag}")

        if ps["errors"]:
            error_class = ps.get("error_class", "execution_error")
            if error_class == "review_finding":
                lines.append(f"  ⚠ {ps['errors']} iterations with errors "
                             f"[review findings — expected for review process]")
            elif error_class == "reviewer_error":
                lines.append(f"  ⚠ {ps['errors']} iterations with errors "
                             f"[reviewer malfunction]")
            else:
                lines.append(f"  ⚠ {ps['errors']} iterations with errors")
        if ps["failed_tools"]:
            for item in ps["failed_tools"][:5]:
                tool, args, code = item[0], item[1], item[2]
                tag = ""
                if len(item) > 3:
                    if item[3] == "review_finding":
                        tag = "  [review finding]"
                    elif item[3] == "reviewer_error":
                        tag = "  [REVIEWER ERROR]"
                lines.append(f"  ✗ {tool}({args}) → exit {code}{tag}")
            if len(ps["failed_tools"]) > 5:
                lines.append(f"  ... and {len(ps['failed_tools'])-5} more failures")

        if ps["markers"]:
            lines.append(f"  markers: {', '.join(ps['markers'])}")

        # Untested completion flag for this session
        if s_id in findings.untested_completion_ids:
            lines.append(f"  ✗ [UNTESTED COMPLETION] TASK_COMPLETE declared with zero bash commands executed")

        # Suspicious validations for this session
        if s_id in findings.suspicious_validations:
            for sv in findings.suspicious_validations[s_id]:
                lines.append(f"  ⚠ iter {sv['iter']}: [SUSPICIOUS VALIDATION] {sv['reason']}")
                lines.append(f"    cmd: {sv['command'][:100]}")

        # Unresolved exit codes at final iteration
        if s_id in findings.unresolved_exits:
            for ue in findings.unresolved_exits[s_id]:
                lines.append(f"  ✗ [UNRESOLVED EXIT] Final iter {ue['iter']}: "
                             f"{ue['command'][:80]} → exit {ue['exit_code']}")
                if ue["error_snippet"]:
                    lines.append(f"    {ue['error_snippet'][:120]}")

        # Write without test
        if s_id in findings.write_without_tests:
            for wt in findings.write_without_tests[s_id]:
                lines.append(f"  ⚠ iter {wt['iter_written']}: [WRITE WITHOUT TEST] "
                             f"{wt['path']} written but never referenced in bash")

        # Repeated failing command
        if s_id in findings.repeated_fails:
            for rf in findings.repeated_fails[s_id]:
                lines.append(f"  ⚠ [REPEATED FAILURE] `{rf['command'][:80]}` → exit {rf['exit_code']} "
                             f"x{rf['count']} (iters {rf['iterations']}) with no code change")

        # Marker mismatches
        if s_id in findings.marker_mismatches:
            for reason in findings.marker_mismatches[s_id]:
                lines.append(f"  ⚠ [MARKER MISMATCH] {reason}")

        lines.append("")

    # Command prefix pattern
    if findings.command_prefix:
        lines.append(f"{'─'*70}")
        lines.append(f"  PATTERN: 'command' prefix hallucination in bash tool calls")
        lines.append(f"  Affected sessions: {list(findings.command_prefix.keys())} "
                     f"({findings.total_command_prefix_count} total occurrences)")
        for sid, variants in findings.command_prefix.items():
            lines.append(f"    Session {sid}: {variants}")
        lines.append("")

    # Stop-sequence truncation warning
    if findings.stop_seq_truncation_ids:
        lines.append(f"⚠ STOP-SEQUENCE TRUNCATION in write_file: "
                     f"sessions {findings.stop_seq_truncation_ids}")
        lines.append(f"  Trailing quotes stripped by \\n</tool> stop sequence "
                     f"(bash.py has _fix_unbalanced_quotes; write_file does not).")
        for sid, hits in findings.stop_seq_truncations.items():
            for h in hits:
                lines.append(f"    Session {sid} iter {h['iter']}: {h['path']} — {h['reason']}")
        lines.append("")

    # Summary
    lines.append(f"{'─'*70}")
    if findings.problem_session_ids:
        if findings.execution_error_ids:
            lines.append(f"Sessions with execution errors: {findings.execution_error_ids} "
                         f"({len(findings.execution_error_ids)}/{findings.total_sessions})")
        if findings.review_finding_ids:
            lines.append(f"Sessions with review findings:  {findings.review_finding_ids} "
                         f"({len(findings.review_finding_ids)}/{findings.total_sessions})"
                         f"  — errors from testing code under review")
        if findings.reviewer_error_ids:
            lines.append(f"Sessions with reviewer errors:  {findings.reviewer_error_ids} "
                         f"({len(findings.reviewer_error_ids)}/{findings.total_sessions})"
                         f"  — reviewer itself malfunctioned")
        if findings.suspicious_validation_ids:
            lines.append(f"Suspicious validations:         {findings.suspicious_validation_ids} "
                         f"({len(findings.suspicious_validation_ids)}/{findings.total_sessions})"
                         f"  — commands claim success without real testing")
        if findings.untested_completion_ids:
            lines.append(f"Untested completions:           {findings.untested_completion_ids} "
                         f"({len(findings.untested_completion_ids)}/{findings.total_sessions})"
                         f"  — TASK_COMPLETE with zero bash commands")
        if findings.unresolved_exit_ids:
            lines.append(f"Unresolved exits at end:        {findings.unresolved_exit_ids} "
                         f"({len(findings.unresolved_exit_ids)}/{findings.total_sessions})"
                         f"  — final iteration still has non-zero exit codes")
        if findings.write_without_test_ids:
            lines.append(f"Write without test:             {findings.write_without_test_ids} "
                         f"({len(findings.write_without_test_ids)}/{findings.total_sessions})"
                         f"  — files written but never referenced in bash")
        if findings.repeated_fail_ids:
            lines.append(f"Repeated failing commands:      {findings.repeated_fail_ids} "
                         f"({len(findings.repeated_fail_ids)}/{findings.total_sessions})"
                         f"  — same command fails repeatedly with no code change")
        if findings.marker_mismatch_ids:
            lines.append(f"Marker/role mismatches:         {findings.marker_mismatch_ids} "
                         f"({len(findings.marker_mismatch_ids)}/{findings.total_sessions})"
                         f"  — markers contradict session role or each other")
        lines.append(f"Clean sessions:                 {findings.clean_session_ids} "
                     f"({len(findings.clean_session_ids)}/{findings.total_sessions})")
    else:
        if findings.suspicious_validation_ids:
            lines.append(f"Suspicious validations:         {findings.suspicious_validation_ids} "
                         f"({len(findings.suspicious_validation_ids)}/{findings.total_sessions})"
                         f"  — commands claim success without real testing")
        if findings.untested_completion_ids:
            lines.append(f"Untested completions:           {findings.untested_completion_ids} "
                         f"({len(findings.untested_completion_ids)}/{findings.total_sessions})"
                         f"  — TASK_COMPLETE with zero bash commands")
        if findings.unresolved_exit_ids:
            lines.append(f"Unresolved exits at end:        {findings.unresolved_exit_ids} "
                         f"({len(findings.unresolved_exit_ids)}/{findings.total_sessions})")
        if findings.write_without_test_ids:
            lines.append(f"Write without test:             {findings.write_without_test_ids} "
                         f"({len(findings.write_without_test_ids)}/{findings.total_sessions})")
        if findings.repeated_fail_ids:
            lines.append(f"Repeated failing commands:      {findings.repeated_fail_ids} "
                         f"({len(findings.repeated_fail_ids)}/{findings.total_sessions})")
        if findings.marker_mismatch_ids:
            lines.append(f"Marker/role mismatches:         {findings.marker_mismatch_ids} "
                         f"({len(findings.marker_mismatch_ids)}/{findings.total_sessions})")
        if not any([findings.suspicious_validation_ids, findings.untested_completion_ids,
                     findings.unresolved_exit_ids, findings.write_without_test_ids,
                     findings.repeated_fail_ids, findings.marker_mismatch_ids]):
            lines.append("No obvious problems detected in pre-filter.")
    lines.append("")

    return "\n".join(lines)


def format_findings_for_prompt(findings: PrefilterFindings) -> str:
    """Format pre-filter findings into a text block for LLM injection.

    This gives the LLM ground-truth data that it MUST incorporate,
    preventing it from miscounting errors or missing detected patterns.
    """
    parts = [
        "=== AUTOMATED PRE-FILTER FINDINGS (ground truth — do not contradict) ===",
        "",
        f"Total sessions: {findings.total_sessions}",
        f"Total iterations: {findings.total_iterations}",
        f"Sessions with execution errors: {len(findings.execution_error_ids)} — IDs: {findings.execution_error_ids}",
        f"Sessions with review findings: {len(findings.review_finding_ids)} — IDs: {findings.review_finding_ids}"
        f"  (reviewer tested code and found issues — this is expected, not a failure)",
    ]
    if findings.reviewer_error_ids:
        parts.append(
            f"Sessions with reviewer errors: {len(findings.reviewer_error_ids)} — IDs: {findings.reviewer_error_ids}"
            f"  (reviewer itself malfunctioned)"
        )
    parts.extend([
        f"Clean sessions: {len(findings.clean_session_ids)} — IDs: {findings.clean_session_ids}",
        "",
    ])

    # Pipeline role guide
    parts.append("PIPELINE ROLES — each session has a specific role with different expectations:")
    for role_name, info in ROLE_EXPECTATIONS.items():
        if role_name == ROLE_UNKNOWN:
            continue
        parts.append(f"  {role_name}: {info['description']}")
        parts.append(f"    {info['error_guidance']}")
    parts.append("")

    # Per-session summary with roles and error classification
    parts.append("Per-session breakdown:")
    for ps in findings.per_session:
        error_class = ps.get("error_class", "clean")
        if error_class == "clean":
            status = "CLEAN"
        elif error_class == "review_finding":
            status = "REVIEW_FINDINGS (errors are from testing code under review — expected)"
        elif error_class == "reviewer_error":
            status = "REVIEWER_ERROR (reviewer itself malfunctioned)"
        else:
            status = "EXECUTION_ERRORS"
        parts.append(f"  Session {ps['id']} [{ps['role']}]: {status}, {ps['iterations']} iters, "
                     f"outcome={ps['outcome']}, error_iters={ps['errors']}, "
                     f"failed_tools={len(ps['failed_tools'])}")

    parts.append("")

    # Command prefix pattern — explicit and detailed
    if findings.command_prefix:
        total = findings.total_command_prefix_count
        affected = list(findings.command_prefix.keys())
        parts.append(f"DETECTED PATTERN — 'command' prefix hallucination:")
        parts.append(f"  This is a MAJOR pattern affecting {len(affected)} sessions "
                     f"with {total} total occurrences.")
        parts.append(f"  The LLM prepends the word 'command' before bash commands inside "
                     f"<tool name=\"bash\"> tags.")
        parts.append(f"  This works accidentally because 'command' is a bash builtin, "
                     f"but it reveals fragile tool-call formatting.")
        parts.append(f"  Affected sessions: {affected}")
        for sid, variants in findings.command_prefix.items():
            parts.append(f"    Session {sid}: {variants}")
        parts.append(f"  YOU MUST include this pattern in your Recurring Patterns section.")
    else:
        parts.append("No 'command' prefix hallucination detected.")

    # Suspicious validations
    if findings.suspicious_validations:
        parts.append("")
        parts.append("DETECTED PATTERN — suspicious validations (bogus tests):")
        parts.append("  Commands that exited 0 but appear to test nothing meaningful.")
        parts.append("  These are SUSPICIOUS, not confirmed errors — the LLM analysis "
                     "should evaluate whether each flagged command is truly bogus.")
        for sid, items in findings.suspicious_validations.items():
            for item in items:
                parts.append(f"  Session {sid} iter {item['iter']}: {item['reason']}")
                parts.append(f"    cmd: {item['command'][:120]}")

    # Untested completions
    if findings.untested_completion_ids:
        parts.append("")
        parts.append("DETECTED PATTERN — untested completions:")
        parts.append("  Coder sessions that declared [TASK_COMPLETE] without executing ANY bash commands.")
        parts.append("  This means zero testing was performed — the coder only wrote files and/or reasoned.")
        parts.append(f"  Affected sessions: {findings.untested_completion_ids}")
        parts.append("  YOU MUST flag these sessions as critical failures in your analysis.")

    # Unresolved exits at final iteration
    if findings.unresolved_exits:
        parts.append("")
        parts.append("DETECTED PATTERN — unresolved exits at final iteration:")
        parts.append("  Sessions where the last iteration still has non-zero exit codes.")
        parts.append("  The agent ended without resolving these errors.")
        for sid, exits in findings.unresolved_exits.items():
            for ue in exits:
                parts.append(f"  Session {sid} iter {ue['iter']}: "
                             f"{ue['command'][:100]} → exit {ue['exit_code']}")
                if ue["error_snippet"]:
                    parts.append(f"    {ue['error_snippet'][:150]}")

    # Write without test
    if findings.write_without_tests:
        parts.append("")
        parts.append("DETECTED PATTERN — write_file without subsequent test:")
        parts.append("  Files were written but never referenced in any subsequent bash command.")
        for sid, writes in findings.write_without_tests.items():
            for wt in writes:
                parts.append(f"  Session {sid} iter {wt['iter_written']}: {wt['path']}")

    # Repeated failing commands
    if findings.repeated_fails:
        parts.append("")
        parts.append("DETECTED PATTERN — repeated identical failing commands:")
        parts.append("  The same command failed multiple times with no code change between attempts.")
        parts.append("  This indicates the agent retried without fixing the underlying issue.")
        for sid, fails in findings.repeated_fails.items():
            for rf in fails:
                parts.append(f"  Session {sid}: `{rf['command'][:100]}` → exit {rf['exit_code']} "
                             f"x{rf['count']} (iters {rf['iterations']})")

    # Marker/role mismatches
    if findings.marker_mismatches:
        parts.append("")
        parts.append("DETECTED PATTERN — marker/role mismatches:")
        parts.append("  Sessions where markers contradict the detected role or each other.")
        for sid, reasons in findings.marker_mismatches.items():
            for reason in reasons:
                parts.append(f"  Session {sid}: {reason}")

    parts.append("")
    parts.append("=== END PRE-FILTER FINDINGS ===")
    return "\n".join(parts)


def _format_session_prefilter_flags(
    session_id: int, findings: PrefilterFindings,
) -> str:
    """Format pre-filter behavioral flags for a specific session.

    Returns a text block to inject into the map prompt so the LLM knows about
    deterministically detected issues (untested completions, write-without-test,
    etc.) that aren't visible from the fact sheet alone.
    """
    flags: list[str] = []

    if session_id in findings.untested_completion_ids:
        flags.append(
            "⚠ UNTESTED COMPLETION: This session declared [TASK_COMPLETE] without "
            "executing ANY bash commands. Zero testing was performed. "
            "You MUST report this as an 'untested_completion' finding. "
            "Use evidence_ref 'iter=0/marker=TASK_COMPLETE' with the iteration "
            "where the marker was emitted."
        )

    if session_id in findings.write_without_tests:
        for wt in findings.write_without_tests[session_id]:
            flags.append(
                f"⚠ WRITE WITHOUT TEST: File '{wt['path']}' was written at "
                f"iter {wt['iter_written']} but never referenced in any subsequent "
                f"bash command. You MUST report this as a 'write_without_test' finding. "
                f"Use evidence_ref 'iter={wt['iter_written']}/tool=write_file'."
            )

    if session_id in findings.suspicious_validations:
        for sv in findings.suspicious_validations[session_id]:
            flags.append(
                f"⚠ SUSPICIOUS VALIDATION: iter {sv['iter']}: {sv['reason']}. "
                f"cmd: {sv['command'][:120]}. "
                f"You MUST report this as a 'suspicious_validation' finding."
            )

    if session_id in findings.unresolved_exits:
        for ue in findings.unresolved_exits[session_id]:
            flags.append(
                f"⚠ UNRESOLVED EXIT: Final iter {ue['iter']}: "
                f"{ue['command'][:80]} → exit {ue['exit_code']}. "
                f"The session ended with this error unresolved. "
                f"You MUST report this as an 'unresolved_error' finding."
            )

    if session_id in findings.repeated_fails:
        for rf in findings.repeated_fails[session_id]:
            flags.append(
                f"⚠ REPEATED FAILURE: `{rf['command'][:80]}` → exit {rf['exit_code']} "
                f"x{rf['count']} times (iters {rf['iterations']}) with no code change. "
                f"You MUST report this as a 'repeated_failure' finding."
            )

    if session_id in findings.marker_mismatches:
        for reason in findings.marker_mismatches[session_id]:
            flags.append(f"⚠ MARKER MISMATCH: {reason}.")

    if not flags:
        return ""

    parts = [
        "=== PRE-FILTER BEHAVIORAL FLAGS (deterministic — these are confirmed issues) ===",
        "The following issues were detected by automated analysis and are ground truth.",
        "You MUST include each of these as a finding in your JSON output.",
        "",
    ]
    for flag in flags:
        parts.append(flag)
    parts.append("")
    parts.append("=== END PRE-FILTER FLAGS ===")
    return "\n".join(parts)
