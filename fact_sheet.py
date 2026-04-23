"""Per-session fact sheets and structured map results.

The fact sheet is the deterministic ground-truth view of a session, extracted
directly from parsed log data with no LLM involvement. LLM map analysis must
not contradict it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from config import ERROR_PATTERNS, EXIT_CODE_RE, VALID_FINDING_TYPES, _EXCEPTION_LINE_RE
from parsing import (
    Session,
    _extract_bash_command,
    _extract_write_file_content,
    _extract_write_file_path,
)
from prefilter import _check_bogus_validation, _check_stop_seq_truncation


# ── Session Fact Sheet ────────────────────────────────────────────────────────


@dataclass
class SessionFactSheet:
    """All verifiable facts for one session, extracted deterministically.

    Every field is computed from parsed log data — no LLM involved.
    Used as ground truth to verify LLM analysis claims.
    """
    session_id: int
    role: str
    outcome: str
    iterations: int
    time_range: str

    # Every tool call with exact args and results
    tool_calls: list[dict] = field(default_factory=list)

    # Every file created/modified (from write_file calls)
    files_written: list[dict] = field(default_factory=list)

    # Non-zero exit code errors with details
    errors: list[dict] = field(default_factory=list)

    # Markers found and which iteration
    markers: list[dict] = field(default_factory=list)

    # Whether the final iteration had no errors
    final_iteration_clean: bool = True

    # Searchable: all bash commands as exact strings
    all_commands: list[str] = field(default_factory=list)

    # Searchable: all non-zero exit codes that occurred
    all_exit_codes: set = field(default_factory=set)

    # Searchable: all error snippets joined
    all_error_text: str = ""

    # Stop-sequence truncation detections in write_file calls
    stop_seq_truncations: list[dict] = field(default_factory=list)


def build_fact_sheet(session: Session) -> SessionFactSheet:
    """Extract all verifiable facts from a session. Purely deterministic."""
    tool_calls_list = []
    files_written = []
    errors = []
    markers_list = []
    all_commands = []
    all_exit_codes = set()
    error_snippets = []
    stop_seq_truncations = []

    for entry in session.entries:
        it = entry.iteration

        # Markers
        for m in entry.markers_found:
            markers_list.append({"iter": it, "marker": m})

        # Tool calls
        for tc in entry.tool_calls:
            # Exit code
            exit_match = EXIT_CODE_RE.search(tc.output)
            exit_code = int(exit_match.group(1)) if exit_match else None

            # Error detection — prefer actual exception lines (e.g.
            # "SyntaxError: unterminated f-string literal") over generic
            # matches like "Traceback ..." or "Exit code: 1".
            has_error = False
            error_snippet = ""
            if tc.output:
                has_error = any(p.search(tc.output) for p in ERROR_PATTERNS)
                if has_error:
                    best = ""
                    first = ""
                    for line in tc.output.splitlines():
                        if any(p.search(line) for p in ERROR_PATTERNS):
                            if not first:
                                first = line.strip()[:200]
                            # Prefer actual exception lines over generic matches
                            if _EXCEPTION_LINE_RE.match(line):
                                best = line.strip()[:200]
                    error_snippet = best or first

            # Command extraction for bash
            command_text = ""
            suspicious_reason = ""
            if tc.tool_name == "bash":
                command_text = _extract_bash_command(tc.args)
                if command_text:
                    all_commands.append(command_text)
                    # Check for bogus validation (only on exit-0 commands)
                    if exit_code == 0:
                        suspicious_reason = _check_bogus_validation(command_text)

            tool_calls_list.append({
                "iter": it,
                "tool": tc.tool_name,
                "command": command_text,
                "args_raw": tc.args[:300],
                "exit_code": exit_code,
                "has_error": has_error,
                "error_snippet": error_snippet,
                "suspicious_validation": suspicious_reason,
            })

            # Track errors
            if exit_code is not None and exit_code != 0:
                all_exit_codes.add(exit_code)
                errors.append({
                    "iter": it,
                    "tool": tc.tool_name,
                    "command": command_text[:200],
                    "error_text": error_snippet,
                    "exit_code": exit_code,
                })
                if error_snippet:
                    error_snippets.append(error_snippet)

            # Track file writes and check for stop-sequence truncation
            if tc.tool_name == "write_file":
                path_match = re.search(r"path[=:]\s*(\S+)", tc.args)
                if path_match:
                    files_written.append({"iter": it, "path": path_match.group(1)})
                # Extract file content and check for truncation artifacts
                file_content = _extract_write_file_content(tc.args)
                if file_content:
                    trunc_reason = _check_stop_seq_truncation(file_content)
                    if trunc_reason:
                        stop_seq_truncations.append({
                            "iter": it,
                            "path": _extract_write_file_path(tc.args),
                            "reason": trunc_reason,
                        })

    final_clean = not session.entries[-1].has_errors if session.entries else True

    return SessionFactSheet(
        session_id=session.session_id,
        role=session.role,
        outcome=session.outcome,
        iterations=session.total_iterations,
        time_range=f"{session.start_time:%H:%M:%S} → {session.end_time:%H:%M:%S}",
        tool_calls=tool_calls_list,
        files_written=files_written,
        errors=errors,
        markers=markers_list,
        final_iteration_clean=final_clean,
        all_commands=all_commands,
        all_exit_codes=all_exit_codes,
        all_error_text=" | ".join(error_snippets),
        stop_seq_truncations=stop_seq_truncations,
    )


def format_fact_sheet(fs: SessionFactSheet) -> str:
    """Format a fact sheet as structured text for LLM consumption."""
    parts = []
    parts.append(
        f"=== VERIFIED FACTS (Session {fs.session_id}) — "
        f"extracted deterministically, do not contradict ==="
    )
    parts.append(
        f"Role: {fs.role} | Outcome: {fs.outcome} | "
        f"Iterations: {fs.iterations} | Time: {fs.time_range}"
    )
    parts.append(f"Final iteration clean: {fs.final_iteration_clean}")
    parts.append("")

    if fs.tool_calls:
        parts.append("ALL TOOL CALLS (in execution order):")
        for tc in fs.tool_calls:
            status = "ERROR" if tc["has_error"] else "OK"
            if tc.get("suspicious_validation"):
                status = "SUSPICIOUS"
            exit_str = f" → exit {tc['exit_code']}" if tc["exit_code"] is not None else ""
            cmd_display = tc["command"][:120] if tc["command"] else tc["args_raw"][:120]
            parts.append(
                f"  iter {tc['iter']} | {tc['tool']}: {cmd_display}{exit_str} [{status}]"
            )
            if tc["error_snippet"]:
                parts.append(f"           Error: {tc['error_snippet'][:150]}")
            if tc.get("suspicious_validation"):
                parts.append(f"           ⚠ Suspicious: {tc['suspicious_validation']}")
        parts.append("")

    if fs.files_written:
        parts.append("FILES WRITTEN:")
        for fw in fs.files_written:
            parts.append(f"  iter {fw['iter']} | {fw['path']}")
        parts.append("")

    if fs.errors:
        parts.append(f"ERRORS ({len(fs.errors)} total, non-zero exit codes):")
        for err in fs.errors:
            parts.append(
                f"  iter {err['iter']} | {err['tool']}: "
                f"{err['command'][:100]} → exit {err['exit_code']}"
            )
            if err["error_text"]:
                parts.append(f"           {err['error_text'][:150]}")
        parts.append("")
    else:
        parts.append("ERRORS: none")
        parts.append("")

    if fs.markers:
        parts.append("MARKERS FOUND:")
        for m in fs.markers:
            parts.append(f"  iter {m['iter']} | {m['marker']}")
        parts.append("")

    if fs.stop_seq_truncations:
        parts.append(
            f"⚠ STOP-SEQUENCE TRUNCATION DETECTED "
            f"({len(fs.stop_seq_truncations)} write_file calls affected):"
        )
        parts.append(
            "  The stop sequence (\\n</tool>) is known to consume trailing "
            "closing quotes — see bash.py _fix_unbalanced_quotes."
        )
        parts.append(
            "  write_file has NO equivalent fix, so the LLM's correct closing "
            "quote gets silently stripped before the file is written."
        )
        for t in fs.stop_seq_truncations:
            parts.append(f"  iter {t['iter']} | {t['path']}: {t['reason']}")
        parts.append(
            "  ROOT CAUSE: This is an infrastructure bug, NOT an LLM behavior "
            "failure. The LLM generates the correct code, but the closing "
            "character is consumed by stop-sequence token stripping."
        )
        parts.append("")

    parts.append("=== END VERIFIED FACTS ===")
    return "\n".join(parts)


# ── Structured Map Output ─────────────────────────────────────────────────────


@dataclass
class MapFinding:
    """A single finding from the map analysis of one session."""
    type: str
    iteration: int
    tool: str
    command_exact: str
    exit_code: int | None
    evidence: str
    evidence_ref: str       # e.g. "iter=4/tool=bash/exit=1"
    confidence: float
    supported_by_fact_sheet: bool

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "iteration": self.iteration,
            "tool": self.tool,
            "command_exact": self.command_exact,
            "exit_code": self.exit_code,
            "evidence": self.evidence,
            "evidence_ref": self.evidence_ref,
            "confidence": self.confidence,
            "supported_by_fact_sheet": self.supported_by_fact_sheet,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MapFinding":
        return cls(
            type=str(d.get("type", "execution_error")),
            iteration=int(d.get("iteration", 0)),
            tool=str(d.get("tool", "")),
            command_exact=str(d.get("command_exact", "")),
            exit_code=d.get("exit_code"),
            evidence=str(d.get("evidence", "")),
            evidence_ref=str(d.get("evidence_ref", "")),
            confidence=float(d.get("confidence", 0.5)),
            supported_by_fact_sheet=bool(d.get("supported_by_fact_sheet", False)),
        )


@dataclass
class MapResult:
    """Structured output from the map analysis of one session."""
    task_summary: str
    findings: list[MapFinding]
    unknowns: list[str]
    final_assessment: str
    severity: str   # CRITICAL | WARNING | INFO

    def to_dict(self) -> dict:
        return {
            "task_summary": self.task_summary,
            "findings": [f.to_dict() for f in self.findings],
            "unknowns": self.unknowns,
            "final_assessment": self.final_assessment,
            "severity": self.severity,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "MapResult":
        findings = []
        for f in d.get("findings", []):
            if isinstance(f, dict):
                findings.append(MapFinding.from_dict(f))
        return cls(
            task_summary=str(d.get("task_summary", "")),
            findings=findings,
            unknowns=[str(u) for u in d.get("unknowns", [])],
            final_assessment=str(d.get("final_assessment", "")),
            severity=str(d.get("severity", "INFO")).upper(),
        )

    def to_prose(self, session_id: int) -> str:
        """Convert back to readable prose for the reducer and display."""
        parts = [f"## Session {session_id}"]
        parts.append(f"**Task:** {self.task_summary}")
        parts.append(f"**Severity:** {self.severity}")
        parts.append("")
        if self.findings:
            parts.append("**Findings:**")
            for f in self.findings:
                exit_str = f" exit={f.exit_code}" if f.exit_code is not None else ""
                parts.append(
                    f"- [{f.type}] iter {f.iteration}, {f.tool}{exit_str} "
                    f"(confidence: {f.confidence:.0%}, ref: {f.evidence_ref})"
                )
                if f.command_exact:
                    parts.append(f"  cmd: `{f.command_exact[:120]}`")
                if f.evidence:
                    parts.append(f"  evidence: {f.evidence[:200]}")
            parts.append("")
        if self.unknowns:
            parts.append("**Unknowns (insufficient evidence):**")
            for u in self.unknowns:
                parts.append(f"- {u}")
            parts.append("")
        parts.append(f"**Assessment:** {self.final_assessment}")
        return "\n".join(parts)


def _extract_json_from_response(text: str) -> dict | None:
    """Extract JSON from an LLM response, handling markdown fences."""
    # Try direct parse first
    text_stripped = text.strip()
    if text_stripped.startswith("{"):
        try:
            return json.loads(text_stripped)
        except json.JSONDecodeError:
            pass

    # Try extracting from ```json ... ``` fences
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)\n\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding the first { ... } block
    brace_start = text.find("{")
    if brace_start != -1:
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start:i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def _validate_map_result(data: dict, session: Session) -> list[str]:
    """Validate a parsed map result dict. Returns list of error strings."""
    errors: list[str] = []
    if not isinstance(data.get("task_summary"), str) or not data["task_summary"]:
        errors.append("Missing or empty task_summary")
    if not isinstance(data.get("findings"), list):
        errors.append("Missing or invalid findings array")
    else:
        for i, f in enumerate(data["findings"]):
            if not isinstance(f, dict):
                errors.append(f"Finding {i} is not an object")
                continue
            ftype = f.get("type", "")
            if ftype not in VALID_FINDING_TYPES:
                errors.append(f"Finding {i}: invalid type '{ftype}'")
            it = f.get("iteration")
            if it is not None and isinstance(it, (int, float)):
                if int(it) < 0 or int(it) > session.max_iteration:
                    errors.append(f"Finding {i}: iteration {it} out of range "
                                  f"(0-{session.max_iteration})")
            conf = f.get("confidence")
            if conf is not None and isinstance(conf, (int, float)):
                if conf < 0.0 or conf > 1.0:
                    errors.append(f"Finding {i}: confidence {conf} out of range (0-1)")
    if not isinstance(data.get("severity"), str):
        errors.append("Missing severity")
    elif data["severity"].upper() not in ("CRITICAL", "WARNING", "INFO"):
        errors.append(f"Invalid severity: {data['severity']}")
    return errors


def _parse_map_response(text: str, session: Session) -> MapResult | None:
    """Parse and validate a JSON map response. Returns None if invalid."""
    data = _extract_json_from_response(text)
    if data is None:
        return None
    validation_errors = _validate_map_result(data, session)
    if validation_errors:
        # Log but still try to use — partial results are better than nothing
        for err in validation_errors:
            print(f"    [map validation] {err}")
    try:
        return MapResult.from_dict(data)
    except (KeyError, TypeError, ValueError):
        return None
