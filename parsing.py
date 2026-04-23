"""Core data structures and log file parsing."""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from config import (
    ERROR_PATTERNS,
    EXIT_CODE_RE,
    HEADER_DIVIDER,
    ITERATION_FILENAME_RE,
    MARKERS,
    ROLE_CODER,
    ROLE_DELEGATOR,
    ROLE_INTEGRATION,
    ROLE_REVIEWER,
    ROLE_UNKNOWN,
    SECTION_REASONING,
    SECTION_RESPONSE,
    SECTION_TOOL_OUTPUT,
    TIMESTAMP_FORMAT,
    TIMESTAMP_RE,
    TOOL_ARGS_RE,
    TOOL_NAME_RE,
    TOOL_OUTPUT_SEPARATOR,
    TOOL_OUTPUT_SEPARATOR_FALLBACK,
)


# ── Data Structures ──────────────────────────────────────────────────────────


@dataclass
class ToolCall:
    tool_name: str
    args: str
    output: str


@dataclass
class LogEntry:
    filepath: Path
    timestamp: datetime
    iteration: int
    reasoning: str
    response: str
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        full_text = self.response + "\n".join(t.output for t in self.tool_calls)
        return any(p.search(full_text) for p in ERROR_PATTERNS)

    @property
    def has_tool_errors(self) -> bool:
        """Check for errors only in tool outputs (not response text).

        Useful for delegator sessions where the response is a JSON task plan
        containing natural language like "handle errors gracefully" that would
        false-positive on keyword matching.
        """
        if not self.tool_calls:
            return False
        tool_text = "\n".join(t.output for t in self.tool_calls)
        return any(p.search(tool_text) for p in ERROR_PATTERNS)

    @property
    def markers_found(self) -> list[str]:
        full_text = self.response
        return [m for m in MARKERS if m in full_text]

    @property
    def exit_codes(self) -> list[int]:
        """Extract non-zero exit codes from tool outputs."""
        codes = []
        for tc in self.tool_calls:
            match = EXIT_CODE_RE.search(tc.output)
            if match:
                code = int(match.group(1))
                if code != 0:
                    codes.append(code)
        return codes


@dataclass
class Session:
    """A group of sequential log entries forming one task execution."""

    session_id: int
    entries: list[LogEntry] = field(default_factory=list)

    @property
    def start_time(self) -> datetime:
        return self.entries[0].timestamp

    @property
    def end_time(self) -> datetime:
        return self.entries[-1].timestamp

    @property
    def total_iterations(self) -> int:
        return len(self.entries)

    @property
    def max_iteration(self) -> int:
        return max(e.iteration for e in self.entries)

    @property
    def error_count(self) -> int:
        # Delegator responses are JSON task plans full of natural language
        # like "handle errors gracefully" — only check their tool outputs.
        if self.role == ROLE_DELEGATOR:
            return sum(1 for e in self.entries if e.has_tool_errors)
        return sum(1 for e in self.entries if e.has_errors)

    @property
    def all_markers(self) -> list[str]:
        markers = []
        for e in self.entries:
            markers.extend(e.markers_found)
        return markers

    @property
    def outcome(self) -> str:
        markers = self.all_markers
        if "[INTEGRATION_PASS]" in markers:
            return "INTEGRATION_PASS"
        if "[INTEGRATION_FAIL]" in markers:
            return "INTEGRATION_FAIL"
        if "[REVIEW_PASS]" in markers:
            return "REVIEW_PASS"
        if "[REVIEW_FAIL]" in markers:
            return "REVIEW_FAIL"
        if "[TASK_COMPLETE]" in markers:
            return "TASK_COMPLETE (no review)"
        return "NO_MARKER (unclear outcome)"

    @property
    def role(self) -> str:
        return _detect_session_role(self)

    @property
    def total_chars(self) -> int:
        return sum(
            len(e.reasoning) + len(e.response) + sum(len(t.output) for t in e.tool_calls)
            for e in self.entries
        )


# ── Role Detection ────────────────────────────────────────────────────────────


def _detect_session_role(session: Session) -> str:
    """Detect the pipeline role of a session from its content.

    Detection priority:
    1. Markers (most reliable — each role uses distinct markers)
    2. Content heuristics (for NO_MARKER sessions)
    """
    markers = session.all_markers

    # Integration reviewer markers
    if any(m.startswith("[INTEGRATION_") for m in markers):
        return ROLE_INTEGRATION

    # Reviewer markers
    if any(m.startswith("[REVIEW_") for m in markers):
        return ROLE_REVIEWER

    # Coder markers
    if "[TASK_COMPLETE]" in markers or "[OBSERVATIONS]" in markers:
        return ROLE_CODER

    # No markers — use content heuristics
    all_tool_calls = [tc for e in session.entries for tc in e.tool_calls]
    has_tool_calls = len(all_tool_calls) > 0

    # Check for write_file tool calls (strong coder signal)
    has_write_file = any(tc.tool_name == "write_file" for tc in all_tool_calls)
    if has_write_file:
        return ROLE_CODER

    # No tool calls + few iterations: delegator or integration reviewer
    if not has_tool_calls and session.total_iterations <= 2:
        full_response = " ".join(e.response for e in session.entries)
        full_reasoning = " ".join(e.reasoning for e in session.entries)
        combined_lower = (full_response + " " + full_reasoning).lower()

        # Integration reviewer: mentions integration testing, cross-module, end-to-end
        integration_signals = [
            "integration test", "integration review", "cross-module",
            "end-to-end", "modules work together", "integration_pass",
            "integration_fail", "integration_blocked",
        ]
        if any(sig in combined_lower for sig in integration_signals):
            return ROLE_INTEGRATION

        # Reviewer that didn't produce a marker
        reviewer_signals = [
            "re-review", "review the code", "code review", "checklist",
            "review_pass", "review_fail", "behavioral analysis",
        ]
        if any(sig in combined_lower for sig in reviewer_signals):
            return ROLE_REVIEWER

        # Delegator: response contains JSON-like task structure
        json_signals = ['"tasks"', '"instructions"', '"agent_type"', '"description"']
        if any(sig in full_response for sig in json_signals):
            return ROLE_DELEGATOR

        # Delegator: reasoning mentions task planning
        delegator_signals = ["break down", "task plan", "break this into", "decompos"]
        if any(sig in combined_lower for sig in delegator_signals):
            return ROLE_DELEGATOR

    # Has tool calls but no markers — likely a coder that didn't finish
    if has_tool_calls:
        return ROLE_CODER

    return ROLE_UNKNOWN


def _classify_tool_failure(
    tool_name: str, args: str, exit_code: int, output: str, session_role: str,
) -> str:
    """Classify a failed tool call relative to the session's role.

    Returns:
        'execution_error' — failure in the session's own work
        'review_finding' — reviewer testing code under review, failure is a finding
        'reviewer_error' — reviewer's own command was malformed or wrong
    """
    # Non-reviewer sessions: all failures are execution errors
    if session_role not in (ROLE_REVIEWER, ROLE_INTEGRATION):
        return "execution_error"

    output_lower = output.lower() if output else ""

    # Signs the reviewer's own command was broken (not the code under review)
    if "command not found" in output_lower:
        return "reviewer_error"

    # Script file doesn't exist — reviewer referenced wrong path
    if "can't open file" in output_lower and "no such file or directory" in output_lower:
        return "reviewer_error"

    # Non-Python "No such file or directory" without a traceback
    if "no such file or directory" in output_lower and "traceback" not in output_lower:
        return "reviewer_error"

    # Default for reviewer sessions: failures are findings about code under review
    return "review_finding"


# ── Tool Argument Extraction ──────────────────────────────────────────────────


def _extract_bash_command(args_str: str) -> str:
    """Extract the bash command string from a tool call's args field.

    Args are typically in Python dict repr format:
        {'command': 'python3 main.py --url "test"'}
    """
    try:
        parsed = ast.literal_eval(args_str)
        if isinstance(parsed, dict) and "command" in parsed:
            return str(parsed["command"])
    except (ValueError, SyntaxError, TypeError):
        pass

    # Regex fallback — greedy match for command value
    for pattern in [
        r"'command'\s*:\s*'(.+)'\s*\}?\s*$",
        r'"command"\s*:\s*"(.+)"\s*\}?\s*$',
    ]:
        m = re.search(pattern, args_str, re.DOTALL)
        if m:
            return m.group(1)

    return ""


def _extract_write_file_path(args_str: str) -> str:
    """Extract the file path from a write_file tool call's args field."""
    try:
        parsed = ast.literal_eval(args_str)
        if isinstance(parsed, dict):
            raw = str(parsed.get("command", parsed.get("path", "")))
            if raw.startswith("path:"):
                return raw[5:].split("\n")[0].strip()
            if "path" in parsed:
                return str(parsed["path"])
    except (ValueError, SyntaxError, TypeError):
        pass
    # Fallback: match path: <name> and stop at literal \n or whitespace
    m = re.search(r"path[=:]\s*([^\s\\]+)", args_str)
    return m.group(1) if m else "?"


def _extract_write_file_content(args_str: str) -> str:
    """Extract file content from a write_file tool call's args field.

    Args are typically in Python dict repr format with the content after
    a ``\\n---\\n`` separator inside the 'command' value.

    Returns the file content string, or empty string if unparseable.
    """
    # Parse the dict repr to get the actual string with real newlines
    raw = ""
    try:
        parsed = ast.literal_eval(args_str)
        if isinstance(parsed, dict) and "command" in parsed:
            raw = str(parsed["command"])
    except (ValueError, SyntaxError, TypeError):
        pass
    if not raw:
        # Regex fallback
        for pattern in [
            r"'command'\s*:\s*'(.+)'\s*\}?\s*$",
            r'"command"\s*:\s*"(.+)"\s*\}?\s*$',
        ]:
            m = re.search(pattern, args_str, re.DOTALL)
            if m:
                raw = m.group(1)
                break
    if not raw:
        return ""
    sep_idx = raw.find("\n---\n")
    if sep_idx == -1:
        return ""
    return raw[sep_idx + 5:]


# ── Parsing ───────────────────────────────────────────────────────────────────


def parse_log_file(filepath: Path) -> LogEntry:
    """Parse a single log file into a LogEntry."""
    text = filepath.read_text(errors="replace")

    # Extract timestamp
    ts_match = TIMESTAMP_RE.search(text)
    timestamp = datetime.min
    if ts_match:
        try:
            timestamp = datetime.strptime(ts_match.group(1).strip(), TIMESTAMP_FORMAT)
        except ValueError:
            pass

    # Extract iteration from filename (more reliable than file content)
    iter_match = ITERATION_FILENAME_RE.search(filepath.name)
    iteration = int(iter_match.group(1)) if iter_match else 0

    # Split into sections
    reasoning = ""
    response = ""

    if SECTION_REASONING in text:
        parts = text.split(SECTION_REASONING, 1)
        after_reasoning = parts[1]

        if SECTION_RESPONSE in after_reasoning:
            reasoning_part, after_response = after_reasoning.split(SECTION_RESPONSE, 1)
            reasoning = reasoning_part.strip()
            # Response is everything until first tool output or end
            if SECTION_TOOL_OUTPUT in after_response:
                response = after_response.split(SECTION_TOOL_OUTPUT, 1)[0].strip()
            else:
                response = after_response.strip()
        else:
            # Reasoning exists but no separate response section
            if SECTION_TOOL_OUTPUT in after_reasoning:
                reasoning = after_reasoning.split(SECTION_TOOL_OUTPUT, 1)[0].strip()
            else:
                reasoning = after_reasoning.strip()
    elif HEADER_DIVIDER in text:
        # No reasoning section — response is after the header
        header_end = text.rfind(HEADER_DIVIDER)
        after_header = text[header_end + len(HEADER_DIVIDER):]
        if SECTION_TOOL_OUTPUT in after_header:
            response = after_header.split(SECTION_TOOL_OUTPUT, 1)[0].strip()
        else:
            response = after_header.strip()

    # Extract tool calls
    tool_calls = []
    tool_sections = text.split(SECTION_TOOL_OUTPUT)[1:] if SECTION_TOOL_OUTPUT in text else []
    for section in tool_sections:
        tc = _parse_tool_section(section)
        if tc:
            tool_calls.append(tc)

    return LogEntry(
        filepath=filepath,
        timestamp=timestamp,
        iteration=iteration,
        reasoning=reasoning,
        response=response,
        tool_calls=tool_calls,
    )


def _parse_tool_section(section: str) -> ToolCall | None:
    """Parse a single tool output section."""
    tool_match = TOOL_NAME_RE.search(section)
    args_match = TOOL_ARGS_RE.search(section)

    tool_name = tool_match.group(1).strip() if tool_match else "unknown"
    args = args_match.group(1).strip() if args_match else ""

    # Output is everything after the --- separator line.
    # Use "\n---\n" (line-delimited) to avoid splitting on "---" that
    # appears inside the Args dict repr (e.g. write_file content has
    # a \n---\n separator that shows as literal \n---\n in the repr).
    if TOOL_OUTPUT_SEPARATOR in section:
        output = section.split(TOOL_OUTPUT_SEPARATOR, 1)[1].strip()
    elif section.strip().endswith(TOOL_OUTPUT_SEPARATOR_FALLBACK):
        output = ""
    elif TOOL_OUTPUT_SEPARATOR_FALLBACK in section:
        output = section.split(TOOL_OUTPUT_SEPARATOR_FALLBACK, 1)[1].strip()
    else:
        output = ""

    return ToolCall(tool_name=tool_name, args=args, output=output)


def load_all_logs(log_dir: Path) -> list[LogEntry]:
    """Load and parse all log files, sorted by timestamp then iteration."""
    files = sorted(log_dir.glob("*.txt"))
    if not files:
        print(f"No log files found in {log_dir}")
        sys.exit(1)

    entries = []
    for f in files:
        try:
            entries.append(parse_log_file(f))
        except Exception as e:
            print(f"  Warning: failed to parse {f.name}: {e}")

    # Sort by timestamp, then iteration
    entries.sort(key=lambda e: (e.timestamp, e.iteration))
    return entries


def group_into_sessions(entries: list[LogEntry]) -> list[Session]:
    """Group entries into sessions by detecting iteration resets to 0."""
    if not entries:
        return []

    sessions: list[Session] = []
    current = Session(session_id=0)
    prev_iter = -1

    for entry in entries:
        # New session when iteration resets to 0 (and we already have entries)
        if entry.iteration == 0 and current.entries:
            sessions.append(current)
            current = Session(session_id=len(sessions))

        current.entries.append(entry)
        prev_iter = entry.iteration

    if current.entries:
        sessions.append(current)

    return sessions


# ── Token Counting ────────────────────────────────────────────────────────────


def count_log_tokens(log_dir: Path) -> tuple[int, str]:
    """Count the total tokens across all log files in the directory.

    Returns (token_count, method_description).
    Uses tiktoken if available, otherwise estimates from character count.
    """
    files = sorted(log_dir.glob("*.txt"))

    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        total_tokens = 0
        for f in files:
            try:
                text = f.read_text(errors="replace")
                total_tokens += len(enc.encode(text, disallowed_special=()))
            except OSError:
                continue
        return total_tokens, "tiktoken/cl100k_base"
    except ImportError:
        pass

    # Fallback: estimate from character count (~3.5 chars per token for mixed code/log text)
    total_chars = 0
    for f in files:
        try:
            total_chars += f.stat().st_size
        except OSError:
            continue
    token_count = int(total_chars / 3.5)
    return token_count, f"estimated (~3.5 chars/token from {total_chars:,} chars)"
