"""Text manipulation utilities shared across modules."""

from __future__ import annotations

from pathlib import Path


def _truncate(text: str, max_chars: int = 6000) -> str:
    """Truncate long text, keeping head and tail."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + f"\n\n[... {len(text) - max_chars} chars truncated ...]\n\n" + text[-half:]


def _normalize_paths(text: str, project_root: str) -> str:
    """Replace long absolute paths with short aliases to save tokens.

    Replaces the project root with $ROOT and the home directory with ~.
    """
    result = text
    # Replace project root (most specific, do first)
    if project_root and project_root != "/":
        result = result.replace(project_root, "$ROOT")
    # Replace home directory
    home = str(Path.home()).rstrip("/")
    if home and home != "/":
        result = result.replace(home, "~")
    return result


def _collapse_whitespace(text: str) -> str:
    """Collapse multiple blank lines into one and strip trailing whitespace per line."""
    lines = [line.rstrip() for line in text.splitlines()]
    collapsed = []
    prev_blank = False
    for line in lines:
        if not line:
            if not prev_blank:
                collapsed.append(line)
            prev_blank = True
        else:
            collapsed.append(line)
            prev_blank = False
    return "\n".join(collapsed)


def _deduplicate_lines(text: str) -> str:
    """Collapse 3+ consecutive identical non-blank lines into one with a count."""
    lines = text.splitlines()
    if not lines:
        return text
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        count = 1
        while i + count < len(lines) and lines[i + count] == line:
            count += 1
        result.append(line)
        if count >= 3 and line.strip():
            result.append(f"  [above line repeated {count - 1} more times]")
        elif count == 2:
            result.append(line)
        i += count
    return "\n".join(result)


def _apply_text_compression(text: str, project_root: str) -> str:
    """Apply path normalization and whitespace collapsing to a text block."""
    if project_root:
        text = _normalize_paths(text, project_root)
    return _collapse_whitespace(text)
