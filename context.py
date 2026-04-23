"""Source code context loading and citation validation.

Loads source files from the agent system as context for the LLM,
so it can trace root causes back to specific code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from config import CONTEXT_BUDGET, MIN_PER_FILE, PRIORITY_FILES
from utils import _truncate


@dataclass
class ContextFile:
    """A source file loaded as context for the LLM."""
    path: Path
    content: str
    priority: int  # lower = higher priority


def _discover_context_paths(paths: list[Path]) -> list[Path]:
    """Expand directories into individual .py files, skip non-Python."""
    result = []
    for p in paths:
        if p.is_file() and p.suffix == ".py":
            result.append(p)
        elif p.is_dir():
            result.extend(sorted(p.rglob("*.py")))
    return result


def _assign_priority(filepath: Path) -> int:
    """Assign priority based on filename. Lower = more important."""
    name = filepath.name
    for i, pname in enumerate(PRIORITY_FILES):
        if name == pname:
            return i
    return len(PRIORITY_FILES)  # unknown files get lowest priority


def load_context_files(paths: list[Path]) -> list[ContextFile]:
    """Load source files, optionally truncating to fit CONTEXT_BUDGET.

    If CONTEXT_BUDGET is 0, files are included in full (no truncation).
    """
    all_paths = _discover_context_paths(paths)
    if not all_paths:
        return []

    # Read all files and assign priorities
    files: list[ContextFile] = []
    for p in all_paths:
        try:
            content = p.read_text(errors="replace")
            files.append(ContextFile(
                path=p,
                content=content,
                priority=_assign_priority(p),
            ))
        except OSError:
            continue

    # Sort by priority (most important first)
    files.sort(key=lambda f: f.priority)

    # Apply budget-based truncation only if CONTEXT_BUDGET is set
    if CONTEXT_BUDGET > 0:
        remaining = CONTEXT_BUDGET
        for i, cf in enumerate(files):
            files_left = len(files) - i
            share = max(remaining // files_left, MIN_PER_FILE)
            if len(cf.content) > share:
                cf.content = _truncate(cf.content, share)
            remaining -= len(cf.content)
            remaining = max(remaining, 0)

    return files


def format_context_for_prompt(context_files: list[ContextFile]) -> str:
    """Format loaded context files into a single text block for the LLM."""
    if not context_files:
        return ""

    parts = [
        "=== SOURCE CODE CONTEXT ===",
        "The following source files implement the AI agent system that produced the logs above.",
        "Use this code to trace root causes — e.g., prompt wording that confuses the LLM,",
        "parsing bugs that mishandle tool output, or missing error handling.\n",
        "LOADED FILES (you may ONLY reference these files by name):",
    ]
    for cf in context_files:
        parts.append(f"  - {cf.path.name}")
    parts.append("Do NOT reference any .py file not in this list. If the relevant code")
    parts.append("is not in these files, say so explicitly rather than guessing a filename.\n")

    for cf in context_files:
        parts.append(f"──── {cf.path.name} ({cf.path}) ────")
        # Add line numbers so the LLM can reference real ones
        numbered_lines = []
        for i, line in enumerate(cf.content.splitlines(), 1):
            numbered_lines.append(f"{i:4d} | {line}")
        parts.append("\n".join(numbered_lines))
        parts.append("")

    parts.append("=== END SOURCE CODE ===")
    return "\n".join(parts)


def _auto_detect_context(log_dir: Path) -> list[Path]:
    """Try to find the tools/ directory relative to the log directory."""
    # Try common locations relative to the log dir
    candidates = [
        log_dir.parent / "tools",       # ../tools/ from LLM_Responses
        log_dir.parent / "src",         # ../src/
        log_dir.parent / "agents",      # ../agents/
    ]
    for c in candidates:
        if c.is_dir():
            return [c]

    # Also check for a run.py next to the tools dir
    run_py = log_dir.parent / "run.py"
    if run_py.exists():
        return [run_py]

    return []


def count_context_tokens(
    context_files: list[ContextFile],
) -> tuple[dict[str, int], int, str]:
    """Count tokens for each loaded context file.

    Returns (per_file_counts, total_tokens, method_description).
    per_file_counts maps filename to token count.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        per_file: dict[str, int] = {}
        total = 0
        for cf in context_files:
            count = len(enc.encode(cf.content))
            per_file[cf.path.name] = count
            total += count
        return per_file, total, "tiktoken/cl100k_base"
    except ImportError:
        pass

    # Fallback: estimate from character count
    per_file = {}
    total = 0
    for cf in context_files:
        count = int(len(cf.content) / 3.5)
        per_file[cf.path.name] = count
        total += count
    return per_file, total, "estimated (~3.5 chars/token)"


def _validate_source_citations(
    report: str, context_files: list[ContextFile],
) -> str:
    """Check LLM report for source file references and flag unverified ones.

    Scans for .py filenames cited with line numbers (indicating source code
    citations rather than references to files the agent created).
    """
    if not context_files:
        return ""

    loaded_names = {cf.path.name for cf in context_files}

    # Match patterns like: filename.py (L42), filename.py L42-50, `filename.py` L68
    citation_re = re.compile(r"(\w+\.py)\s*[\(`]?\s*L\d+")
    cited_files = set()
    for match in citation_re.finditer(report):
        cited_files.add(match.group(1))

    if not cited_files:
        return ""

    verified = cited_files & loaded_names
    unverified = cited_files - loaded_names

    if not unverified:
        return ""

    lines = [
        "",
        "─" * 70,
        "  CITATION VALIDATION",
        "─" * 70,
        f"  ⚠ The LLM referenced {len(unverified)} source file(s) NOT in loaded context:",
    ]
    for f in sorted(unverified):
        lines.append(f"    • {f} — not loaded (citation likely hallucinated)")
    if verified:
        lines.append(f"  ✓ Verified references: {', '.join(sorted(verified))}")
    lines.append("")

    return "\n".join(lines)
