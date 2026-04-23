"""Configuration for the log analyzer.

This is the file you edit to adapt the analyzer to your own agent system.
Sections are ordered by how often users need to change them:

  1. LLM API              — where to reach your local model
  2. Log file format      — delimiters the parser looks for
  3. Pipeline markers     — tokens your agents emit to signal outcomes
  4. Agent roles          — what your pipeline stages are called
  5. Error patterns       — regexes that flag something as "went wrong"
  6. AI_Coder detectors   — heuristics specific to one agent system
  7. Source code context  — files the LLM reads to trace root causes
  8. Internals            — schema, importance levels, command-detection prefixes

Most users only need to touch sections 2, 3, 4, and 6.
"""

from __future__ import annotations

import re
from pathlib import Path


# ── 1. LLM API Configuration ──────────────────────────────────────────────────
# Point at any OpenAI-compatible endpoint: llama-server, vLLM, LM Studio,
# Ollama (with OpenAI compat mode), Anthropic via proxy, etc.

DEFAULT_LOG_DIR = Path(__file__).parent.parent / "LLM_Responses"
API_BASE = "http://127.0.0.1:8080/v1"
API_KEY = "not-needed"
MODEL = "local"  # llama-server ignores model name

# Token budgets — thinking models need extra headroom since max_tokens covers
# both reasoning and output. If your model supports a separate reasoning
# budget (e.g. llama-server --jinja --reasoning-budget N), you can keep
# these lower; otherwise set them high enough for reasoning + answer.
MAP_MAX_TOKENS = 65536        # Per-session analysis
REDUCE_MAX_TOKENS = 65536     # Final synthesis report
INTERACTIVE_MAX_TOKENS = 65536  # Interactive Q&A


# ── 2. Log File Format ────────────────────────────────────────────────────────
# The parser splits each log file on these section markers. If your logs use
# different delimiters, change the strings/regexes here.

# Section headers — each log file is split on these (exact string matches)
SECTION_REASONING = "=== Reasoning ==="
SECTION_RESPONSE = "=== Response ==="
SECTION_TOOL_OUTPUT = "=== Tool Output ==="
HEADER_DIVIDER = "====================================="

# Within each "=== Tool Output ===" section, the args/output are separated by
# a line containing just "---" (preferred: "\n---\n" so it doesn't collide
# with "---" that may appear inside the args dict repr).
TOOL_OUTPUT_SEPARATOR = "\n---\n"
TOOL_OUTPUT_SEPARATOR_FALLBACK = "---"

# Per-tool-call field extractors
TIMESTAMP_RE = re.compile(r"Timestamp:\s*(.+)")
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
ITERATION_FILENAME_RE = re.compile(r"iter(\d+)")
TOOL_NAME_RE = re.compile(r"Tool:\s*(.+)")
TOOL_ARGS_RE = re.compile(r"Args:\s*(.+)")

# Exit code extraction from tool output — matches "Exit code: N"
EXIT_CODE_RE = re.compile(r"Exit code:\s*(\d+)")


# ── 3. Pipeline Markers ───────────────────────────────────────────────────────
# Markers are tokens your agents emit in their text responses to signal
# outcomes. When the analyzer sees one of these in a response, it records
# it as the session's outcome.

MARKERS = {
    "[TASK_COMPLETE]",
    "[REVIEW_PASS]",
    "[REVIEW_FAIL]",
    "[REVIEW_BLOCKED]",
    "[INTEGRATION_PASS]",
    "[INTEGRATION_FAIL]",
    "[INTEGRATION_BLOCKED]",
    "[OBSERVATIONS]",
}


# ── 4. Agent Roles ────────────────────────────────────────────────────────────
# Each session has a role in the pipeline. Roles drive what counts as an
# error vs. an expected finding (e.g. reviewer sessions EXPECT non-zero
# exit codes — that's the reviewer finding bugs).
#
# To adapt to your own pipeline: rename or add role constants, and update
# ROLE_EXPECTATIONS with matching entries.

ROLE_DELEGATOR = "delegator"
ROLE_CODER = "coder"
ROLE_REVIEWER = "reviewer"
ROLE_INTEGRATION = "integration_reviewer"
ROLE_UNKNOWN = "unknown"

ROLE_EXPECTATIONS = {
    ROLE_DELEGATOR: {
        "description": "Plans tasks and outputs a JSON task array. Does NOT write code or call tools.",
        "normal_iterations": "1",
        "expected_markers": "none (delegator does not use markers)",
        "error_guidance": (
            "A delegator session with no code output and no tool calls is NORMAL. "
            "Errors here mean the JSON task plan was malformed or unparseable."
        ),
    },
    ROLE_CODER: {
        "description": "Writes code, runs commands, tests its work. Produces [TASK_COMPLETE] when done.",
        "normal_iterations": "3-15",
        "expected_markers": "[TASK_COMPLETE] or [OBSERVATIONS]",
        "error_guidance": (
            "Coder errors from testing (exit code 1 from python3 test_*.py) are EXPECTED during "
            "development — the coder writes code, tests it, fixes it. Only flag errors that the "
            "coder failed to resolve by the end of the session, or loops where no progress is made."
        ),
    },
    ROLE_REVIEWER: {
        "description": "Reviews code quality, runs tests to find bugs. Produces [REVIEW_PASS] or [REVIEW_FAIL].",
        "normal_iterations": "1-3",
        "expected_markers": "[REVIEW_PASS], [REVIEW_FAIL], or [REVIEW_BLOCKED]",
        "error_guidance": (
            "Reviewer errors are often INTENTIONAL — the reviewer runs edge-case tests "
            "(invalid inputs, permission errors) specifically to see if they fail properly. "
            "A non-zero exit code in a reviewer session means 'the reviewer found a bug', "
            "not 'the reviewer made a mistake'. Only flag reviewer errors if the reviewer "
            "itself malfunctioned (e.g., wrong tool syntax, hallucinated file paths)."
        ),
    },
    ROLE_INTEGRATION: {
        "description": "Validates cross-module integration. Produces [INTEGRATION_PASS] or [INTEGRATION_FAIL].",
        "normal_iterations": "1-3",
        "expected_markers": "[INTEGRATION_PASS], [INTEGRATION_FAIL], or [INTEGRATION_BLOCKED]",
        "error_guidance": (
            "Integration reviewer errors from running the full program are EXPECTED — "
            "it's testing whether modules work together. Non-zero exit codes mean "
            "'integration test found an issue', not 'the reviewer broke'. "
            "Only flag if the reviewer itself malfunctioned."
        ),
    },
    ROLE_UNKNOWN: {
        "description": "Could not determine this session's role in the pipeline.",
        "normal_iterations": "varies",
        "expected_markers": "unknown",
        "error_guidance": "Evaluate errors at face value.",
    },
}


# ── 5. Error Patterns ─────────────────────────────────────────────────────────
# Regexes that signal something worth flagging. Matched against tool outputs
# (and response text, for non-delegator sessions).

ERROR_PATTERNS = [
    re.compile(r"(?i)error[:\s]"),
    re.compile(r"(?i)traceback \(most recent call last\)"),
    re.compile(r"(?i)exception[:\s]"),
    re.compile(r"(?i)failed"),
    re.compile(r"(?i)syntax\s*error"),
    re.compile(r"Exit code:\s*[1-9]"),
    re.compile(r"\[REVIEW_FAIL\]"),
    re.compile(r"\[REVIEW_BLOCKED\]"),
    re.compile(r"\[INTEGRATION_FAIL\]"),
    re.compile(r"\[INTEGRATION_BLOCKED\]"),
]


# ── 6. AI_Coder-Specific Detectors ────────────────────────────────────────────
# These heuristics target known issues in the AI_Coder agent system that this
# analyzer was originally written for. They look for specific hallucination
# and stop-sequence-truncation patterns that other agent systems won't share.
#
# Set DETECT_AI_CODER_BUGS = False (or pass --no-ai-coder-detectors on the
# command line) if you're adapting this analyzer to a different agent system.

DETECT_AI_CODER_BUGS = True


# ── 7. Source Code Context ────────────────────────────────────────────────────
# When analyzing agent misbehavior, the LLM can trace root causes by reading
# your agent's source code. Priority = ordering; the analyzer gives a bigger
# share of the budget to files earlier in this list.

PRIORITY_FILES = [
    "prompts.py",          # System prompts define all agent behavior
    "config.py",           # Configuration drives iteration limits, truncation, etc.
    "orchestrator.py",     # How agents are deployed and how review cycles work
    "xml_parser.py",       # Tool call parsing — malformed tool calls trace here
    "agent.py",            # Core agent loop — tool execution, retries, streaming
    "bash.py",             # Sandbox, command prefix stripping, quote fixing
    "write_file.py",       # File creation logic
    "reasoning_parser.py", # Reasoning tag extraction
    "task_manager.py",     # Task/issue state management
]

# Total character budget for all context files combined. Set to 0 to disable
# truncation and include every file in full.
CONTEXT_BUDGET = 0
MIN_PER_FILE = 1500  # Minimum chars per file (so even low-priority files are useful)


# ── 8. Internals ──────────────────────────────────────────────────────────────
# Rarely need changes — these drive the structured-output schema, importance
# ranking, and verification helpers.

IMPORTANCE_HIGH = "high"
IMPORTANCE_MEDIUM = "medium"
IMPORTANCE_LOW = "low"

VALID_FINDING_TYPES = {
    "execution_error",
    "review_finding",
    "reviewer_error",
    "suspicious_validation",
    "untested_completion",
    "infrastructure_bug",
    "behavioral_issue",
    "unresolved_error",
    "repeated_failure",
    "write_without_test",
}

MAP_JSON_SCHEMA = """\
{
  "task_summary": "<1 sentence>",
  "findings": [
    {
      "type": "execution_error | review_finding | reviewer_error | suspicious_validation | untested_completion | infrastructure_bug | behavioral_issue | unresolved_error | repeated_failure | write_without_test",
      "iteration": <int>,
      "tool": "<tool name>",
      "command_exact": "<exact command string from fact sheet>",
      "exit_code": <int or null>,
      "evidence": "<exact error text or description from logs>",
      "evidence_ref": "iter=<N>/tool=<name>/exit=<code>",
      "confidence": <0.0 to 1.0>,
      "supported_by_fact_sheet": <true|false>
    }
  ],
  "unknowns": ["<observations without direct evidence>"],
  "final_assessment": "<1-2 sentences>",
  "severity": "CRITICAL | WARNING | INFO"
}"""

# Matches actual Python exception lines (e.g. "SyntaxError: unterminated ...").
# Used by error_snippet extraction to prefer the real error over generic matches
# like "Traceback ..." or "Exit code: 1".
_EXCEPTION_LINE_RE = re.compile(
    r"^\s*\w*(?:Error|Exception|Warning):\s", re.IGNORECASE
)

# Markdown formatting inside a backtick span means it's prose, not an error
_MARKDOWN_NOISE_RE = re.compile(r'\*\*|^#{1,4}\s|^\s*-\s')

# Words that indicate a command is claiming success (used by bogus-validation
# detection). These are matched inside print()/echo strings, not globally.
_BOGUS_SUCCESS_RE = re.compile(
    r"(?i)\b(is\s+valid|valid(?:ated)?|passed|pass(?:es)?|works?"
    r"|working|correct|success(?:ful(?:ly)?)?|verified|confirmed"
    r"|is\s+good|all\s+good|looks?\s+good)\b"
)

# Prefixes that identify a backtick-quoted string as a bash command
_CMD_PREFIXES = (
    "python", "pip", "grep", "cat", "ls", "find", "echo", "curl",
    "wget", "mkdir", "touch", "cd", "bash", "chmod", "cp", "mv",
    "rm", "sed", "awk", "head", "tail", "sort", "wc", "diff",
    "pytest", "# ",
)
