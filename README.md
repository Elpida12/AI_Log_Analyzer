<p align="center">AI Log Analyzer</p># 

# Log Analyzer for AI Agent Sessions

Analyze logs from AI coding agent systems using a deterministic pre-filter plus
LLM map-reduce. All LLM-generated claims are cross-checked against a
deterministic fact sheet - contradicted findings are dropped.   \
 <br> Built with a strong emphasis on reducing cognitive load and minimizing hallucinations.

This was originally written for another AI coding agent system that I was working on, but designed so the log
format and pipeline shape are configurable in one file (`config.py`).

## What it does

1. **Parse** each log file into a structured record (reasoning, response,
   tool calls, outputs, exit codes).
2. **Group** entries into sessions - each time the iteration counter resets to
   zero, a new session begins.
3. **Pre-filter** (no LLM) runs deterministic heuristics across all sessions:
   unresolved exits, repeated failing commands, writes without tests,
   suspicious "success" validations, marker/role mismatches, etc.
4. **Map** - the LLM analyzes each problematic session, producing structured
   JSON findings with evidence references.
5. **Verify** - each LLM finding's `evidence_ref`, cited commands, exit codes,
   and file writes are checked against the deterministic fact sheet.
   Contradicted findings get dropped, not included in the final report.
6. **Reduce** - the LLM synthesizes verified per-session findings into a final
   cross-session report.
7. **Interactive** - follow-up Q&A with full context available.

## Quick start

```bash
pip install openai tiktoken
python analyze.py --dir /path/to/your/logs
```

The analyzer talks to any OpenAI-compatible endpoint. Default is
`http://127.0.0.1:8080/v1` (llama-server). Override with `--api-base`.

Useful flags:

```bash
python analyze.py --quick           # pre-filter only, no LLM calls
python analyze.py --session 3       # just one session
python analyze.py -q "why does the reviewer keep failing?"
python analyze.py --no-interactive  # skip interactive Q&A at the end
python analyze.py --no-stream       # wait for full responses instead of streaming
python analyze.py --no-ai-coder-detectors  # If you're using this analyzer with a different agent system, you'll want to use this flag so the analyzer doesn't produce meaningless noise.
```

## Log format expected

Each file is one iteration of an agent session. Filenames must contain
`iter<N>` (e.g. `session1_iter3_coder.txt`) so the analyzer can extract the
iteration number. When iteration resets to 0, a new session starts.

File contents:

```
Timestamp: 2025-01-15 14:30:00
=====================================
=== Reasoning ===
<model's reasoning text>
=== Response ===
<model's visible output - may contain markers like [TASK_COMPLETE]>
=== Tool Output ===
Tool: bash
Args: {'command': 'python3 test.py'}
---
<tool output>
Exit code: 0
```

If your logs use different delimiters, edit `config.py` section 2 (log file
format) - you can swap section headers, tool-output separators, timestamp
format, filename iteration pattern, and exit-code regex without touching
parser code.

## Architecture

```
  log files
      │
      ▼
  ┌──────────┐      ┌────────────┐
  │  parse   │─────▶│ group into │
  └──────────┘      │  sessions  │
                    └─────┬──────┘
                          │
                          ▼
            ┌──────────────────────┐
            │   deterministic      │
            │   pre-filter         │◀── heuristic detectors
            │   (no LLM)           │    (prefilter.py)
            └─────┬────────────────┘
                  │
    ┌─────────────┴─────────────┐
    │                           │
    ▼                           ▼
 clean sessions              problem sessions
 (skip LLM, one-line)        │
                             ▼
                       ┌──────────┐
                       │ fact     │◀── deterministic ground truth
                       │ sheet    │    (fact_sheet.py)
                       └────┬─────┘
                            │
                            ▼
                       ┌──────────┐
                       │ map LLM  │───▶ structured JSON findings
                       └────┬─────┘
                            │
                            ▼
                       ┌──────────┐
                       │ verify   │◀── drops contradicted claims
                       └────┬─────┘    (verification.py)
                            │
                            ▼
                       ┌──────────┐
                       │reduce LLM│───▶ final report
                       └────┬─────┘
                            │
                            ▼
                       ┌──────────┐
                       │interactive│───▶ Q&A
                       └──────────┘
```

The fact sheet + verification loop is the key idea: the LLM can be creative
with interpretation, but it cannot invent commands, exit codes, or error text
that don't appear in the deterministic data. Every finding must carry an
`evidence_ref` pointing at a specific tool call or marker.

## Adapting to your agent system

All customization lives in `config.py`, organized into numbered sections:

| Section | Edit when | What changes |
|---------|-----------|--------------|
| 1. LLM API | You use a different endpoint / model | `API_BASE`, `MODEL`, token budgets |
| 2. Log file format | Your logs use different delimiters | Section headers, separators, timestamp format, iteration regex |
| 3. Pipeline markers | Your agents emit different completion tokens | `MARKERS` set |
| 4. Agent roles | Your pipeline has different stages | `ROLE_*` constants and `ROLE_EXPECTATIONS` |
| 5. Error patterns | Your tool output signals errors differently | `ERROR_PATTERNS` regex list |
| 6. AI_Coder detectors | You're not analyzing logs from my AI_Coder | Set `DETECT_AI_CODER_BUGS = False` (or use `--no-ai-coder-detectors`) |
| 7. Source code context | You want the LLM to read your agent's source | `PRIORITY_FILES`, `CONTEXT_BUDGET` |

For format changes (section 2), that's usually all you need. For more invasive
adaptations (e.g., different tool argument formats than Python dict repr), see
the extractors in `parsing.py` (`_extract_bash_command`, etc.).

## Known AI_Coder-specific heuristics

Two detectors target known issues in the AI_Coder agent system specifically:

- **Stop-sequence truncation** in `write_file` calls - detects unbalanced
  trailing quotes caused by the `\n</tool>` stop sequence consuming a closing
  quote when they share a BPE token.
- **`command` prefix hallucination** - detects the model prepending the word
  `command` before bash commands inside tool calls.

Both are gated behind `config.DETECT_AI_CODER_BUGS` and disabled by
`--no-ai-coder-detectors`. If you're using this analyzer with a different
agent system, disable these so they don't produce meaningless noise.

Generic heuristics (suspicious validations, unresolved exits, writes without
tests, repeated failing commands, marker/role mismatches) stay on - they
apply to any iterative agent system.

## Module layout

| File | Purpose |
|------|---------|
| `analyze.py` | CLI entry point |
| `config.py` | All user-editable configuration |
| `utils.py` | Text manipulation helpers (truncation, compression) |
| `parsing.py` | Log file parser, `Session`/`LogEntry`/`ToolCall` types |
| `prefilter.py` | Deterministic detection heuristics |
| `fact_sheet.py` | Ground-truth extraction + structured map results |
| `context.py` | Source code context loading + citation validation |
| `llm_client.py` | OpenAI-compatible client + streaming + think-tag handling |
| `map_reduce.py` | LLM map/reduce pipeline + interactive mode |
| `verification.py` | Cross-check LLM claims against fact sheets |

## Output

Each run creates a timestamped directory under `analyzer_logs/` containing:

- `PreFilter_Log.txt` - deterministic report (also printed to stdout)
- `Session_<N>.txt` - per-session analysis
- `Final_Analysis_Report.txt` - synthesized cross-session report with
  verification annotations appended

## Requirements

- Python 3.9+
- `openai` Python SDK (for any OpenAI-compatible endpoint)
- `tiktoken` (optional - falls back to char-count estimation if missing)
- A local or remote LLM reachable via an OpenAI-compatible API


