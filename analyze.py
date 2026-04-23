#!/usr/bin/env python3
"""
Log Analyzer for AI_Coder agent sessions.

Uses a map-reduce strategy to efficiently analyze LLM response logs:
  1. Parse all log files into structured records
  2. Group by task session (iteration resets signal new tasks)
  3. Pre-filter: extract errors, markers, anomalies (no LLM needed)
  4. Map: LLM summarizes each session independently
  5. Reduce: LLM correlates findings across all sessions
  6. Interactive: ask follow-up questions with full context

Usage:
    python analyze.py                         # Analyze all logs
    python analyze.py --dir /path/to/logs     # Custom log directory
    python analyze.py --session 3             # Analyze only session 3
    python analyze.py --quick                 # Pre-filter only (no LLM)
    python analyze.py --question "Why does the reviewer keep failing?"
    python analyze.py --context ../tools/     # Include source code for deeper analysis
    python analyze.py --no-context            # Disable auto-detected source context
    python analyze.py --no-stream             # Disable streaming (wait for complete responses)
    python analyze.py --no-thinking           # Stream output but hide LLM reasoning
    python analyze.py --no-ai-coder-detectors # Disable AI_Coder-specific heuristics
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from openai import OpenAI

import config
from config import API_BASE, API_KEY, DEFAULT_LOG_DIR, PRIORITY_FILES
from context import (
    ContextFile,
    _auto_detect_context,
    _validate_source_citations,
    count_context_tokens,
    format_context_for_prompt,
    load_context_files,
)
from fact_sheet import SessionFactSheet, build_fact_sheet
from map_reduce import (
    _build_session_digest,
    interactive_mode,
    map_analyze_session,
    reduce_synthesize,
)
from parsing import count_log_tokens, group_into_sessions, load_all_logs
from prefilter import (
    compute_prefilter_findings,
    format_findings_for_prompt,
    prefilter_report,
)
from verification import verify_final_report, verify_map_result, verify_session_analysis


def main():
    parser = argparse.ArgumentParser(description="Analyze AI_Coder agent logs with LLM-powered map-reduce.")
    parser.add_argument("--dir", type=Path, default=DEFAULT_LOG_DIR,
                        help="Path to LLM_Responses directory")
    parser.add_argument("--session", type=int, default=None,
                        help="Analyze only a specific session number")
    parser.add_argument("--quick", action="store_true",
                        help="Pre-filter only — no LLM calls")
    parser.add_argument("--question", "-q", type=str, default=None,
                        help="Specific question to focus the analysis on")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Skip interactive follow-up mode")
    parser.add_argument("--context", type=Path, nargs="+", default=None,
                        help="Source code paths (files or dirs) to include as context for deeper analysis. "
                             "Auto-detected from ../tools/ if not specified.")
    parser.add_argument("--no-context", action="store_true",
                        help="Disable auto-detection of source code context")
    parser.add_argument("--api-base", type=str, default=API_BASE,
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--no-stream", action="store_true",
                        help="Disable streaming — wait for complete responses (original behavior)")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Hide the LLM's thinking/reasoning blocks during streaming")
    parser.add_argument("--no-ai-coder-detectors", action="store_true",
                        help="Disable heuristics specific to the AI_Coder agent system "
                             "(stop-sequence truncation, 'command' prefix hallucination). "
                             "Use this when analyzing logs from other agent systems.")
    args = parser.parse_args()

    # Apply runtime config overrides
    if args.no_ai_coder_detectors:
        config.DETECT_AI_CODER_BUGS = False

    # Create logging directory for this run
    log_output_dir = Path(__file__).parent / "analyzer_logs" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logging analyzer outputs to: {log_output_dir}\n")

    # 1. Parse
    print(f"Loading logs from {args.dir}...")
    entries = load_all_logs(args.dir)
    print(f"Parsed {len(entries)} log entries.")

    # Count total tokens across all log files
    token_count, count_method = count_log_tokens(args.dir)
    print(f"Total log file tokens: {token_count:,} ({count_method})\n")

    # 2. Group into sessions
    sessions = group_into_sessions(entries)
    print(f"Grouped into {len(sessions)} sessions.\n")

    # Filter to specific session if requested
    if args.session is not None:
        sessions = [s for s in sessions if s.session_id == args.session]
        if not sessions:
            print(f"Session {args.session} not found.")
            sys.exit(1)

    # 3. Pre-filter report (always runs — no LLM needed)
    findings = compute_prefilter_findings(sessions)
    report = prefilter_report(sessions, findings)
    print(report)
    (log_output_dir / "PreFilter_Log.txt").write_text(report)

    if args.quick:
        return

    # Format findings for LLM injection
    findings_block = format_findings_for_prompt(findings)

    # 3b. Load source code context
    context_block = ""
    loaded_context_files: list[ContextFile] = []
    if not args.no_context:
        if args.context:
            context_paths = args.context
        else:
            context_paths = _auto_detect_context(args.dir)

        if context_paths:
            loaded_context_files = load_context_files(context_paths)
            if loaded_context_files:
                context_block = format_context_for_prompt(loaded_context_files)
                ctx_tokens, ctx_total, ctx_method = count_context_tokens(loaded_context_files)
                print(f"Loaded {len(loaded_context_files)} source files as context "
                      f"({len(context_block):,} chars, {ctx_total:,} tokens [{ctx_method}]):")
                for cf in loaded_context_files:
                    tag = f"  [pri {cf.priority}]" if cf.priority < len(PRIORITY_FILES) else ""
                    ftokens = ctx_tokens.get(cf.path.name, 0)
                    print(f"  {cf.path.name} ({len(cf.content):,} chars, {ftokens:,} tokens){tag}")
                print()
            else:
                print(f"Warning: --context paths resolved to no Python files: "
                      f"{[str(p) for p in context_paths]}\n")
        else:
            print("No source context found (use --context to specify paths).\n")

    # 4. Build fact sheets for all sessions (deterministic)
    fact_sheets: dict[int, SessionFactSheet] = {}
    for session in sessions:
        fact_sheets[session.session_id] = build_fact_sheet(session)

    # 5. Map phase — analyze each session
    client = OpenAI(base_url=args.api_base, api_key=API_KEY, timeout=3500.0)
    project_root = str(args.dir.resolve().parent).rstrip("/")
    clean_ids = set(findings.clean_session_ids)

    stream_enabled = not args.no_stream
    show_thinking = not args.no_thinking

    print("Starting LLM analysis (map phase)...")
    session_analyses = []
    map_results: dict = {}  # session_id -> structured result
    all_annotations: list[str] = []
    verification_totals: dict[str, int] = {
        "verified": 0, "contradicted": 0, "unverifiable": 0,
    }
    for session in sessions:
        if session.session_id in clean_ids:
            # Clean session — skip LLM call, use one-line summary
            summary = _build_session_digest(
                session, project_root=project_root, is_clean=True,
            )
            session_analyses.append(
                f"## Session {session.session_id} [CLEAN — no LLM analysis needed]\n"
                f"{summary}\n"
            )
            print(f"  Session {session.session_id} [{session.role}]: clean (skipped)")
            (log_output_dir / f"Session_{session.session_id}.txt").write_text(
                f"## Session {session.session_id} [CLEAN — no LLM analysis needed]\n{summary}\n"
            )
            continue
        fs = fact_sheets[session.session_id]
        analysis, map_result = map_analyze_session(
            client, session, project_root=project_root, fact_sheet=fs,
            prefilter_findings=findings,
            stream=stream_enabled, show_thinking=show_thinking,
        )

        # Verify and filter the LLM's claims against deterministic facts
        if map_result is not None:
            map_result, vstats, vannotations = verify_map_result(map_result, fs)
            map_results[session.session_id] = map_result
            # Regenerate prose from the filtered (verified) result
            analysis = map_result.to_prose(session.session_id) + "\n"
        else:
            # Prose fallback — use legacy verification
            analysis, vstats, vannotations = verify_session_analysis(analysis, fs)
        for k in vstats:
            verification_totals[k] += vstats[k]
        if vannotations:
            all_annotations.extend(vannotations)

        session_analyses.append(analysis)
        (log_output_dir / f"Session_{session.session_id}.txt").write_text(analysis)

    # Print verification summary
    total_checks = sum(verification_totals.values())
    if total_checks:
        print(f"\nClaim verification: {verification_totals['verified']} verified, "
              f"{verification_totals['contradicted']} contradicted, "
              f"{verification_totals['unverifiable']} unverifiable "
              f"({total_checks} claims checked)")
        if all_annotations:
            print("\n  Per-session verification details:")
            for a in all_annotations:
                print(f"    {a}")

    # 6. Reduce phase — synthesize (with source context + pre-filter findings)
    final_report = reduce_synthesize(
        client, session_analyses,
        question=args.question,
        context_block=context_block,
        findings_block=findings_block,
        stream=stream_enabled,
        show_thinking=show_thinking,
    )
    if not stream_enabled:
        print("\n" + "="*70)
        print("  FINAL ANALYSIS REPORT")
        print("="*70 + "\n")
        print(final_report)

    # 6b. Verify final report claims against all fact sheets
    claim_validation = verify_final_report(final_report, fact_sheets, findings)
    if claim_validation:
        print(claim_validation)

    # 6c. Validate source citations if context was loaded
    source_validation = ""
    if loaded_context_files:
        source_validation = _validate_source_citations(final_report, loaded_context_files)
        if source_validation:
            print(source_validation)

    # Write final report log (includes verification and source citation results)
    final_log_parts = [final_report]
    if claim_validation:
        final_log_parts.append(claim_validation)
    if source_validation:
        final_log_parts.append(source_validation)
    (log_output_dir / "Final_Analysis_Report.txt").write_text("\n".join(final_log_parts))
    print(f"\nAnalyzer logs saved to: {log_output_dir}")

    # 7. Interactive mode (with source context + findings)
    if not args.no_interactive:
        interactive_mode(
            client, sessions, session_analyses,
            context_block=context_block,
            findings_block=findings_block,
            stream=stream_enabled,
            show_thinking=show_thinking,
        )


if __name__ == "__main__":
    main()
