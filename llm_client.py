"""OpenAI-compatible LLM client, streaming, and content extraction.

Handles thinking models (Qwen3-Thinking, DeepSeek-R1) that may embed
<think>...</think> blocks or separate reasoning from output.
"""

from __future__ import annotations

import re
import sys

from openai import OpenAI

from config import API_BASE, API_KEY, MODEL


def _make_client() -> OpenAI:
    return OpenAI(base_url=API_BASE, api_key=API_KEY, timeout=3500.0)


def _strip_think_tags(text: str) -> str:
    """Strip ``<think>...</think>`` reasoning blocks from model output.

    Handles complete blocks, unclosed tags (model cut off mid-reasoning),
    and multiple/nested blocks.
    """
    if "<think>" not in text:
        return text
    # Remove complete <think>...</think> blocks
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Handle unclosed <think> (truncated mid-reasoning) — drop everything after it
    if "<think>" in stripped:
        stripped = stripped[:stripped.index("<think>")]
    return stripped.strip()


def _extract_content(response) -> str:
    """Extract content from an LLM response, handling thinking models.

    Thinking models (e.g. Qwen3-Thinking, DeepSeek-R1) may:
      1. Embed <think>...</think> tags directly in 'content'
      2. Separate reasoning into 'reasoning_content', leaving 'content' empty
      3. Exhaust max_tokens on reasoning, producing no actual output

    This function strips reasoning tags, detects truncation, and avoids
    returning raw chain-of-thought as the answer.
    """
    choice = response.choices[0]
    msg = choice.message
    finish_reason = choice.finish_reason  # "stop" or "length"

    raw_content = msg.content or ""
    reasoning = getattr(msg, "model_extra", {}).get("reasoning_content", "")

    # Strip <think>...</think> blocks from content
    content = _strip_think_tags(raw_content)

    # Case 1: Content has actual text after stripping think tags
    if content.strip():
        if finish_reason == "length":
            content += (
                "\n\n[WARNING: Response truncated — hit max_tokens limit. "
                "Output may be incomplete.]"
            )
        return content

    # Content is empty — either the model separated reasoning properly,
    # or it exhausted tokens on thinking before producing output.

    # Case 2: Token budget exhausted (reasoning consumed all tokens)
    if finish_reason == "length":
        reasoning_len = len(reasoning) if reasoning else len(raw_content)
        return (
            "[ERROR: LLM exhausted its token budget on reasoning and produced "
            "no actual output. Increase max_tokens or configure the server to "
            "allocate reasoning tokens separately "
            "(e.g. llama-server --jinja --reasoning-budget N).]\n"
            f"[Reasoning length: {reasoning_len:,} chars]"
        )

    # Case 3: Model completed normally (finish_reason == "stop") but content
    # is empty — some servers put the real output in reasoning_content
    if reasoning.strip():
        return _strip_think_tags(reasoning)

    return "(empty response)"


def _hold_partial(text: str, tag: str) -> int:
    """Return count of trailing chars in *text* that form a prefix of *tag*."""
    for length in range(min(len(tag) - 1, len(text)), 0, -1):
        if tag.startswith(text[-length:]):
            return length
    return 0


def _stream_response(
    client: OpenAI,
    messages: list[dict],
    max_tokens: int,
    temperature: float = 0.3,
    show_thinking: bool = True,
) -> str:
    """Stream an LLM response with real-time display.

    Detects ``<think>...</think>`` blocks and renders thinking text with dim
    ANSI styling so the user can watch the model reason live.  Returns the
    cleaned content (think blocks stripped) for downstream processing — the
    same value ``_extract_content`` would return for a non-streamed call.
    """
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )

    # ANSI escape codes
    DIM_ITALIC = "\033[2;3m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    full_text = ""
    buffer = ""
    in_think = False
    finish_reason = None

    for chunk in stream:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if choice.finish_reason:
            finish_reason = choice.finish_reason
        delta_content = choice.delta.content if choice.delta else None
        if not delta_content:
            continue

        full_text += delta_content
        buffer += delta_content

        # Process buffer, detecting <think> / </think> boundaries
        changed = True
        while changed and buffer:
            changed = False
            if not in_think:
                tag_pos = buffer.find("<think>")
                if tag_pos != -1:
                    sys.stdout.write(buffer[:tag_pos])
                    if show_thinking:
                        sys.stdout.write(
                            f"\n{DIM}━━━ thinking ━━━{RESET}\n{DIM_ITALIC}"
                        )
                    in_think = True
                    buffer = buffer[tag_pos + 7:]
                    changed = True
                    continue
                held = _hold_partial(buffer, "<think>")
                if held:
                    sys.stdout.write(buffer[:-held])
                    buffer = buffer[-held:]
                else:
                    sys.stdout.write(buffer)
                    buffer = ""
            else:
                tag_pos = buffer.find("</think>")
                if tag_pos != -1:
                    if show_thinking:
                        sys.stdout.write(buffer[:tag_pos])
                        sys.stdout.write(
                            f"{RESET}\n{DIM}━━━ end thinking ━━━{RESET}\n\n"
                        )
                    in_think = False
                    buffer = buffer[tag_pos + 8:]
                    changed = True
                    continue
                held = _hold_partial(buffer, "</think>")
                if held:
                    if show_thinking:
                        sys.stdout.write(buffer[:-held])
                    buffer = buffer[-held:]
                else:
                    if show_thinking:
                        sys.stdout.write(buffer)
                    buffer = ""

        sys.stdout.flush()

    # Flush remaining buffer
    if buffer:
        if in_think:
            if show_thinking:
                sys.stdout.write(buffer)
        else:
            sys.stdout.write(buffer)
    if in_think and show_thinking:
        sys.stdout.write(
            f"{RESET}\n{DIM}━━━ end thinking (truncated) ━━━{RESET}\n"
        )
    sys.stdout.write("\n")
    sys.stdout.flush()

    # Return cleaned content — mirrors _extract_content() logic
    content = _strip_think_tags(full_text)
    if content.strip():
        if finish_reason == "length":
            content += (
                "\n\n[WARNING: Response truncated — hit max_tokens limit. "
                "Output may be incomplete.]"
            )
        return content

    if finish_reason == "length":
        return (
            "[ERROR: LLM exhausted its token budget on reasoning and produced "
            "no actual output. Increase max_tokens or configure the server to "
            "allocate reasoning tokens separately "
            "(e.g. llama-server --jinja --reasoning-budget N).]\n"
            f"[Reasoning length: {len(full_text):,} chars]"
        )

    return full_text.strip() if full_text.strip() else "(empty response)"


def _is_budget_error(text: str) -> bool:
    """Check if an LLM response is a token budget exhaustion error."""
    return text.startswith("[ERROR: LLM exhausted")
