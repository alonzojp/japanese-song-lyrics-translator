"""
LLM-based transcription correction via Groq API.

Sends Whisper's raw segment text to an LLM to fix common transcription errors:
  - Wrong kanji (homophones with different characters)
  - Missing/incorrect particles (は, が, を, に, etc.)
  - Merged or split words
  - Incorrect kana readings

Timing data (startTime, endTime, words) is preserved from the original segments.
Only runs when GROQ_API_KEY is set.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama3-8b-8192"

_PROMPT_TEMPLATE = """\
You are correcting a Japanese song lyrics transcription. The text below was automatically transcribed from audio by an AI speech model and may contain errors specific to sung Japanese.

Common errors to fix:
- Wrong kanji (homophones that sound identical but use different characters, e.g. 晴れ vs 春)
- Missing or wrong particles (は, が, を, に, へ, で, も, etc.)
- Incorrect kana (ぶ vs む, ず vs づ, etc.)
- Words incorrectly merged or split

Rules:
- Output EXACTLY the same number of lines as the input — one output line per input line
- Do NOT add, remove, or reorder lines
- Do NOT rewrite lines that look correct — minimal changes only
- Do NOT add any explanation, headers, or extra text
- Output ONLY the corrected lyrics lines, nothing else

Input ({n} lines):
{text}"""


def correct_segments(
    segments: list[dict],
    job_log=None,
) -> list[dict]:
    """
    Attempt to correct segment text using Groq LLM.
    Returns original segments unchanged if correction fails or key is missing.
    """
    if not GROQ_API_KEY or not segments:
        return segments

    def log(msg: str) -> None:
        if job_log:
            job_log.info(msg, stage="transcribe")
        logger.info(msg)

    def warn(msg: str) -> None:
        if job_log:
            job_log.warning(msg, stage="transcribe")
        logger.warning(msg)

    lines = [seg.get("text", "").strip() for seg in segments]
    combined = "\n".join(lines)

    prompt = _PROMPT_TEMPLATE.format(n=len(lines), text=combined)

    try:
        resp = httpx.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model":       GROQ_MODEL,
                "messages":    [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens":  2048,
            },
            timeout=30.0,
        )
        resp.raise_for_status()

        corrected_text  = resp.json()["choices"][0]["message"]["content"].strip()
        corrected_lines = corrected_text.splitlines()

        # Only apply if line count matches exactly
        if len(corrected_lines) != len(segments):
            warn(
                f"Groq returned {len(corrected_lines)} lines for {len(segments)} segments "
                "— skipping correction to avoid misalignment"
            )
            return segments

        result = []
        changes = 0
        for i, (seg, new_text) in enumerate(zip(segments, corrected_lines)):
            new_text = new_text.strip()
            old_text = seg.get("text", "").strip()
            if new_text != old_text:
                changes += 1
                log(f"[Groq] seg{i:02d} changed: '{old_text}' → '{new_text}'")
            result.append({**seg, "text": new_text})

        log(f"[Groq] corrected {changes}/{len(segments)} segments")
        return result

    except Exception as exc:
        warn(f"Groq correction failed (non-fatal): {exc}")
        return segments
