"""
AI Transcript Refiner & Post-Processing Layer using FreeToken / Qwen LLM.
Cleans up disfluencies, punctuation, and casing while preserving verbatim meaning and speaker turns.
"""

import json
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import httpx
from dotenv import load_dotenv

from .mom import _get_llm_config, _parse_sse_chunk, _extract_turn_info

load_dotenv(".secrets")
load_dotenv(".env")

DEFAULT_REFINER_PROMPT = """# Role
You are an expert Speech-to-Text Transcript Editor and Linguist. Your task is to polish and refine raw automatic speech recognition (ASR) transcripts into clean, readable, professional text.

# Objectives & Rules
1. CRITICAL LANGUAGE RULE (DO NOT EVER TRANSLATE): You MUST refine the text in the EXACT SAME LANGUAGE spoken by each speaker (e.g. Bahasa Indonesia for Indonesian speakers, English for English speakers). NEVER translate or convert the dialogue into English or any other language.
2. Fix capitalization, punctuation, grammar, and sentence boundaries.
3. Remove filler words and speech disfluencies (e.g., "um", "uh", "you know", "like", stuttering repetitions) only when they do not add substantive meaning.
4. STRICT FIDELITY: Never alter factual meaning, numbers, technical terms, acronyms, product names, or speaker intent.
5. Preserve the exact speaker turns in format:
Speaker Name: Refined text...

# Output Format
Output only the refined transcript directly with each speaker turn separated by a blank line. Do not include introductory notes or commentary.
"""


def format_transcript_for_refinement(segments: List[Union[Dict[str, Any], Any]]) -> str:
    """Format diarized segments into clean dialogue script for refinement."""
    turns: List[str] = []
    current_speaker: Optional[str] = None
    current_texts: List[str] = []

    for seg in segments:
        spk, txt = _extract_turn_info(seg)
        if not txt:
            continue
        if spk != current_speaker:
            if current_speaker and current_texts:
                turns.append(f"{current_speaker}: {' '.join(current_texts)}")
            current_speaker = spk
            current_texts = [txt]
        else:
            current_texts.append(txt)

    if current_speaker and current_texts:
        turns.append(f"{current_speaker}: {' '.join(current_texts)}")

    return "\n\n".join(turns)


def _build_refine_payload(
    cfg: Dict[str, str],
    raw_transcript: str,
    system_prompt: Optional[str],
    temperature: float,
    stream: bool,
) -> Dict[str, Any]:
    """Construct chat completions request payload for transcript refinement."""
    return {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": system_prompt or DEFAULT_REFINER_PROMPT},
            {
                "role": "user",
                "content": f"Please refine the following raw meeting transcript:\n\n{raw_transcript}",
            },
        ],
        "temperature": temperature,
        "stream": stream,
    }


def refine_transcript_sync(
    segments: List[Union[Dict[str, Any], Any]],
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.1,
    timeout: float = 120.0,
) -> str:
    """Refine transcript synchronously using FreeToken / OpenAI-compatible endpoint."""
    cfg = _get_llm_config(base_url, api_key, model)
    raw_transcript = format_transcript_for_refinement(segments)
    if not raw_transcript:
        return "*(Empty transcript provided; unable to refine)*"

    payload = _build_refine_payload(cfg, raw_transcript, system_prompt, temperature, False)
    headers = {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}
    url = f"{cfg['base_url']}/chat/completions"

    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()


async def refine_transcript_stream(
    segments: List[Union[Dict[str, Any], Any]],
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.1,
    timeout: float = 180.0,
) -> AsyncGenerator[str, None]:
    """Refine transcript asynchronously yielding streaming text chunks."""
    cfg = _get_llm_config(base_url, api_key, model)
    raw_transcript = format_transcript_for_refinement(segments)
    if not raw_transcript:
        yield "*(Empty transcript provided; unable to refine)*"
        return

    payload = _build_refine_payload(cfg, raw_transcript, system_prompt, temperature, True)
    headers = {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}
    url = f"{cfg['base_url']}/chat/completions"

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                delta = _parse_sse_chunk(line)
                if delta == "[DONE]":
                    break
                if delta:
                    yield delta
