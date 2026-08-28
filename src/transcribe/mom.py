"""
Minutes of Meeting (MOM) Generator using FreeToken / OpenAI-Compatible Qwen LLMs.
Analyzes multi-speaker transcripts and generates structured executive MOM documents.
"""

import json
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import httpx
from dotenv import load_dotenv

load_dotenv(".secrets")
load_dotenv(".env")

DEFAULT_MOM_PROMPT = """# Role
You are an expert Executive Assistant and Project Management Officer (PMO). Your task is to analyze the raw meeting transcript provided earlier in this chat history and synthesize it into a highly structured, professional, and actionable Minutes of Meeting (MOM).

# Objectives
1. Reference the complete transcript provided in the previous message(s) of this conversation.
2. Filter out casual banter, filler words, and repetitive loops, focusing strictly on substance.
3. Maintain the professional context and technical accuracy of the discussion.
4. Clearly distinguish between facts, decisions, and proposed ideas.

# Output Format
Please format the output using clean Markdown according to the following structure:

## 1. Meeting Overview
* **Topic/Project:** [Infer from context if not explicitly stated]
* **Date/Time:** [Extract if mentioned, otherwise leave as "As per transcript"]
* **Key Attendees:** [List identified speakers and their primary focus areas based on the discussion]

## 2. Executive Summary
Provide a concise, 3-4 sentence paragraph summarizing the main objective of the meeting and the primary outcome.

## 3. Detailed Discussion Points
Break down the discussion into logical thematic sections or agenda items. For each section, provide:
* **[Topic/Agenda Title]**
    * **Context/Problem:** What issues or updates were brought to the table?
    * **Key Perspectives:** Summarize the viewpoints, technical points, or arguments raised by the participants. 
    * **Resolution/Outcome:** What was the conclusion or current standing of this topic?

## 4. Key Decisions Made
List all concrete decisions made during the meeting in a bulleted list. (e.g., "Approved the layout for X", "Decided to pivot from Y to Z"). If no final decisions were reached on a topic, explicitly state "None finalized; pending further data."

## 5. Action Items & Accountability
Present this in a Markdown table with the following columns: Task, Owner, and Deadline. 
* *Note: If a task is assigned but no specific owner is named, assign it to "Team [Contextual Department]" or "Unassigned". If no deadline is stated, mark it as "TBD".*

| Task | Owner | Deadline |
| :--- | :--- | :--- |
| [Insert Task] | [Insert Owner] | [Insert Deadline] |

## 6. Next Steps & Next Meeting
* State any immediate next milestones.
* Note the tentative date/topic for the next follow-up alignment, if mentioned.

# Guidelines & Constraints
* **CRITICAL LANGUAGE RULE (DO NOT EVER TRANSLATE):** You MUST write all summaries, descriptions, discussion points, decisions, and action items in the EXACT SAME LANGUAGE spoken by the speakers in the transcript (e.g. Bahasa Indonesia if the participants spoke Indonesian, English if English, Japanese if Japanese). Retain fixed standard English section headers (## 1. Meeting Overview, ## 2. Executive Summary, etc.), but NEVER translate the spoken conversation into English or any other language.
* **Tone:** Professional, objective, and analytical.
* **Ambiguity:** If a critical point or action item is discussed but left vague or unresolved, add a section called "### Items Requiring Clarification" to flag them.
* **Accuracy:** Preserve technical terms, acronyms, and specific product names exactly as spoken. Do not assume or hallucinate dates or metrics not present in the text.
"""


def _extract_turn_info(seg: Union[Dict[str, Any], Any]) -> Tuple[str, str]:
    """Extract speaker label and clean text from segment dictionary or object."""
    speaker = seg.get("speaker") if isinstance(seg, dict) else getattr(seg, "speaker", None)
    text = seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")
    return speaker or "Speaker", (text or "").strip()


def format_transcript_for_mom(segments: List[Union[Dict[str, Any], Any]]) -> str:
    """Format diarized transcript segments into clean Speaker: Text turns."""
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


def _get_llm_config(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, str]:
    """Resolve LLM endpoint configurations with fallback defaults and live model auto-discovery."""
    resolved_base = (base_url or os.getenv("LLM_BASE_URL", "http://127.0.0.1:4050/v1")).rstrip("/")
    resolved_key = api_key or os.getenv("LLM_API_KEY", "freetoken-local")
    resolved_model = model or os.getenv("LLM_MODEL", "Qwen3.6-35B-A3B-FP8")

    try:
        with httpx.Client(timeout=1.5) as client:
            resp = client.get(f"{resolved_base}/models", headers={"Authorization": f"Bearer {resolved_key}"})
            if resp.status_code == 200:
                data = resp.json()
                models_list = [m.get("id") for m in data.get("data", []) if m.get("id")]
                if models_list and resolved_model not in models_list:
                    matched = next((m for m in models_list if m == resolved_model.split("/")[-1]), models_list[0])
                    resolved_model = matched
    except Exception:
        pass

    return {
        "base_url": resolved_base,
        "api_key": resolved_key,
        "model": resolved_model,
    }


def _build_mom_payload(
    cfg: Dict[str, str],
    transcript_text: str,
    system_prompt: Optional[str],
    temperature: float,
    stream: bool,
) -> Dict[str, Any]:
    """Construct chat completions request payload."""
    return {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": system_prompt or DEFAULT_MOM_PROMPT},
            {
                "role": "user",
                "content": f"Here is the raw meeting transcript:\n\n{transcript_text}\n\nPlease generate the comprehensive Minutes of Meeting (MOM).",
            },
        ],
        "temperature": temperature,
        "stream": stream,
    }


def generate_mom_sync(
    segments: List[Union[Dict[str, Any], Any]],
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.2,
    timeout: float = 120.0,
) -> str:
    """Generate Minutes of Meeting (MOM) markdown synchronously."""
    cfg = _get_llm_config(base_url, api_key, model)
    transcript_text = format_transcript_for_mom(segments)
    if not transcript_text:
        return "*(Empty transcript provided; unable to generate MOM)*"

    payload = _build_mom_payload(cfg, transcript_text, system_prompt, temperature, False)
    headers = {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}
    url = f"{cfg['base_url']}/chat/completions"

    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()


def _parse_sse_chunk(line: str) -> Optional[str]:
    """Extract content delta from SSE data line."""
    if not line.startswith("data: "):
        return None
    data_str = line[6:].strip()
    if data_str == "[DONE]":
        return "[DONE]"
    try:
        chunk = json.loads(data_str)
        return chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
    except Exception:
        return None


async def generate_mom_stream(
    segments: List[Union[Dict[str, Any], Any]],
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.2,
    timeout: float = 180.0,
) -> AsyncGenerator[str, None]:
    """Generate Minutes of Meeting (MOM) markdown asynchronously yielding streaming text chunks."""
    cfg = _get_llm_config(base_url, api_key, model)
    transcript_text = format_transcript_for_mom(segments)
    if not transcript_text:
        yield "*(Empty transcript provided; unable to generate MOM)*"
        return

    payload = _build_mom_payload(cfg, transcript_text, system_prompt, temperature, True)
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
