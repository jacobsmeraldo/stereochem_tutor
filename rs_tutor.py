from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict, Iterable, List, Mapping

MODEL_NAME = "gemini-2.5-flash-8b"
RETRYABLE_STATUS_CODES = {429, 503}

def _chat_history_text(chat_history: Iterable[Mapping[str, str]]) -> str:
    lines = []
    for message in list(chat_history)[-8:]:
        role = message.get("role", "user")
        content = message.get("content", "")
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "No previous conversation."

def _build_stereo_prompt(
    chat_history: Iterable[Mapping[str, str]],
    problem_data: Mapping[str, Any],
    student_assignments: Mapping[str, str],
) -> str:
    chiral_centers = problem_data.get("chiral_centers", {})
    
    results = []
    if student_assignments:
        for atom_idx, correct_code in chiral_centers.items():
            user_code = student_assignments.get(str(atom_idx), "Unassigned")
            results.append({
                "atom_index": int(atom_idx),
                "student_guess": user_code,
                "is_correct": user_code.upper() == correct_code.upper(),
                "internal_expected_cip_code": correct_code,
            })

    evaluation = {
        "molecule_name": problem_data.get("name"),
        "smiles": problem_data.get("smiles"),
        "stereo_results": results,
    }

    return (
        "Recent conversation:\n"
        f"{_chat_history_text(chat_history)}\n\n"
        "Current student submission report follows as JSON. "
        "Fields named `internal_expected_cip_code` are for your private reasoning only.\n"
        f"{json.dumps(evaluation, indent=2)}"
    )

def _error_code(exc: BaseException) -> int | None:
    for attr in ("code", "status_code", "status"):
        value = getattr(exc, attr, None)
        if value is not None:
            try: return int(value)
            except (TypeError, ValueError): return None
    return None

def get_stereo_tutor_response(
    chat_history: Iterable[Mapping[str, str]],
    problem_data: Mapping[str, Any],
    student_assignments: Mapping[str, str],
) -> str:
    from google import genai
    from google.genai import errors, types

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY before requesting tutor feedback.")

    client = genai.Client(api_key=api_key)

    system_instruction = """
You are a Socratic Coach for undergraduate Organic Chemistry stereochemistry.

CORE TASK - R/S ASSIGNMENTS:
- Help students assign R and S absolute configurations to chiral centers.
- NEVER give away the answer key. If they are wrong, guide them through the CIP steps:
  1. Priority ranking (atomic number).
  2. Points of difference (moving down the chain).
  3. 3D orientation (is priority #4 pointing away?).
- If a student asks how to draw this molecule as a Newman projection or Chair conformation, enthusiastically guide them through it! Connect their 2D wedge/dash knowledge to 3D up/down and axial/equatorial relationships.

Keep feedback concise: 2 to 4 bullet points or guiding questions per message.
""".strip()

    prompt = _build_stereo_prompt(chat_history, problem_data, student_assignments)
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.35,
        max_output_tokens=700,
    )

    max_attempts = 5
    base_delay = 1.0

    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=config,
            )
            return (response.text or "").strip() or "I could not generate feedback for this submission."
        except errors.APIError as exc:
            code = _error_code(exc)
            if code not in RETRYABLE_STATUS_CODES or attempt == max_attempts - 1:
                raise
            time.sleep(base_delay * (2**attempt) + random.uniform(0, 0.5))