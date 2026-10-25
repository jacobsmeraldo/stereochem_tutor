from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict, Iterable, List, Mapping

MODEL_NAME = "gemini-3.8-flash"
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
- NEVER give away the answer key.
- Guide them through: 1. Priority ranking. 2. Points of difference. 3. 3D orientation.
- DO NOT mention Newman projections or Chair conformations. Keep the student entirely focused on R/S CIP rules.

CRITICAL: You must be extremely concise. Limit every response to 2 short bullet points maximum.
""".strip()

    prompt = _build_stereo_prompt(chat_history, problem_data, student_assignments)
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.35,
        max_output_tokens=200, 
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
