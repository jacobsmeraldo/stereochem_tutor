from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Mapping

MODEL_NAME = "gemini-3.8-flash"

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

def get_stereo_tutor_stream(
    chat_history: Iterable[Mapping[str, str]],
    problem_data: Mapping[str, Any],
    student_assignments: Mapping[str, str],
):
    """Yields text chunks as they arrive from the API for real-time streaming."""
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        yield "Error: Set GEMINI_API_KEY before requesting tutor feedback."
        return

    client = genai.Client(api_key=api_key)

    system_instruction = """
You are a Socratic Coach for undergraduate Organic Chemistry stereochemistry.

CORE TASK - R/S ASSIGNMENTS:
- Help students assign R and S absolute configurations to chiral centers.
- NEVER give away the answer key immediately.
- Guide them methodically through: 1. Priority ranking (atomic number). 2. Points of difference. 3. 3D spatial orientation (is #4 in back?).
- DO NOT mention Newman projections or Chair conformations. Keep the student entirely focused on R/S CIP rules.
- Provide thorough, encouraging, and conversational step-by-step guidance. Break down your reasoning clearly.
""".strip()

    prompt = _build_stereo_prompt(chat_history, problem_data, student_assignments)
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.35,
        max_output_tokens=800, 
    )

    try:
        response_stream = client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=prompt,
            config=config,
        )
        for chunk in response_stream:
            if chunk.text:
                yield chunk.text
    except Exception as exc:
        yield f"I encountered an error reaching the API: {exc}"
