"""
Generation: turn retrieved chunks + a query into a final answer.

Two modes are provided:
- "extractive" (default): no API key needed, works immediately. Just stitches
  together the retrieved chunks so you can verify retrieval quality before wiring
  up an LLM.
- "llm": calls DeepSeek API to write a grounded recommendation from the retrieved
  context. Set DEEPSEEK_API_KEY environment variable to use it.
"""

import os
from pathlib import Path
from typing import List, Tuple

from .ingest import Chunk


def _load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE entries without replacing existing variables."""
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


# Load local development secrets without overriding deployment variables.
_load_env_file(Path(__file__).resolve().parents[1] / ".env")

SYSTEM_PROMPT = (
    "You are a Khmer and Cambodian movie search assistant. "
    "You MUST answer the user's question using ONLY the sources provided below. "
    "You MUST NOT use any of your own general knowledge about movies, directors, "
    "actors, or Cambodian cinema. "
    "You MUST cite the specific source movie title for every claim you make, "
    "using the format: [Movie Title]. "
    "If the sources do not contain enough information to answer the question, "
    "you MUST say: 'Based on the available sources, I do not have enough "
    "information to answer this question.' "
    "Do not mention actors, release dates, or details that are not present "
    "in the provided sources. "
    "Keep your response concise — 2 to 4 sentences."
)


def extractive_answer(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    if not retrieved:
        return "No relevant passages were found for that query."
    lines = [f"Top passages related to: \u201c{query}\u201d\n"]
    for chunk, score in retrieved:
        lines.append(f"[{chunk.doc_title}, score={score:.2f}] {chunk.text}\n")
    return "\n".join(lines)


def llm_answer(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    if not retrieved:
        return "No relevant sources were found for that query."

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return (
            "[LLM mode not configured] Set the DEEPSEEK_API_KEY environment "
            "variable before starting the app. "
            "Falling back to extractive mode:\n\n" + extractive_answer(query, retrieved)
        )

    context = "\n\n".join(
        f"[{c.doc_title}]\n{c.text}" for c, _ in retrieved
    )

    user_message = (
        f"SOURCES (only use these — do not add outside knowledge):\n\n"
        f"{context}\n\n"
        f"QUESTION: {query}\n\n"
        f"ANSWER (cite every claim with [Title]):"
    )

    import openai
    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=1000,
        temperature=0.3,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return response.choices[0].message.content


def generate_answer(
    query: str, retrieved: List[Tuple[Chunk, float]], mode: str = "extractive"
) -> str:
    if mode == "llm":
        return llm_answer(query, retrieved)
    return extractive_answer(query, retrieved)
