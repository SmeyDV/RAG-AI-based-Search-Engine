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
from typing import List, Tuple

from .ingest import Chunk

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


def llm_answer(query: str, retrieved: List[Tuple[Chunk, float]], api_key: str = "") -> str:
    if not retrieved:
        return "No relevant sources were found for that query."

    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return (
            "[LLM mode not configured] Enter your DeepSeek API key in the sidebar, "
            "or set the DEEPSEEK_API_KEY environment variable. "
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
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=500,
        temperature=0.3,
    )
    return response.choices[0].message.content


def generate_answer(query: str, retrieved: List[Tuple[Chunk, float]], mode: str = "extractive", api_key: str = "") -> str:
    if mode == "llm":
        return llm_answer(query, retrieved, api_key=api_key)
    return extractive_answer(query, retrieved)
