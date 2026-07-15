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
    "You are a Khmer and Cambodian movie recommendation assistant. "
    "Answer the user's question using ONLY the sources provided below. "
    "Cite the movie titles you used as evidence (e.g. 'The Haunted House (2005)'). "
    "If the sources don't contain enough information to answer confidently, "
    "say so instead of making up an answer. "
    "Format your response as a natural paragraph or two of recommendation text "
    "with the source citations worked in."
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
        f"Sources:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer with specific movie title citations:"
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
