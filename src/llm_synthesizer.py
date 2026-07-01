#!/usr/bin/env python3
"""
llm_synthesizer.py

Phase 2 (Intelligence Upgrade) — Milestone 2: local generative synthesis.

Takes the top-ranked clauses returned by graph_traversal's Personalized
PageRank search and asks a local Ollama model to synthesize them into a
coherent, factually-grounded prose answer to the user's query.

Design constraints this module enforces:
  - Strict factual grounding: the prompt explicitly forbids extrapolation
    and requires an explicit "not enough information" admission when the
    context doesn't support an answer.
  - Clean clause separation: each context clause is placed on its own
    numbered line, never concatenated or run together, so a fragmented or
    mid-sentence clause (a known artifact of Phase 1's dependency-parse
    clause splitter) doesn't get misread by the model as continuing the
    previous line's thought.
  - Network resilience: distinguishes "Ollama isn't running" (connection
    refused) from "Ollama is running but slow" (timeout) from "Ollama
    returned an error" (HTTP 4xx/5xx) so the API layer can respond with the
    correct status code instead of a generic 500.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

import httpx

log = logging.getLogger("llm_synthesizer")
if not log.handlers:
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )


# ---------------------------------------------------------------------------
# Errors — distinct types so callers (the FastAPI route) can map each to the
# right HTTP status instead of a blanket 500.
# ---------------------------------------------------------------------------

class LLMSynthesisError(Exception):
    """Base class for all synthesis failures."""


class OllamaUnavailableError(LLMSynthesisError):
    """Raised when the Ollama daemon could not be reached at all (connection refused/DNS/etc)."""


class OllamaTimeoutError(LLMSynthesisError):
    """Raised when Ollama accepted the connection but did not respond in time."""


class OllamaResponseError(LLMSynthesisError):
    """Raised when Ollama responded with a non-2xx status or a malformed payload."""


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = (
    "Synthesize these extracted semantic facts into a coherent, professionally "
    "written response to the user query. Maintain strict factual grounding. "
    "Do not extrapolate, guess, or hallucinate. If the answer cannot be "
    "confidently derived from the provided context, explicitly state that you "
    "do not have enough information."
)


def _clean_clause(text: str) -> str:
    """
    Normalize a single context clause before it goes into the prompt.

    Phase 1's dependency-parse clause splitter can emit fragments without
    trailing punctuation, or with stray leading/trailing whitespace and
    internal double-spacing. This collapses whitespace and guarantees
    terminal punctuation so each numbered line in the prompt reads as a
    clearly bounded, complete-looking statement rather than bleeding into
    the next line.
    """
    cleaned = re.sub(r"\s+", " ", text).strip()
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def build_prompt(
    query: str,
    context_clauses: list[str],
    max_context_chars: int = 6000,
) -> str:
    """
    Assemble the full prompt sent to Ollama.

    Each clause is cleaned and placed on its own numbered line inside a
    clearly delimited context block, keeping clause boundaries unambiguous
    for the model even when individual clauses are short fragments.

    Args:
        query:              The user's natural-language question.
        context_clauses:    Ranked clause texts from personalized_pagerank_search,
                             best first.
        max_context_chars:  Hard cap on the assembled context block's length —
                             a defensive ceiling independent of how many
                             clauses were passed in, in case any single
                             clause is unexpectedly long.

    Returns:
        The complete prompt string.
    """
    numbered_lines: list[str] = []
    running_len = 0

    for i, clause in enumerate(context_clauses, start=1):
        cleaned = _clean_clause(clause)
        if not cleaned:
            continue
        line = f"{i}. {cleaned}"
        if running_len + len(line) > max_context_chars:
            log.warning(
                "Context truncated at %d/%d clauses — max_context_chars=%d reached.",
                i - 1, len(context_clauses), max_context_chars,
            )
            break
        numbered_lines.append(line)
        running_len += len(line) + 1

    if not numbered_lines:
        context_block = "(no supporting context was retrieved)"
    else:
        context_block = "\n".join(numbered_lines)

    prompt = (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"--- EXTRACTED CONTEXT ---\n"
        f"{context_block}\n"
        f"--- END CONTEXT ---\n\n"
        f"User query: {query.strip()}\n\n"
        f"Response:"
    )
    return prompt


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SynthesisResult:
    answer:          str
    model:           str
    prompt_chars:    int
    context_clauses_used: int
    latency_ms:      float


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

async def synthesize_answer(
    query: str,
    context_clauses: list[str],
    ollama_base_url: str,
    model_name: str,
    connect_timeout_seconds: float = 5.0,
    request_timeout_seconds: float = 120.0,
    temperature: float = 0.2,
    num_predict: int = 512,
    max_context_chars: int = 6000,
) -> SynthesisResult:
    """
    Send the assembled prompt to a local Ollama daemon and return the
    synthesized answer.

    Args:
        query:                    User's natural-language question.
        context_clauses:          Ranked clause texts, best first (from PPR).
        ollama_base_url:          e.g. "http://localhost:11434".
        model_name:                Pulled Ollama model identifier (e.g. "llama3").
        connect_timeout_seconds:   Time to wait for the daemon to accept a
                                    connection before declaring it unreachable.
        request_timeout_seconds:   Time to wait for the full response — kept
                                    generous since consumer CPU inference can
                                    genuinely take 30-90s+ for a single pass.
        temperature, num_predict:  Ollama generation options.
        max_context_chars:         Passed through to build_prompt().

    Returns:
        SynthesisResult with the generated prose answer and metadata.

    Raises:
        OllamaUnavailableError: the daemon could not be reached at all.
        OllamaTimeoutError:     the daemon accepted the connection but did
                                 not respond within request_timeout_seconds.
        OllamaResponseError:    the daemon responded with an error status or
                                 a payload that didn't match the expected schema.
    """
    prompt = build_prompt(query, context_clauses, max_context_chars=max_context_chars)

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }

    url = f"{ollama_base_url.rstrip('/')}/api/generate"
    timeout = httpx.Timeout(connect=connect_timeout_seconds, read=request_timeout_seconds,
                             write=request_timeout_seconds, pool=connect_timeout_seconds)

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)
    except httpx.ConnectError as exc:
        raise OllamaUnavailableError(
            f"Could not connect to Ollama at {ollama_base_url}. "
            f"Is the daemon running? (ollama serve)  Underlying error: {exc}"
        ) from exc
    except httpx.ConnectTimeout as exc:
        raise OllamaUnavailableError(
            f"Timed out connecting to Ollama at {ollama_base_url} "
            f"after {connect_timeout_seconds}s. Underlying error: {exc}"
        ) from exc
    except httpx.ReadTimeout as exc:
        raise OllamaTimeoutError(
            f"Ollama accepted the connection but did not respond within "
            f"{request_timeout_seconds}s. The model may be overloaded on this "
            f"hardware — consider a smaller model or a longer timeout. "
            f"Underlying error: {exc}"
        ) from exc
    except httpx.TimeoutException as exc:
        raise OllamaTimeoutError(
            f"Request to Ollama timed out: {exc}"
        ) from exc
    except httpx.HTTPError as exc:
        raise OllamaUnavailableError(f"HTTP transport error contacting Ollama: {exc}") from exc

    elapsed_ms = (time.perf_counter() - t0) * 1000

    if response.status_code != 200:
        raise OllamaResponseError(
            f"Ollama returned HTTP {response.status_code}: {response.text[:500]}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise OllamaResponseError(f"Ollama response was not valid JSON: {exc}") from exc

    answer = data.get("response")
    if answer is None:
        raise OllamaResponseError(
            f"Ollama response payload missing 'response' field. Keys present: {list(data.keys())}"
        )

    answer = answer.strip()
    if not answer:
        raise OllamaResponseError("Ollama returned an empty response.")

    log.info(
        "Synthesis complete: model=%s, prompt_chars=%d, latency_ms=%.1f",
        model_name, len(prompt), elapsed_ms,
    )

    return SynthesisResult(
        answer=answer,
        model=model_name,
        prompt_chars=len(prompt),
        context_clauses_used=len(context_clauses),
        latency_ms=round(elapsed_ms, 1),
    )