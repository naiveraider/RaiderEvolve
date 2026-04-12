from __future__ import annotations

import asyncio
import random
import re
import time
from typing import Any

import httpx

from evolve.settings import settings

# OpenAI rate limits: 429; gateways sometimes return 502/503 under load.
_RETRYABLE_STATUS = frozenset({429, 502, 503})
_MAX_ATTEMPTS = 3


class LLMRequestError(Exception):
    """OpenAI-compatible chat/completions request failed (HTTP or parse)."""


def extract_code_block(text: str) -> str:
    fence = re.search(r"```(?:python)?\s*([\s\S]*?)```", text, re.I)
    if fence:
        return fence.group(1).strip()
    return text.strip()


def _backoff_seconds(attempt: int, response: httpx.Response | None) -> float:
    if response is not None:
        ra = response.headers.get("retry-after")
        if ra is not None:
            try:
                return min(120.0, max(1.0, float(ra)))
            except ValueError:
                pass
    return min(30.0, (2**attempt) * 0.5 + random.uniform(0, 0.35))


def _http_error_message(response: httpx.Response, *, after_retries: bool) -> str:
    status = response.status_code
    url = str(response.request.url)
    detail = (response.text or "").strip()[:3000]
    try:
        data = response.json()
        err = data.get("error")
        if isinstance(err, dict) and err.get("message"):
            detail = str(err["message"])
        elif isinstance(err, str):
            detail = err
    except Exception:
        pass

    hints: list[str] = []
    if status == 401:
        hints.append(
            "401: Invalid or missing API key for this URL, or key not loaded. "
            "Put OPENAI_API_KEY in the project root .env and start uvicorn from that directory "
            "(or rely on evolve/settings loading repo .env)."
        )
    elif status == 403:
        hints.append(
            "403: Often billing/credits, model access, or organization restrictions. Check OpenAI account and model availability."
        )
    elif status == 404:
        hints.append(
            "404: Wrong OPENAI_BASE_URL (expect .../v1, not .../chat/completions)."
        )
    elif status == 429:
        hints.append(
            "429: Rate limited" + (" after retries." if after_retries else ".") + " Reduce strategies/generations or wait."
        )

    tail = " | ".join(hints) if hints else ""
    retry_note = f" (after {_MAX_ATTEMPTS} attempts)" if after_retries and status in _RETRYABLE_STATUS else ""
    return f"HTTP {status}{retry_note} {url}\nAPI message: {detail}\n{tail}".strip()


def _raise_llm_http_error(response: httpx.Response, *, after_retries: bool) -> None:
    raise LLMRequestError(_http_error_message(response, after_retries=after_retries))


def _parse_completion(data: dict[str, Any]) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise LLMRequestError(f"Unexpected completion JSON shape: {data!r}") from e
    return extract_code_block(content or "")


async def improve_code_async(
    system_prompt: str,
    user_prompt: str,
) -> str:
    if not settings.openai_api_key:
        return _mock_improve(user_prompt)

    url = f"{settings.openai_base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.35,
    }
    last_response: httpx.Response | None = None
    async with httpx.AsyncClient(timeout=60.0) as client:
        for attempt in range(_MAX_ATTEMPTS):
            r = await client.post(url, json=payload, headers=headers)
            last_response = r
            if r.status_code in _RETRYABLE_STATUS and attempt < _MAX_ATTEMPTS - 1:
                await asyncio.sleep(_backoff_seconds(attempt, r))
                continue
            if r.is_error:
                _raise_llm_http_error(
                    r, after_retries=r.status_code in _RETRYABLE_STATUS and attempt == _MAX_ATTEMPTS - 1
                )
            return _parse_completion(r.json())
    if last_response is not None and last_response.is_error:
        _raise_llm_http_error(last_response, after_retries=True)
    raise LLMRequestError("LLM request failed after retries")


def improve_code_sync(system_prompt: str, user_prompt: str) -> str:
    if not settings.openai_api_key:
        return _mock_improve(user_prompt)

    url = f"{settings.openai_base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.35,
    }
    last_response: httpx.Response | None = None
    with httpx.Client(timeout=60.0) as client:
        for attempt in range(_MAX_ATTEMPTS):
            r = client.post(url, json=payload, headers=headers)
            last_response = r
            if r.status_code in _RETRYABLE_STATUS and attempt < _MAX_ATTEMPTS - 1:
                time.sleep(_backoff_seconds(attempt, r))
                continue
            if r.is_error:
                _raise_llm_http_error(
                    r, after_retries=r.status_code in _RETRYABLE_STATUS and attempt == _MAX_ATTEMPTS - 1
                )
            return _parse_completion(r.json())
    if last_response is not None and last_response.is_error:
        _raise_llm_http_error(last_response, after_retries=True)
    raise LLMRequestError("LLM request failed after retries")


def _mock_improve(user_prompt: str) -> str:
    """Deterministic tweak when no API key (for local demo)."""
    if "PARENT_CODE:" in user_prompt:
        chunk = user_prompt.split("PARENT_CODE:", 1)[-1].strip()
        if chunk:
            return chunk + "\n# mock-llm: no API key; echoed parent with comment\n"
    if "choose_action" in user_prompt or "pacman" in user_prompt.lower():
        from evolve.pacman_env import baseline_pacman_code

        return baseline_pacman_code().replace(
            "# pacman-agent", "# pacman-agent (mock-llm, set OPENAI_API_KEY for real edits)", 1
        )
    from evolve.matrix_task import baseline_matrix_code

    return baseline_matrix_code() + "\n# mock-llm: set OPENAI_API_KEY for real LLM mutations\n"
