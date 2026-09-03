"""Thin Ollama client — chat + JSON-enforced chat, used by the Phase 2 stages."""
import json

from .config import OLLAMA_MODEL

# Give the model enough context for a typical call transcript.
_OPTIONS = {"temperature": 0.2, "num_ctx": 8192}


def chat(prompt: str, system: str | None = None, model: str | None = None,
         fmt: str | dict | None = None, options: dict | None = None) -> str:
    import ollama
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = ollama.chat(
        model=model or OLLAMA_MODEL,
        messages=messages,
        options={**_OPTIONS, **(options or {})},
        format=fmt,                       # "json" or a JSON schema dict
    )
    return resp["message"]["content"]


def chat_json(prompt: str, system: str | None = None, model: str | None = None,
              retries: int = 1, options: dict | None = None) -> dict:
    """Chat forcing valid JSON. Retries once on parse failure."""
    last = ""
    for attempt in range(retries + 1):
        raw = chat(prompt, system=system, model=model, fmt="json", options=options)
        last = raw
        try:
            return json.loads(_strip_fences(raw))
        except json.JSONDecodeError:
            if attempt < retries:
                prompt = prompt + "\n\nReturn ONLY valid JSON, no prose, no code fences."
    raise ValueError(f"Model did not return valid JSON:\n{last[:500]}")


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
        if s.endswith("```"):
            s = s.rsplit("```", 1)[0]
    return s.strip()
