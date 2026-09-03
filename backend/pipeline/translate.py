"""Stage 3 (optional) — translate turns into a target language via the local LLM."""
from .. import llm

SYSTEM = (
    "You are a professional translator. Translate faithfully and naturally, "
    "preserving meaning and tone. Return only valid JSON."
)

_BATCH = 20   # turns per LLM call, to stay well within context

# Common language name/code aliases -> ISO code, to detect "already in target language".
LANG_ALIASES = {
    "en": "en", "english": "en", "hi": "hi", "hindi": "hi", "es": "es", "spanish": "es",
    "fr": "fr", "french": "fr", "de": "de", "german": "de", "pa": "pa", "punjabi": "pa",
    "ur": "ur", "urdu": "ur", "ar": "ar", "arabic": "ar", "pt": "pt", "portuguese": "pt",
    "zh": "zh", "chinese": "zh", "mandarin": "zh", "ru": "ru", "russian": "ru",
    "ja": "ja", "japanese": "ja", "it": "it", "italian": "it", "bn": "bn", "bengali": "bn",
    "ta": "ta", "tamil": "ta", "te": "te", "telugu": "te", "mr": "mr", "marathi": "mr",
    "gu": "gu", "gujarati": "gu", "kn": "kn", "kannada": "kn", "ml": "ml", "malayalam": "ml",
}


def norm_lang(x: str | None) -> str | None:
    return LANG_ALIASES.get((x or "").strip().lower())


def same_language(detected_code: str | None, target: str | None) -> bool:
    """True if the source is already in the requested target language (so skip translation)."""
    a, b = norm_lang(detected_code), norm_lang(target)
    return bool(a and b and a == b)


def translate_turns(turns: list, target_language: str, on_progress=None) -> list:
    """Add a `text_translated` field to each turn. No-op if target_language is empty."""
    if not turns or not target_language:
        return turns

    n_batches = (len(turns) + _BATCH - 1) // _BATCH
    for bi, start in enumerate(range(0, len(turns), _BATCH), 1):
        if on_progress:
            on_progress(bi, n_batches)
        batch = turns[start:start + _BATCH]
        payload = [{"i": start + k, "text": t["text"]} for k, t in enumerate(batch)]
        prompt = (
            f"Translate each item's `text` into {target_language}.\n"
            f'Return JSON: {{"items":[{{"i":<same index>,"text":"<translation>"}}]}}\n'
            f"Keep the same indices. Translate only; add no commentary.\n\n"
            f"{payload}"
        )
        out = llm.chat_json(prompt, system=SYSTEM)
        by_i = {item["i"]: item.get("text", "") for item in out.get("items", [])}
        for k, t in enumerate(batch):
            t["text_translated"] = by_i.get(start + k, "")
    return turns
