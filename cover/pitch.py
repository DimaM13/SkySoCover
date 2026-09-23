"""Автоподбор параметров RVC.

Честная схема: Gemma через наш API слышать НЕ умеет (только текст/картинки),
поэтому тон считается математикой по F0, а Gemma по описанию подбирает
остальное (index_rate, rms_mix, protect) зная целевой голос.
"""
import math
import numpy as np
import librosa

from .voices import TYPE_F0


def source_stats(vocals_16k: np.ndarray) -> dict:
    f0, _, _ = librosa.pyin(vocals_16k, fmin=50, fmax=1100, sr=16000)
    voiced = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    if len(voiced) < 10:
        return {"median": 150.0, "voiced": 0.0, "kind": "speech"}
    return {"median": float(np.median(voiced)), "voiced": float(len(voiced) / len(f0)),
            "kind": "speech"}


def suggest_transpose(src_median: float, target_type: str) -> int:
    tgt = TYPE_F0.get(target_type, 220.0)
    st = int(round(12 * math.log2(tgt / max(50.0, src_median))))
    # больше ±6 сдвиг рвет качество (проверено: -8 на Girl дало кашу)
    return max(-6, min(6, st))


def gemma_params(source_desc: str, voice_name: str, voice_type: str,
                 src_median: float, transpose: int) -> dict:
    """Gemma выбирает expressive-параметры зная голос и цифры анализа."""
    from google import genai
    from google.genai import types
    import sys
    sys.path.insert(0, ".")
    import config
    client = genai.Client(api_key=config.GOOGLE_API_KEY)
    prompt = (
        "You tune RVC voice conversion. Reply ONLY JSON like "
        '{"index_rate": 0.75, "rms_mix_rate": 1.0, "protect": 0.33} '
        "with one-line reason per key in plain text after JSON.\n"
        f"Source: {source_desc}. Measured median F0 {src_median:.0f} Hz. "
        f"Target voice '{voice_name}' ({voice_type}), transpose {transpose:+d} semitones.\n"
        "Rules: index_rate 0-1 (higher = closer to target timbre; speech 0.6-0.8, "
        "SONG 0.4-0.6 - below 0.4 the voice stays like the ORIGINAL, above 0.6 intonation goes robotic; "
        "noisy source -> ~0.4). "
        "rms_mix_rate 0-1 (1 keeps output dynamics; song -> 0.8). "
        "protect 0-0.5 (speech 0.3, SONG 0.4-0.5 to keep consonants and phrasing)."
    )
    resp = client.models.generate_content(
        model=config.GEMMA_MODEL,
        config=types.GenerateContentConfig(
            temperature=0.3, max_output_tokens=300,
            thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
        contents=prompt,
    )
    import json
    import re
    text = (resp.text or "")
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    out = {"index_rate": 0.75, "rms_mix_rate": 1.0, "protect": 0.33}
    if m:
        try:
            got = json.loads(m.group(0))
            for k in out:
                if k in got:
                    out[k] = float(got[k])
        except Exception:
            pass
    out["index_rate"] = min(1.0, max(0.0, out["index_rate"]))
    out["rms_mix_rate"] = min(1.0, max(0.0, out["rms_mix_rate"]))
    out["protect"] = min(0.5, max(0.0, out["protect"]))
    out["reason"] = text.strip()[:400]
    return out
