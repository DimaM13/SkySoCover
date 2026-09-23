"""Реестр RVC-голосов из D:\\ModelsVoice + ручные типы для автопитча."""
import json
import os

MODELS_DIR = r"D:\ModelsVoice"
SIDECAR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices.json")

# тип голоса -> целевой медианный F0. Типы расставлены руками один раз.
TYPE_F0 = {"female": 220.0, "male": 120.0, "child": 300.0}
DEFAULT_TYPES = {
    "jaily": "female", "sabrina": "female", "keka": "female",
    "wylsa": "female", "Girl - Weights Model": "female",
    "devochka rebenok": "child",
    "torreto": "male", "brejnev": "male", "yuriy dud": "male",
}


def _scan():
    voices = {}
    if not os.path.isdir(MODELS_DIR):
        return voices
    for name in sorted(os.listdir(MODELS_DIR)):
        d = os.path.join(MODELS_DIR, name)
        if not os.path.isdir(d):
            continue
        pth = index = None
        for f in os.listdir(d):
            if f.endswith(".pth") and pth is None:
                pth = os.path.join(d, f)
            if f.endswith(".index") and index is None:
                index = os.path.join(d, f)
        if pth:
            voices[name] = {"pth": pth, "index": index,
                            "type": DEFAULT_TYPES.get(name, "female")}
    return voices


def load():
    data = _scan()
    if os.path.exists(SIDECAR):
        try:
            with open(SIDECAR, encoding="utf-8") as f:
                saved = json.load(f)
            for name, info in saved.items():
                if name in data:
                    data[name].update(info)
        except Exception:
            pass
    return data


def save_overrides(overrides: dict):
    with open(SIDECAR, "w", encoding="utf-8") as f:
        json.dump(overrides, f, ensure_ascii=False, indent=1)
