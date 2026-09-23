"""Загрузка RVC-модели + конвертация файла целиком (с нарезкой длинного по тишине)."""
import os
import numpy as np
import torch
import faiss
import librosa

from .rvclib.models import SynthesizerTrnMs768NSFsid, SynthesizerTrnMs768NSFsid_nono
from . import pipe

_cache = {}
_current_key = None


def unload():
    """Выгрузить RVC-модель из VRAM (при смене голоса). Hubert общий - остается."""
    global _current_key
    for key in list(_cache.keys()):
        try:
            del _cache[key]["net"]
        except Exception:
            pass
        del _cache[key]
    _current_key = None
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass
    print("[rvc] модель выгружена")


def vram_free_gb() -> float:
    try:
        import torch
        free, _ = torch.cuda.mem_get_info()
        return free / 1024 ** 3
    except Exception:
        return 0.0


def load(model_path: str, index_path: str | None = None, device: str = "cuda:0", is_half: bool = True):
    global _current_key
    key = (model_path, device, is_half)
    if key in _cache:
        return _cache[key]
    if _current_key is not None:
        print(f"[rvc] смена голоса - выгружаю предыдущий ({vram_free_gb():.1f}GB свободно)")
        unload()
    cpt = torch.load(model_path, map_location="cpu")
    cfg = list(cpt["config"])
    cfg[-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    cls = SynthesizerTrnMs768NSFsid if if_f0 == 1 else SynthesizerTrnMs768NSFsid_nono
    net_g = cls(*cfg, is_half=is_half)
    try:
        del net_g.enc_q
    except Exception:
        pass
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g.eval().to(device)
    net_g = net_g.half() if is_half else net_g.float()
    tgt_sr = cfg[-1]
    index, index_vecs = None, None
    if index_path and os.path.exists(index_path):
        index = faiss.read_index(index_path)
        index_vecs = index.reconstruct_n(0, index.ntotal)
    pack = {"net": net_g, "tgt_sr": tgt_sr, "index": index,
            "index_vecs": index_vecs, "if_f0": if_f0, "device": device, "is_half": is_half}
    _cache[key] = pack
    _current_key = key
    n = sum(p.numel() for p in net_g.parameters()) / 1e6
    print(f"[rvc] модель {os.path.basename(model_path)}: {n:.1f}M params, sr={tgt_sr}, f0={if_f0}")
    return pack


def split_silence(audio_16k: np.ndarray, max_len_s: float = 60.0):
    """Режем длинное на куски по тишине чтобы не лопнуть VRAM."""
    max_n = int(max_len_s * 16000)
    if len(audio_16k) <= max_n:
        return [(0, len(audio_16k))]
    intervals = librosa.effects.split(audio_16k, top_db=35)
    cuts, start = [0], 0
    for s, e in intervals:
        if e - start >= max_n:
            cuts.append(s)
            start = s
    cuts.append(len(audio_16k))
    return list(zip(cuts[:-1], cuts[1:]))


def convert(pack, audio_16k: np.ndarray, transpose: int = 0, index_rate: float = 0.75,
            rms_mix_rate: float = 1.0, protect: float = 0.33,
            ref_level: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Вход моно 16k float32. Выход (моно float32, tgt_sr).
    ref_level: оригинал вокала - подгоняем громкость выхода под его RMS (иначе
    RVC тихий и тонет в музыке). Усиление capped x6 чтобы не разогнать шум."""
    out = []
    for s, e in split_silence(audio_16k):
        y = pipe.convert_chunk(
            pack["net"], 0, audio_16k[s:e], transpose,
            pack["index"], pack["index_vecs"], index_rate, protect,
            rms_mix_rate, pack["device"], pack["is_half"],
        )
        out.append(y)
    audio = np.concatenate(out).astype(np.float32)
    if ref_level is not None and len(ref_level) > 100:
        import librosa
        probe = librosa.resample(audio, orig_sr=pack["tgt_sr"], target_sr=16000)
        n = min(len(probe), len(ref_level))
        rms_out = float(np.sqrt((probe[:n] ** 2).mean()) + 1e-9)
        rms_ref = float(np.sqrt((ref_level[:n] ** 2).mean()) + 1e-9)
        gain = min(rms_ref / rms_out, 6.0)
        audio = audio * gain
        print(f"[rvc] gain x{gain:.2f} под уровень оригинала")
    peak = np.abs(audio).max()
    if peak > 0.99:
        audio = audio / peak * 0.99
    return audio, pack["tgt_sr"]
