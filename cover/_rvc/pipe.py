"""Упрощенный RVC-пайплайн по мотивам официального (MIT):
паддинг/чанки, pyin-F0 вместо parselmouth/rmvpe, faiss-поиск, синтез.
Без cuda graphs - прямой вызов net_g.infer.
"""
import numpy as np
import torch
import torch.nn.functional as F
import librosa
from scipy import signal

from . import hubert_t

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)
WINDOW = 160
F0_MIN, F0_MAX = 50, 1100
F0_MEL_MIN = 1127 * np.log(1 + F0_MIN / 700)
F0_MEL_MAX = 1127 * np.log(1 + F0_MAX / 700)


def change_rms(data1, sr1, data2, sr2, rate):
    rms1 = librosa.feature.rms(y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2)
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
    rms1 = torch.from_numpy(rms1)
    rms1 = F.interpolate(rms1.unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(rms2.unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    data2 *= (torch.pow(rms1, torch.tensor(1 - rate)) * torch.pow(rms2, torch.tensor(rate - 1))).numpy()
    return data2


def medfilt_voiced(f0: np.ndarray, k: int = 5) -> np.ndarray:
    """Медиана от октавных скачков, нули (паузы) не трогаем."""
    from scipy.signal import medfilt
    v = f0 > 1
    if v.sum() > k:
        f = f0.copy()
        f[v] = medfilt(f[v], kernel_size=k if k % 2 else k + 1)
        return f
    return f0


def to_coarse(f0bak: np.ndarray):
    mel = 1127 * np.log(1 + f0bak / 700)
    mel[mel > 0] = (mel[mel > 0] - F0_MEL_MIN) * 254 / (F0_MEL_MAX - F0_MEL_MIN) + 1
    mel[mel <= 1] = 1
    mel[mel > 255] = 255
    return np.rint(mel).astype(np.int32)


def _fit_len(f0: np.ndarray, p_len: int, transpose: int):
    f0 = medfilt_voiced(f0.astype(np.float32)) * pow(2, transpose / 12)
    if len(f0) < p_len:
        f0 = np.pad(f0, (0, p_len - len(f0)), mode="edge")
    else:
        f0 = f0[:p_len]
    return f0.copy(), f0.copy()


def f0_rmvpe(x_16k: np.ndarray, p_len: int, transpose: int = 0):
    """RMVPE (бандловый onnx) -> (coarse, f0bak). Для песен точнее pyin."""
    from . import rmvpe_onnx
    f0 = rmvpe_onnx.infer(x_16k)
    voiced = f0 > 1
    if voiced.sum() >= 2:
        f0[~voiced] = np.interp(np.where(~voiced)[0], np.where(voiced)[0], f0[voiced])
    else:
        f0 = np.nan_to_num(f0, nan=110.0)
    coarse, bak = _fit_len(f0, p_len, transpose)
    return to_coarse(coarse), bak


def f0_pyin(x_16k: np.ndarray, p_len: int, transpose: int = 0):
    """pyin -> (coarse(1-255), f0bak float), длина p_len."""
    f0, _, _ = librosa.pyin(x_16k, fmin=F0_MIN, fmax=F0_MAX, sr=16000)
    if f0 is None:
        f0 = np.zeros(len(x_16k) // WINDOW + 1)
    voiced = ~np.isnan(f0)
    if voiced.sum() >= 2:
        f0[~voiced] = np.interp(np.where(~voiced)[0], np.where(voiced)[0], f0[voiced])
    else:
        f0 = np.nan_to_num(f0, nan=110.0)
    coarse, bak = _fit_len(f0, p_len, transpose)
    return to_coarse(coarse), bak


@torch.no_grad()
def convert_chunk(net_g, sid: int, audio_16k: np.ndarray, transpose: int,
                  index, index_vectors, index_rate: float, protect: float,
                  rms_mix_rate: float, device: str, is_half: bool,
                  f0_method: str = "rmvpe"):
    audio = signal.filtfilt(bh, ah, audio_16k.astype(np.float32))
    pad = 16000 // 2
    audio_pad = np.pad(audio, (pad, pad), mode="reflect")

    feats = torch.from_numpy(audio_pad).float().view(1, -1)
    feats = hubert_t.extract_features(feats[0], device)
    feats = feats.half() if is_half else feats.float()
    if protect < 0.5:
        feats0 = feats.clone()

    p_len = audio_pad.shape[0] // WINDOW
    if f0_method == "rmvpe":
        try:
            pitch_c, pitch_f = f0_rmvpe(audio_pad, p_len, transpose)
        except Exception as e:
            print(f"[rvc] rmvpe упал ({e}), падаю на pyin")
            pitch_c, pitch_f = f0_pyin(audio_pad, p_len, transpose)
    else:
        pitch_c, pitch_f = f0_pyin(audio_pad, p_len, transpose)
    pitch = torch.tensor(pitch_c, device=device).unsqueeze(0).long()
    pitchf = torch.tensor(pitch_f, device=device).unsqueeze(0).float()

    if index is not None and index_vectors is not None and index_rate != 0:
        npy = feats[0].float().cpu().numpy()
        score, ix = index.search(npy, k=8)
        weight = np.square(1 / score)
        weight /= weight.sum(axis=1, keepdims=True)
        npy = np.sum(index_vectors[ix] * np.expand_dims(weight, axis=2), axis=1)
        if is_half:
            npy = npy.astype("float16")
        feats = (
            torch.from_numpy(npy).unsqueeze(0).to(device) * index_rate
            + (1 - index_rate) * feats.to(device)
        )

    feats = F.interpolate(feats.float().permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
    if is_half:
        feats = feats.half()
    if protect < 0.5:
        feats0 = F.interpolate(feats0.float().permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if is_half:
            feats0 = feats0.half()
    n = feats.shape[1]
    if n < p_len:
        p_len = n
        pitch, pitchf = pitch[:, :p_len], pitchf[:, :p_len]
    if protect < 0.5:
        pitchff = pitchf.clone()
        pitchff[pitchf > 0] = 1
        pitchff[pitchf < 1] = protect
        pitchff = pitchff.unsqueeze(-1)
        feats = feats * pitchff + feats0 * (1 - pitchff)
        feats = feats.to(feats0.dtype)

    p_len_t = torch.tensor([p_len], device=device).long()
    out = net_g.infer(feats, p_len_t, pitch, pitchf, torch.tensor(sid, device=device).unsqueeze(0).long())[0]
    audio1 = out[0, 0].data.cpu().float().numpy()

    # срезаем паддинг: выход в tgt_sr, считаем пропорцией
    ratio = len(audio1) / len(audio_pad)
    cut = int(pad * ratio)
    audio1 = audio1[cut: len(audio1) - cut] if cut > 0 else audio1
    if rms_mix_rate != 1:
        audio1 = change_rms(audio, 16000, audio1, 40000, rms_mix_rate)
    return audio1
