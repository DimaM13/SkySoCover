"""Кавер/даббинг через RVC: видео -> demucs(голос отдельно) -> RVC -> музыка назад -> mux.

Музыка полностью вырезается ДО RVC (чтобы модель ее не захватила),
после озвучки фоновая музыка подмешивается обратно.
"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SR_OUT = 40000


def run(cmd):
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg fail [{e.returncode}]: {(e.stderr or '')[-500:]}")


def vram_free_gb() -> float:
    try:
        import torch
        free, _ = torch.cuda.mem_get_info()
        return free / 1024 ** 3
    except Exception:
        return 0.0


def split_wav(path: str, work: str, max_s: float = 150.0):
    """Нарезка на куски по тишине чтобы demucs не лопнул память на длинном."""
    import soundfile as sf
    import librosa
    d, sr = sf.read(path)
    mono = librosa.to_mono(d.T) if d.ndim > 1 else d
    y16 = librosa.resample(mono, orig_sr=sr, target_sr=16000)
    intervals = librosa.effects.split(y16, top_db=40)
    cuts, start = [0], 0
    for s, e in intervals:
        t0, t1 = s / 16000, e / 16000
        if t1 - start >= max_s:
            cuts.append(t0)
            start = t0
    cuts.append(len(y16) / 16000)
    paths = []
    for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
        if b - a < 1.0:
            continue
        p = os.path.join(work, f"part{i:02d}.wav")
        run(["ffmpeg", "-y", "-v", "error", "-i", path, "-ss", f"{a:.2f}", "-t", f"{b - a:.2f}", p])
        paths.append(p)
    return paths or [path]


def demucs_one(wav: str, out_dir: str, device: str):
    cmd = [sys.executable, "-m", "demucs", "--two-stems", "vocals",
           "-o", out_dir, "-d", device, wav]
    subprocess.run(cmd, check=True, capture_output=True)


def separate(wav: str, work: str, log=print) -> tuple[str | None, str | None]:
    """Demucs по кускам: целые альбомы не роняют GPU/ОЗУ. Возвращает (vocals, music)."""
    parts = split_wav(wav, work)
    log(f"[cover] demucs кусков: {len(parts)}")
    vocs, muss = [], []
    for i, part in enumerate(parts):
        out_dir = os.path.join(work, f"demucs_{i:02d}")
        free = vram_free_gb()
        dev = "cuda" if free >= 2.5 else "cpu"
        if dev == "cpu":
            log(f"[cover] VRAM {free:.1f}GB - demucs на CPU (медленно, зато без вылета; останови бота чтобы было быстрее)")
        try:
            demucs_one(part, out_dir, dev)
        except Exception:
            if dev == "cuda":
                log("[cover] cuda не взлетел, пробую CPU...")
                demucs_one(part, out_dir, "cpu")
            else:
                raise
        hit_v = hit_m = None
        for root, _, files in os.walk(out_dir):
            if "vocals.wav" in files:
                hit_v = os.path.join(root, "vocals.wav")
            if "no_vocals.wav" in files:
                hit_m = os.path.join(root, "no_vocals.wav")
        if not hit_v or not hit_m:
            raise RuntimeError("demucs не отдал дорожки")
        vocs.append(hit_v)
        muss.append(hit_m)

    def concat(files, out):
        lst = os.path.join(work, os.path.basename(out) + ".txt")
        with open(lst, "w", encoding="utf-8") as f:
            for p in files:
                f.write(f"file '{os.path.abspath(p)}'\n")
        run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
             "-ar", "44100", "-ac", "2", out])
        return out

    return concat(vocs, os.path.join(work, "vocals_all.wav")), \
        concat(muss, os.path.join(work, "music_all.wav"))


def run_job(job: dict, log=print):
    """job: input/voice/transpose/index_rate/rms_mix/protect/music(bool)/music_vol/out.
    Возвращает путь к готовому видео."""
    import numpy as np
    import soundfile as sf
    import librosa
    from cover._rvc import rvc
    from cover import voices, pitch

    vinfo = voices.load().get(job["voice"])
    if not vinfo:
        log(f"[cover] нет такого голоса: {job['voice']}")
        return None

    work = tempfile.mkdtemp(prefix="skysocover_")
    in44 = os.path.join(work, "in44.wav")
    log("[cover] 1/5 аудио из видео...")
    run(["ffmpeg", "-y", "-i", job["input"], "-ar", "44100", "-ac", "2", in44])
    total_dur = sf.info(in44).frames / 44100

    log("[cover] 2/5 вырезаю музыку (demucs, чтобы RVC ее не захватил)...")
    voc44, mus44 = separate(in44, work, log=log)

    log("[cover] 3/5 анализ голоса + параметры...")
    v16, _ = librosa.load(voc44, sr=16000, mono=True)
    stats = pitch.source_stats(v16.astype("float32"))
    is_song = stats["voiced"] > 0.5
    transpose = job.get("transpose")
    if transpose is None:
        transpose = pitch.suggest_transpose(stats["median"], vinfo["type"])
        log(f"[cover] автотон: F0 {stats['median']:.0f} -> {transpose:+d} полутонов ({vinfo['type']})")
    params = job.get("params") or pitch.gemma_params(
        f"{'песня' if stats['voiced'] > 0.5 else 'речь'}, {total_dur:.0f} сек",
        job["voice"], vinfo["type"], stats["median"], transpose)
    log(f"[cover] Gemma: index={params['index_rate']} rms={params['rms_mix_rate']} protect={params['protect']}")

    log("[cover] 4/5 RVC-конвертация...")
    pack = rvc.load(vinfo["pth"], vinfo.get("index"))
    out, sr = rvc.convert(pack, v16.astype("float32"), transpose=transpose,
                          index_rate=params["index_rate"],
                          rms_mix_rate=params["rms_mix_rate"], protect=params["protect"],
                          ref_level=v16.astype("float32"))
    conv_wav = os.path.join(work, "conv.wav")
    sf.write(conv_wav, out, sr)
    rvc.unload()  # сразу выгружаем голос из VRAM чтобы не копился
    log("[cover] голос выгружен из памяти")

    log("[cover] 5/5 музыка назад + склейка...")
    # входы ffmpeg: 0=видео, 1=конверт, 2=фон. НЕ ПУТАТЬ: [0:a] это оригинал!
    mux_in = ["-i", conv_wav]
    filt = "[1:a]aresample=44100,aformat=channel_layouts=stereo[voice]"
    if job.get("music", True):
        import soundfile as _sf
        import numpy as _np
        mux_in += ["-i", mus44]
        vv, _ = _sf.read(conv_wav)
        mm, _ = _sf.read(mus44)
        # фон относительно голоса: песня - музыка важна (0.8), речь - фон тише (0.4)
        target = 0.8 if is_song else 0.4
        ratio = (float(_np.sqrt((vv ** 2).mean())) * target) / (float(_np.sqrt((mm ** 2).mean())) + 1e-9)
        vol = ratio * float(job.get("music_vol", 1.0))
        log(f"[cover] баланс: голос {float(_np.sqrt((vv ** 2).mean())):.3f}, фон x{vol:.2f}")
        filt += f";[2:a]aresample=44100,aformat=channel_layouts=stereo,volume={vol:.3f}[bg];[voice][bg]amix=inputs=2:normalize=0[aout]"
    else:
        filt += ";[voice]acopy[aout]"
    out_mp4 = job.get("out") or os.path.splitext(job["input"])[0] + "_cover.mp4"
    run(["ffmpeg", "-y", "-i", job["input"], *mux_in, "-filter_complex", filt,
         "-map", "0:v:0", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac",
         "-shortest", out_mp4])
    log(f"[cover] ГОТОВО: {out_mp4}")
    try:
        import json as _json
        with open(out_mp4 + ".log", "w", encoding="utf-8") as _f:
            _f.write(_json.dumps(
                {"voice": job.get("voice"), "transpose": transpose, "params": params,
                 "music": job.get("music", True), "music_vol": job.get("music_vol", 1.0),
                 "src_median_f0": round(stats["median"], 1)},
                ensure_ascii=False, indent=1))
    except Exception:
        pass
    return out_mp4


def main():
    import argparse
    ap = argparse.ArgumentParser(description="SkySoCover: RVC-даббинг видео")
    ap.add_argument("input")
    ap.add_argument("--voice", required=True)
    ap.add_argument("--transpose", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-music", action="store_true")
    ap.add_argument("--music-vol", type=float, default=1.0)
    args = ap.parse_args()
    run_job({"input": args.input, "voice": args.voice, "transpose": args.transpose,
             "out": args.out, "music": not args.no_music, "music_vol": args.music_vol},
            log=lambda m: print(str(m).encode("cp1251", "backslashreplace").decode("cp1251"), flush=True))


if __name__ == "__main__":
    main()
