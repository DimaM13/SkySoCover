"""RMVPE F0 через бандловый rmvpe.onnx (end-to-end: waveform -> F0, 10мс).
Без torch-модели и без м Becа вручную - все внутри onnx."""
import os
import numpy as np

ONNX_PATH = r"D:\MMVCServerSIO\pretrain\rmvpe.onnx"

_session = None


def _session_get():
    global _session
    if _session is None:
        import onnxruntime as ort
        if not os.path.exists(ONNX_PATH):
            raise FileNotFoundError(f"нет {ONNX_PATH}")
        _session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    return _session


def infer(audio_16k: np.ndarray, thred: float = 0.03) -> np.ndarray:
    """16k моно float32 -> f0 по 10мс (0 = нет голоса)."""
    sess = _session_get()
    x = np.ascontiguousarray(audio_16k, dtype=np.float32)[np.newaxis, :]
    th = np.array([thred], dtype=np.float32)
    out = sess.run(["pitchf"], {"waveform": x, "threshold": th})[0]
    return np.ascontiguousarray(out.ravel(), dtype=np.float32)
