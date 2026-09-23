"""Hubert-признаки для RVC: веса fairseq hubert_base.pt (эталон, на нем учились
RVC-модели), сконверченные в формат transformers. НЕ Tencent-порт -
он оказался другой моделью (проверено численно).
"""
import os
import torch
from transformers import HubertModel, HubertConfig

LOCAL_PT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hubert_rvc.pt")

_model = None
_device = None


def get_hubert(device: str = "cuda:0"):
    global _model, _device
    if _model is None or _device != device:
        cfg = HubertConfig.from_pretrained("TencentGameMate/chinese-hubert-base")
        _model = HubertModel(cfg)
        torch.nn.utils.parametrize.remove_parametrizations(
            _model.encoder.pos_conv_embed.conv, tensor_name="weight")
        _model.load_state_dict(torch.load(LOCAL_PT, map_location="cpu"), strict=False)
        _model = _model.to(device).eval()
        _device = device
    return _model


@torch.no_grad()
def extract_features(wav_16k_1d: torch.Tensor, device: str = "cuda:0"):
    """wav (T,) float -> (1, T', 768) float. Шаг 320 сэмплов как в RVC."""
    model = get_hubert(device)
    out = model(wav_16k_1d.unsqueeze(0).to(device)).last_hidden_state
    return out
