"""Load the model for inference."""

from __future__ import annotations

from pathlib import Path

import torch
from nemo.collections.asr.models import ASRModel

from haas_asr._kda import KDABackend, resolve_kda_backend, use_kda_backend

# Sentinel for load_model's default precision (None already means "as restored").
AUTO = "auto"


def resolve_dtype(
    dtype: torch.dtype | str | None, device: str | torch.device
) -> torch.dtype | None:
    """``"auto"``: bfloat16 on a GPU, float32 elsewhere. Other values pass through."""
    if dtype != AUTO:
        return dtype
    return torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32


def load_model(
    model: str | Path,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype | str | None = AUTO,
    kda_backend: KDABackend | None = None,
) -> ASRModel:
    """Load a ``.nemo`` file or a Hugging Face model id, in eval mode.

    `dtype` casts the whole model. The default, ``"auto"``, is bfloat16 on a
    GPU and float32 on the CPU: the published accuracy and speed were measured
    in bfloat16, and NeMo restores a model into float32 parameters even when
    the file holds bfloat16 weights, so without a cast the model would compute
    in float32 and give slightly different transcripts. ``None`` keeps the
    parameters as restored.

    `kda_backend` chooses the decoder's attention kernels (see ``haas_asr._kda``):
    ``"fla"`` (CUDA only), ``"torch"`` (any device) or ``"auto"``. ``None`` reads
    ``HAAS_KDA_BACKEND``, falling back to ``"auto"``.
    """
    backend = resolve_kda_backend(kda_backend, device)
    if backend == "fla" and torch.device(device).type != "cuda":
        raise ValueError(f"the fla KDA backend needs a CUDA device, got {device!r}")
    with use_kda_backend(backend):
        if Path(model).is_file():
            asr_model = ASRModel.restore_from(str(model), map_location=device)
        else:
            asr_model = ASRModel.from_pretrained(str(model), map_location=device)
    resolved = resolve_dtype(dtype, device)
    if resolved is not None:
        asr_model.to(dtype=resolved)
    return asr_model.eval()
