"""Choose the Kimi Delta Attention implementation the decoder is built with.

Both backends build the same layer, with the same parameters and cache:

- ``"fla"``: flash-linear-attention's Triton kernels (``haas_asr._fla``). CUDA only.
- ``"torch"``: the pure-PyTorch port (``haas_asr._kda_torch``). Runs anywhere,
  including the CPU and macOS.

``"auto"`` picks fla when a CUDA GPU and Triton are available, and the port otherwise.
The backend is fixed when the decoder is built, so choose it before loading a model:
pass ``kda_backend`` to ``haas_asr.load_model``, set ``HAAS_KDA_BACKEND``, or wrap
``ASRModel.restore_from`` in ``use_kda_backend()``.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal, cast, get_args

import torch

from haas_asr import _fla
from haas_asr._kda_torch import TorchKimiDeltaAttention

KDABackend = Literal["auto", "fla", "torch"]
KDA_BACKEND_ENV = "HAAS_KDA_BACKEND"

_requested: ContextVar[KDABackend | None] = ContextVar("haas_kda_backend", default=None)


def _validate(name: str) -> KDABackend:
    if name not in get_args(KDABackend):
        raise ValueError(f"unknown KDA backend {name!r}; expected one of {get_args(KDABackend)}")
    return cast(KDABackend, name)


def fla_available() -> bool:
    """Whether fla's kernels can run here: a CUDA GPU and an importable Triton."""
    return torch.cuda.is_available() and importlib.util.find_spec("triton") is not None


def resolve_kda_backend(
    backend: KDABackend | None = None, device: str | torch.device | None = None
) -> Literal["fla", "torch"]:
    """Turn a requested backend into ``"fla"`` or ``"torch"``.

    The first one set wins: `backend`, ``use_kda_backend()``, ``HAAS_KDA_BACKEND``,
    then ``"auto"``. For ``"auto"``, a `device` other than CUDA selects the port,
    because fla's kernels only take CUDA tensors.
    """
    name = _validate(backend or _requested.get() or os.environ.get(KDA_BACKEND_ENV) or "auto")
    if name != "auto":
        return name
    if device is not None and torch.device(device).type != "cuda":
        return "torch"
    return "fla" if fla_available() else "torch"


@contextmanager
def use_kda_backend(backend: KDABackend) -> Iterator[None]:
    """Build KDA layers with `backend` inside this block (e.g. around ``restore_from``)."""
    token = _requested.set(_validate(backend))
    try:
        yield
    finally:
        _requested.reset(token)


def load_kda_class() -> type:
    """Return the KDA layer class of the backend chosen by ``resolve_kda_backend()``."""
    if resolve_kda_backend() == "torch":
        return TorchKimiDeltaAttention
    return _fla.load_kda_class()
