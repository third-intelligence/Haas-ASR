import torch

from haas_asr.model import AUTO, resolve_dtype


def test_auto_is_bfloat16_on_gpu():
    assert resolve_dtype(AUTO, "cuda") is torch.bfloat16
    assert resolve_dtype(AUTO, "cuda:1") is torch.bfloat16
    assert resolve_dtype(AUTO, torch.device("cuda", 0)) is torch.bfloat16


def test_auto_is_float32_on_cpu():
    assert resolve_dtype(AUTO, "cpu") is torch.float32


def test_explicit_values_pass_through():
    assert resolve_dtype(torch.float16, "cuda") is torch.float16
    assert resolve_dtype(None, "cuda") is None
