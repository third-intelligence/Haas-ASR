"""Tests for the PyTorch KDA port and the backend choice. They run on the CPU.

The port is checked against fla's own pure-PyTorch reference (``fla/ops/kda/naive.py``),
which is loaded from its file because importing ``fla.ops`` needs Triton. With a CUDA
GPU, it is also checked against fla's layer itself.
"""

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from haas_asr import _kda
from haas_asr._kda import load_kda_class, resolve_kda_backend, use_kda_backend
from haas_asr._kda_torch import TorchKimiDeltaAttention, recurrent_kda
from haas_asr.prednet import KDAConfig, KDAPredictionNetwork, _LayerCache

# The released geometry: hidden 640, 5 heads of 64.
HIDDEN, HEADS, HEAD_DIM, CONV = 640, 5, 64, 4


def _load_fla_naive():
    spec = importlib.util.find_spec("fla")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("flash-linear-attention is not installed")
    path = Path(next(iter(spec.submodule_search_locations))) / "ops" / "kda" / "naive.py"
    naive_spec = importlib.util.spec_from_file_location("fla_kda_naive", path)
    naive = importlib.util.module_from_spec(naive_spec)
    naive_spec.loader.exec_module(naive)
    return naive


def _random_layer(seed: int = 0) -> TorchKimiDeltaAttention:
    torch.manual_seed(seed)
    layer = TorchKimiDeltaAttention(
        hidden_size=HIDDEN, head_dim=HEAD_DIM, num_heads=HEADS, conv_size=CONV, layer_idx=0
    ).eval()
    for p in layer.parameters():
        torch.nn.init.normal_(p, std=0.05)
    return layer


@pytest.mark.parametrize("reference", ["naive_recurrent_kda", "naive_chunk_kda"])
def test_recurrence_matches_fla_reference(reference):
    naive = _load_fla_naive()
    torch.manual_seed(0)
    batch, steps, heads, dim = 3, 64, HEADS, HEAD_DIM
    q, k, v, g = (torch.randn(batch, steps, heads, dim) for _ in range(4))
    beta = torch.randn(batch, steps, heads)
    a_log = torch.log(torch.empty(heads).uniform_(1, 16))
    dt_bias = torch.randn(heads * dim) * 0.1
    state = torch.randn(batch, heads, dim, dim) * 0.1  # fla's naive layout: [B, H, K, V]

    o, final = recurrent_kda(q, k, v, g, beta, a_log, dt_bias, state.transpose(-1, -2))

    # The naive reference takes the activations already applied.
    qn = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    kn = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    gate = -a_log.view(heads, 1).exp() * F.softplus(g + dt_bias.view(heads, dim))
    o_ref, final_ref = getattr(naive, reference)(
        qn, kn, v, gate, beta.sigmoid(), initial_state=state.clone(), output_final_state=True
    )
    torch.testing.assert_close(o, o_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(final.transpose(-1, -2), final_ref, atol=1e-4, rtol=1e-4)


def test_step_by_step_matches_full_sequence():
    layer = _random_layer()
    x = torch.randn(2, 20, HIDDEN)
    with torch.no_grad():
        full, _, _ = layer(x, past_key_values=_LayerCache(), use_cache=True)
        cache, outs = _LayerCache(), []
        for t in range(x.shape[1]):
            out, _, _ = layer(x[:, t : t + 1], past_key_values=cache, use_cache=True)
            outs.append(out)
    torch.testing.assert_close(torch.cat(outs, 1), full, atol=1e-5, rtol=1e-5)


def test_prednet_state_round_trips_with_the_port():
    config = KDAConfig(
        num_heads=HEADS, head_dim=HEAD_DIM, expand_v=1.0, conv_size=CONV, final_norm=True
    )
    with use_kda_backend("torch"):
        net = KDAPredictionNetwork(hidden_size=HIDDEN, num_layers=2, config=config).eval()
    assert isinstance(net.layers[0], TorchKimiDeltaAttention)
    y = torch.randn(12, 3, HIDDEN)  # time-first, as NeMo calls it
    with torch.no_grad():
        full, _ = net(y)
        state, outs = None, []
        for t in range(y.shape[0]):
            out, state = net(y[t : t + 1], state)
            outs.append(out)
    torch.testing.assert_close(torch.cat(outs, 0), full, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not _kda.fla_available(), reason="needs a CUDA GPU and Triton")
def test_port_matches_fla_layer_on_cuda():
    from fla.layers.kda import KimiDeltaAttention

    port = _random_layer().cuda()
    fla_layer = KimiDeltaAttention(
        hidden_size=HIDDEN, head_dim=HEAD_DIM, num_heads=HEADS, conv_size=CONV, layer_idx=0
    ).cuda()
    fla_layer.load_state_dict(port.state_dict())
    fla_layer.eval()
    x = torch.randn(2, 16, HIDDEN, device="cuda")
    with torch.no_grad():
        cache_port, cache_fla = _LayerCache(), _LayerCache()
        for t in range(x.shape[1]):  # one label per step, as in decoding
            o_port, _, _ = port(x[:, t : t + 1], past_key_values=cache_port, use_cache=True)
            o_fla, _, _ = fla_layer(x[:, t : t + 1], past_key_values=cache_fla, use_cache=True)
            torch.testing.assert_close(o_port, o_fla, atol=1e-3, rtol=1e-3)


def test_backend_precedence(monkeypatch):
    monkeypatch.setattr(_kda, "fla_available", lambda: True)
    monkeypatch.delenv(_kda.KDA_BACKEND_ENV, raising=False)
    assert resolve_kda_backend() == "fla"
    assert resolve_kda_backend(device="cuda:0") == "fla"
    assert resolve_kda_backend(device="cpu") == "torch"
    monkeypatch.setenv(_kda.KDA_BACKEND_ENV, "torch")
    assert resolve_kda_backend(device="cuda") == "torch"
    with use_kda_backend("fla"):
        assert resolve_kda_backend() == "fla"
        assert resolve_kda_backend("torch") == "torch"
    assert resolve_kda_backend() == "torch"


def test_auto_falls_back_to_the_port_without_fla(monkeypatch):
    monkeypatch.setattr(_kda, "fla_available", lambda: False)
    monkeypatch.delenv(_kda.KDA_BACKEND_ENV, raising=False)
    assert resolve_kda_backend() == "torch"
    assert resolve_kda_backend(device="cuda") == "torch"
    assert load_kda_class() is TorchKimiDeltaAttention


def test_unknown_backend_is_refused(monkeypatch):
    with pytest.raises(ValueError, match="triton"):
        resolve_kda_backend("triton")
    monkeypatch.setenv(_kda.KDA_BACKEND_ENV, "gpu")
    with pytest.raises(ValueError, match="gpu"):
        resolve_kda_backend()
