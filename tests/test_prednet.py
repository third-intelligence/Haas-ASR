"""CPU tests for the prednet's config handling and state layout (no fla needed)."""

import pytest
import torch

from haas_asr.prednet import KDAConfig, _LayerCache, _StateLayout

RELEASED_KEYS = {
    "kda_num_heads": 5,
    "kda_head_dim": 64,
    "kda_expand_v": 1.0,
    "kda_conv_size": 4,
    "kda_final_norm": True,
}


def test_config_reads_prefixed_keys_only():
    config = KDAConfig.from_prednet({"pred_hidden": 640, "dropout": 0.2, **RELEASED_KEYS})
    assert config == KDAConfig(num_heads=5, head_dim=64, expand_v=1.0, conv_size=4, final_norm=True)


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="kda_use_gate"):
        KDAConfig.from_prednet({**RELEASED_KEYS, "kda_use_gate": True})


def test_config_rejects_missing_keys():
    keys = dict(RELEASED_KEYS)
    del keys["kda_head_dim"]
    with pytest.raises(ValueError, match="kda_head_dim"):
        KDAConfig.from_prednet(keys)


@pytest.mark.parametrize(
    "layout",
    [
        # Recurrent state wider than the conv caches (the released geometry).
        _StateLayout(5, 64, 64, key_dim=320, value_dim=320, conv_size=4),
        # Conv caches wider, so the recurrent component carries padding.
        _StateLayout(1, 4, 4, key_dim=16, value_dim=16, conv_size=4),
    ],
)
def test_state_round_trips_through_cache(layout):
    num_layers, batch_size = 2, 3
    shape = (num_layers, batch_size, layout.width)
    state = (torch.randn(shape), torch.randn(shape))
    state[0][..., layout.recurrent_size :] = 0
    state[1][..., layout.conv_total :] = 0

    cache = layout.to_cache(state)
    assert cache[0]["recurrent_state"].shape == (
        batch_size,
        layout.num_v_heads,
        layout.head_k_dim,
        layout.head_v_dim,
    )
    restored = layout.from_cache(cache, num_layers, batch_size, "cpu", torch.float32)
    torch.testing.assert_close(restored, state)


def test_missing_cache_entries_become_zeros():
    layout = _StateLayout(1, 4, 4, key_dim=16, value_dim=16, conv_size=4)
    recurrent, conv = layout.from_cache(_LayerCache(), 2, 3, "cpu", torch.float32)
    assert not recurrent.any() and not conv.any()
