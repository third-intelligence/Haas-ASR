from nemo.core.classes import common
from nemo.core.classes.common import _is_target_allowed

import haas_asr  # noqa: F401  (importing registers the package)
from haas_asr._nemo import PACKAGE_PREFIX, allow_package_targets

# What exported .nemo configs name as the decoder's _target_. Must never change.
DECODER_TARGET = "haas_asr.prednet.KDATransducerDecoder"


def test_allowing_twice_adds_one_entry():
    allow_package_targets()
    assert common.ALLOWED_TARGET_PREFIXES.count(PACKAGE_PREFIX) == 1


def test_nemo_allows_the_decoder_target():
    assert _is_target_allowed(DECODER_TARGET)
