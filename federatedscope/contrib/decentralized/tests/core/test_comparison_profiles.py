"""The ``common_study`` comparison profile.

``common_study`` admits a contact trace in which a peer may meet two partners in one
frame (a G(n, p) trace, say); a pairwise method takes those sessions in event order, as
under ``paper_native``. Only ``common_matching_trace`` requires every frame to be a
matching. The methods' published fixed-graph profiles (fixed Q, fixed W) need
``paper_native`` and are refused under ``common_study``.
"""
import federatedscope.core.configs.config  # noqa: F401 - the global config first
import pytest

from federatedscope.contrib.configs.cfg_decentralized import COMPARISON_PROFILES
from federatedscope.contrib.decentralized.core.traces import (
    TraceValidationError, load_trace, validate_matching_profile)
from federatedscope.contrib.decentralized.tests.methods.test_dec_lora import dec_cfg
from federatedscope.contrib.decentralized.methods.dec_lora.config import PROFILE_PAPER_FIXED_Q


def _path_trace():
    return load_trace({"trace_schema_version": 1, "time_unit": "ms", "duration_unit": "ms",
                       "goodput_unit": "bps", "peers": [1, 2, 3],
                       "frames": [{"index": 0, "time": 0, "edges": [
                           {"endpoints": [1, 2], "duration": 1000, "goodput": 1_000_000},
                           {"endpoints": [2, 3], "duration": 1000, "goodput": 1_000_000}]}]})


def test_common_study_is_a_comparison_profile():
    assert "common_study" in COMPARISON_PROFILES
    cfg = dec_cfg(**{"decentralized.comparison.profile": "common_study"})
    assert cfg.decentralized.comparison.profile == "common_study"


def test_only_the_matching_profile_requires_a_matching():
    from federatedscope.contrib.decentralized.core.traces import profile_requires_matching

    with pytest.raises(TraceValidationError, match="requires a matching"):
        validate_matching_profile(_path_trace())
    assert profile_requires_matching("common_matching_trace")
    assert not profile_requires_matching("common_study")
    assert not profile_requires_matching("paper_native")


def test_the_papers_own_profiles_are_refused_under_common_study():
    with pytest.raises(Exception, match="reported under comparison.profile='common_study'"):
        dec_cfg(profile=PROFILE_PAPER_FIXED_Q,
                **{"decentralized.comparison.profile": "common_study",
                   "decentralized.methods.dec_lora.fixed_q": "ring"})
