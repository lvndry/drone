import copy

import pytest

from experiments.k10_multidomain import validate_rows


def rows():
    return [
        dict(id="beatport:1", domain="beatport", role="train", sha256="a", ref=0),
        dict(id="fmak:2", domain="fmak", role="train", sha256="b", ref=1),
        dict(id="fmak:3", domain="fmak", role="validation", sha256="c", ref=2),
    ]


@pytest.mark.parametrize("role", ["calibration", "final", "test"])
def test_preparation_refuses_non_development_roles(role):
    data = rows()
    data[-1]["role"] = role
    with pytest.raises(ValueError, match="Only fitting and validation"):
        validate_rows(data)


def test_development_preparation_refuses_audio_leakage_and_duplicate_fit_rows():
    data = rows()
    validate_rows(data)
    leaked = copy.deepcopy(data)
    leaked[-1]["sha256"] = leaked[0]["sha256"]
    with pytest.raises(ValueError, match="crosses"):
        validate_rows(leaked)
    duplicated = copy.deepcopy(data)
    duplicated[1]["sha256"] = duplicated[0]["sha256"]
    with pytest.raises(ValueError, match="unique"):
        validate_rows(duplicated)
