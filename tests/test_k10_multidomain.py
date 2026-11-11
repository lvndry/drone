import copy

import pytest

from experiments.k10_multidomain import validate_rows
from tools.compare_k10_multidomain import compare


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


def paired_runs():
    control = dict(
        status="complete",
        arm="control",
        seed=13,
        config={},
        source_sha256={},
        feature_manifest_sha256="same",
        validation_ids=["beatport:3", "fmak:4"],
        fit_ids=["beatport:1"],
        calibration_used=False,
        final_holdout_used=False,
        domain_metrics={
            d: dict(ids=[d + ":val"], tracks=1, mirex=0.5) for d in ("beatport", "fmak")
        },
    )
    joint = copy.deepcopy(control)
    joint.update(arm="joint", fit_ids=["beatport:1", "fmak:2"])
    joint["domain_metrics"]["fmak"]["mirex"] = 0.53
    return control, joint


def test_gate_requires_a_comparable_completed_pair_and_both_domain_constraints():
    control, joint = paired_runs()
    assert compare(control, joint)["qualifies_for_confirmation"]
    joint["domain_metrics"]["beatport"]["mirex"] = 0.47
    assert not compare(control, joint)["qualifies_for_confirmation"]
    joint["status"] = "budget_exhausted"
    with pytest.raises(ValueError, match="complete"):
        compare(control, joint)


def test_gate_rejects_changed_validation_identities():
    control, joint = paired_runs()
    joint["validation_ids"].reverse()
    with pytest.raises(ValueError, match="validation_ids"):
        compare(control, joint)
