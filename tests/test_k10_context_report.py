"""Completed context evidence must preserve failures and recorded checkpoint identities."""

from __future__ import annotations

import copy

import pytest

from tools.check_k10_context_report import checked_inputs, verify


def test_completed_context_evidence_reproduces_without_inference() -> None:
    result = verify(checked_inputs())
    assert result["gate"]["FMA_exact_gain"] == 1
    assert result["gate"]["technical_pair_valid"] is True
    assert result["gate"]["context_exposure_gate"] is False
    assert result["gate"]["historical_CPU320_noninferiority_guard"] is False
    assert result["gate"]["continuation_earned"] is False


def test_wrong_actual_checkpoint_receipt_rejected() -> None:
    inputs = copy.deepcopy(checked_inputs())
    inputs["postflight"]["checkpoints"]["65/best"]["sha256"] = "0" * 64
    with pytest.raises(AssertionError):
        verify(inputs)


def test_false_continuation_cannot_replace_failed_record() -> None:
    inputs = copy.deepcopy(checked_inputs())
    inputs["pair-result"]["gate"]["continuation_earned"] = True
    with pytest.raises(AssertionError):
        verify(inputs)
