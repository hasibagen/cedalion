"""Placeholder tests for the Yiruide reader (implementation lands with T02)."""

import pytest

import cedalion.io


def test_read_yiruide_importable():
    assert hasattr(cedalion.io, "read_yiruide")
    assert hasattr(cedalion.io, "detect_version")


def test_read_yiruide_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        cedalion.io.read_yiruide("nope.csv")
