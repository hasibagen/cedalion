"""Tests for the Yiruide reader.

依赖外部 demo 数据的用例通过环境变量定位夹具（不在时跳过）：
- YIRUIDE_TEST_DATA：三份 demo CSV 所在目录
- YIRUIDE_POS49：49 通道 pos 文件
纯函数用例（marks_to_events、detect_version）不依赖外部文件。
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

import cedalion.io
from cedalion.io import yiruide

TEST_DATA = Path(os.environ.get(
    "YIRUIDE_TEST_DATA",
    "/home/mind/nut/teamwork/chongyi/code/report/test_data"))
POS49 = Path(os.environ.get(
    "YIRUIDE_POS49",
    "/home/mind/nut/yiruide/roi/49/1代/4、pos文件_csv/49.csv"))
requires_data = pytest.mark.skipif(
    not TEST_DATA.exists() or not POS49.exists(),
    reason="demo 数据或 pos 文件不在本机")

# (文件, n_times, trial_types)
CASES = [
    ("vft_demo_123.csv", 1520, ["1"]),
    ("nback_demo.csv", 1710, ["1"]),
    ("rest_demo.csv", 3900, ["1", "2"]),
]


def test_marks_to_events_paired():
    t = np.arange(6.0)
    evs = yiruide.marks_to_events(["", "A0", "", "", "A1", ""], t, {"A": 30.0})
    assert [(e.name, e.onset, e.duration) for e in evs] == [("A", 1.0, 3.0)]


def test_marks_to_events_instantaneous():
    t = np.arange(4.0)
    evs = yiruide.marks_to_events(["", "1", "", "2"], t, {})
    assert [(e.name, e.onset, e.duration) for e in evs] == [
        ("1", 1.0, 0.0), ("2", 3.0, 0.0)]


def test_marks_to_events_unclosed_uses_default_duration():
    t = np.arange(3.0)
    evs = yiruide.marks_to_events(["B0", "", ""], t, {"B": 12.0})
    assert [(e.name, e.onset, e.duration) for e in evs] == [("B", 0.0, 12.0)]


def test_detect_version():
    small = np.full((1, 1, 10), 0.9)
    big = np.full((1, 1, 10), 5000.0)
    assert yiruide.detect_version(small) == "v1"
    assert yiruide.detect_version(big) == "v2"
    assert cedalion.io.detect_version is yiruide.detect_version


def test_read_yiruide_requires_explicit_pos():
    # pos_path 必传，无自动匹配（多代帽子同通道数，自动匹配不可靠）
    with pytest.raises(TypeError):
        yiruide.read_yiruide("whatever.csv")


@requires_data
@pytest.mark.parametrize("name,n_times,trial_types", CASES)
def test_read_yiruide_demo(name, n_times, trial_types):
    rec = yiruide.read_yiruide(TEST_DATA / name, pos_path=POS49)
    amp = rec["amp"]
    assert amp.dims == ("channel", "wavelength", "time")
    assert amp.sizes["channel"] == 49
    assert amp.sizes["wavelength"] == 2
    assert amp.sizes["time"] == n_times
    labels = list(amp.channel.values)
    assert len(set(labels)) == 49
    assert all(re.fullmatch(r"S\d+D\d+", lab) for lab in labels)
    assert list(amp.wavelength.values) == [690.0, 830.0]
    assert float(rec.meta_data["fs"]) == 10.0
    assert rec.meta_data["ManufacturerName"] == "Yiruide"
    assert rec.meta_data["LengthUnit"] == "mm"
    assert sorted(rec.stim.trial_type.unique()) == sorted(trial_types)
    assert yiruide.detect_version(amp) == "v1"  # demo 为归一化光强
    # geo3d：光源 + 探测器 + landmark，mm
    geo = rec.geo3d.pint.dequantify()
    assert geo.sizes["pos"] == 3
    assert "S1" in geo.label.values and "D1" in geo.label.values
    # vendor_channel 坐标保留设备通道号，且与 S#D# 标签一一对应
    vc = list(amp.coords["vendor_channel"].values)
    assert vc == sorted(vc) and len(set(vc)) == 49


@requires_data
def test_source_header_roundtrip():
    rec = yiruide.read_yiruide(TEST_DATA / "rest_demo.csv", pos_path=POS49)
    header = json.loads(rec.meta_data["SourceHeader"])
    assert header["meta_list"]["Wave[nm]"] == ["690", "830"]
    assert "Sampling Period[s]" in header["meta_list"]
