"""Reading Yiruide (依瑞德) zlhk CSV exports.

Yiruide devices export continuous-wave data as CSV files (GBK encoded,
key-value header rows, a ``Probe1`` row describing the channel layout, and
trailing Mark/Time/BodyMovement/RemovalMark/PreScan columns) plus a separate
pos CSV with optode coordinates.

This module is format-only, mirroring the role of :mod:`cedalion.io.nirs`:
``read_yiruide`` returns a :class:`~cedalion.dataclasses.Recording` with an
``amp`` timeseries carrying ``S#D#`` channel labels at the original sampling
rate. Lab-specific conventions (CHn relabeling, decimation to 10 Hz, Chinese
metadata passthrough) live in the separate ``cedalion_ext`` package.

Provenance: migrated from the validated ``zlhk2snirf.py`` reader (chongyi
project; in turn derived from ``yiruide2snirf.py``, batch-validated on 1272
files). Conversion semantics are unchanged.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import pandas as pd
import xarray as xr

import cedalion.dataclasses as cdc
from cedalion.dataclasses import PointType

LOG = logging.getLogger("cedalion.io.yiruide")

MANUFACTURER = "Yiruide"
FORMAT_VERSION = "1.1"
TASK_NAMES = {"rest", "vft", "nback", "gonogo", "door", "oddball"}
TRAILING_COLS = ("Mark", "Time", "BodyMovement", "RemovalMark", "PreScan")

_CH_LABEL_RE = re.compile(r"^CH(\d+)\(([\d.]+)\)$")
_LANDMARK_RE = re.compile(r"^\[(.+)\]$")
_CH_ROW_RE = re.compile(r"^CH(\d+)$")
_ONOFF_RE = re.compile(r"^([A-Za-z]+)([01])$")

_STR = h5py.string_dtype(encoding="utf-8")


# --------------------------------------------------------------------------- #
# 基础读取：设备 CSV
# --------------------------------------------------------------------------- #
def read_text_lines(path: Path) -> list[str]:
    """按 gbk → utf-8-sig → utf-16 顺序尝试解码，需命中 pos/数据文件特征行。"""
    raw = path.read_bytes()
    for enc in ("gbk", "utf-8-sig", "utf-16"):
        try:
            text = raw.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
        if re.search(r"(命名|通道,X,Y,Z|^Probe1,|^CH\d)", text, re.M):
            return text.splitlines()
    raise ValueError(f"无法识别文件编码（gbk/utf-8/utf-16 均失败或内容不符）: {path}")


@dataclass
class RawCsv:
    """设备 CSV 解析结果。"""

    path: Path
    meta: dict[str, str] = field(default_factory=dict)      # 头段键值（首个值）
    meta_list: dict[str, list[str]] = field(default_factory=dict)  # 头段多值
    stim_time: dict[str, float] = field(default_factory=dict)  # 条件名 → 时长(s)
    ch_cols: list[tuple[int, int, float]] = field(default_factory=list)  # (列号, 通道号, λ)
    named_cols: dict[str, int] = field(default_factory=dict)  # 尾部命名列 → 列号
    data: np.ndarray | None = None        # (n_times, n_cols) float64；字符串列为 NaN
    str_cols: dict[int, list[str]] = field(default_factory=dict)  # 字符串列（Mark/Time）


def _to_float(v: str) -> float | None:
    try:
        return float(v)
    except ValueError:
        return None


def parse_csv(path: Union[str, Path]) -> RawCsv:
    """解析设备 CSV 的头段、Probe1 表头与数据行。"""
    path = Path(path)
    lines = read_text_lines(path)
    rc = RawCsv(path=path)

    # ---- 头段：直到 "Data" 行 ----
    i, n = 0, len(lines)
    while i < n:
        ln = lines[i].strip()
        if ln.split(",")[0].strip() == "Data":
            i += 1
            break
        parts = [p.strip() for p in ln.split(",")] if ln else []
        if not parts or not parts[0]:
            i += 1
            continue
        key, vals = parts[0], [p for p in parts[1:] if p != ""]
        if key == "Stim Time[s]":
            # 刺激定义在随后的首个非空行：A,51,B,15,...
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            if j < n:
                toks = [t.strip() for t in lines[j].split(",") if t.strip()]
                for k in range(0, len(toks) - 1, 2):
                    try:
                        rc.stim_time[toks[k]] = float(toks[k + 1])
                    except ValueError:
                        break
                i = j + 1
                continue
        if vals:
            rc.meta_list.setdefault(key, vals)
            rc.meta.setdefault(key, vals[0])
        i += 1

    # ---- Probe1 表头 ----
    while i < n and not lines[i].startswith("Probe1"):
        i += 1
    if i >= n:
        raise ValueError(f"未找到 Probe1 数据表头（不是设备数据 CSV？）: {path.name}")
    cols = [c.strip() for c in lines[i].split(",")]
    for idx, c in enumerate(cols):
        m = _CH_LABEL_RE.match(c)
        if m:
            rc.ch_cols.append((idx, int(m.group(1)), float(m.group(2))))
        elif c and idx > 0:
            rc.named_cols.setdefault(c, idx)
    if not rc.ch_cols:
        raise ValueError(f"Probe1 行未解析到任何 CHn(λ) 数据列: {path.name}")

    # ---- 数据行（首列为采样序号）。按列探测类型：
    #      数值列 → float（Mark/Time 可为字符串列，单独保留）----
    data_lines = [ln for ln in lines[i + 1:] if ln[:1].isdigit()]
    if not data_lines:
        raise ValueError(f"Probe1 之后没有数据行: {path.name}")
    tokens = [ln.split(",") for ln in data_lines]
    width = len(tokens[0])
    if any(len(r) != width for r in tokens):
        raise ValueError(f"数据行宽度不一致: {path.name}")
    if width != len(cols):
        raise ValueError(f"数据列数 {width} 与表头 {len(cols)} 不一致: {path.name}")
    str_allowed = {rc.named_cols.get("Mark"), rc.named_cols.get("Time")}
    rc.data = np.full((len(tokens), width), np.nan)
    for j in range(width):
        vals = [r[j].strip() for r in tokens]
        try:
            rc.data[:, j] = np.array([float(v) for v in vals])
        except ValueError:
            if j in str_allowed:
                rc.str_cols[j] = vals
            else:
                bad = next(v for v in vals if v and _to_float(v) is None)
                raise ValueError(
                    f"第 {j} 列({cols[j]})含非数值 {bad!r}: {path.name}"
                ) from None
    return rc


# --------------------------------------------------------------------------- #
# pos 文件（光极坐标）
# --------------------------------------------------------------------------- #
@dataclass
class Probe:
    """zlhk pos CSV 解析结果（索引已压缩为 1..n 连续编号）。"""

    path: Path
    mode_name: str | None = None
    layout_name: str | None = None           # "命名" 行，如 "106导"
    src_pos: np.ndarray | None = None        # (nS, 3) mm
    det_pos: np.ndarray | None = None        # (nD, 3) mm
    src_labels: list[str] = field(default_factory=list)   # 原始编号 "S11"...
    det_labels: list[str] = field(default_factory=list)   # "D11"...
    ch_map: dict[int, tuple[int, int]] = field(default_factory=dict)  # 通道号 → (新S, 新D)
    landmarks: list[tuple[str, list[float]]] = field(default_factory=list)


def _det_index(token: str) -> int:
    """探测器编号：字母 A-Z → 1-26（同 MATLAB），否则按数字。"""
    token = token.strip()
    if len(token) == 1 and token.isalpha():
        return ord(token.upper()) - ord("A") + 1
    return int(float(token))


def parse_pos(path: Union[str, Path]) -> Probe:
    """解析 pos CSV 的光源/探测器坐标与通道映射。"""
    path = Path(path)
    lines = read_text_lines(path)
    pr = Probe(path=path)
    src_xyz: dict[int, list[float]] = {}
    det_xyz: dict[int, list[float]] = {}
    ch_rows: dict[int, tuple[int, tuple, int, tuple]] = {}

    for ln in lines:
        parts = [p.strip() for p in ln.split(",")]
        key = parts[0]
        if key == "命名" and len(parts) > 1:
            pr.layout_name = parts[1]
        elif key == "原始CSV文件mode名" and len(parts) > 1:
            pr.mode_name = parts[1]
        elif (m := _LANDMARK_RE.match(key)) and len(parts) >= 4:
            try:
                pr.landmarks.append((m.group(1), [float(x) for x in parts[1:4]]))
            except ValueError:
                pass
        elif (m := _CH_ROW_RE.match(key)) and len(parts) >= 12:
            ch = int(m.group(1))
            s_idx = int(float(parts[4]))
            d_idx = _det_index(parts[8])
            src_xyz[s_idx] = [float(x) for x in parts[5:8]]
            det_xyz[d_idx] = [float(x) for x in parts[9:12]]
            ch_rows[ch] = (s_idx, src_xyz[s_idx], d_idx, det_xyz[d_idx])

    if not ch_rows:
        raise ValueError(f"pos 文件未解析到 CH 行: {path}")
    # 索引压缩重编号（保持升序），原编号保留在 labels
    s_old = sorted(src_xyz)
    d_old = sorted(det_xyz)
    s_map = {old: new for new, old in enumerate(s_old, start=1)}
    d_map = {old: new for new, old in enumerate(d_old, start=1)}
    pr.src_pos = np.array([src_xyz[k] for k in s_old], dtype=np.float64)
    pr.det_pos = np.array([det_xyz[k] for k in d_old], dtype=np.float64)
    pr.src_labels = [f"S{k}" for k in s_old]
    pr.det_labels = [f"D{k}" for k in d_old]
    pr.ch_map = {ch: (s_map[row[0]], d_map[row[2]]) for ch, row in ch_rows.items()}
    return pr


# --------------------------------------------------------------------------- #
# 数据整理：事件、meta、0 值修复
# --------------------------------------------------------------------------- #
def fix_zeros(mat: np.ndarray, name: str = "") -> np.ndarray:
    """MATLAB csv2nirs 的 0 值修复：首个 0 → 最近非零值；其余 0 → 前一采样值。

    ``mat`` 形状 (n_times, n_cols)，按列独立处理。
    """
    out = mat.copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        z = np.flatnonzero(col == 0.0)
        if z.size == 0:
            continue
        nz = np.flatnonzero(col != 0.0)
        if nz.size == 0:
            LOG.warning("%s 第 %d 列全为 0，跳过修复", name, j)
            continue
        first = z[0]
        col[first] = col[nz[np.argmin(np.abs(nz - first))]]
        for k in z[1:]:
            col[k] = col[k - 1]
    return out


@dataclass
class StimEvent:
    name: str
    onset: float
    duration: float


def marks_to_events(marks: list[str], tvec: np.ndarray,
                    stim_time: dict[str, float]) -> list[StimEvent]:
    """mark 字符串序列 → stim 事件列表。

    ``X0``/``X1`` 成对标记解析为带时长的事件（X0 起、X1 止）；ST/ED/数字等
    视为瞬时事件。未闭合事件按 ``stim_time`` 默认时长补齐。
    """
    events: list[StimEvent] = []
    open_ev: dict[str, StimEvent] = {}
    singles: dict[str, list[float]] = {}

    for mark, t in zip(marks, tvec):
        if mark in ("", "0", "0.0"):
            continue
        m = _ONOFF_RE.match(mark)
        if m and m.group(1).upper() not in ("ST", "ED"):  # X0/X1 成对条件
            cond, kind = m.group(1), m.group(2)
            if kind == "0":
                if cond in open_ev:  # 未闭合又开新事件：先按默认时长闭合
                    ev = open_ev.pop(cond)
                    ev.duration = stim_time.get(cond, 0.0)
                    events.append(ev)
                    LOG.warning("条件 %s 上一个 onset 未闭合，按默认时长 %.2gs 补齐", cond, ev.duration)
                open_ev[cond] = StimEvent(cond, float(t), 0.0)
            else:
                ev = open_ev.pop(cond, None)
                if ev is None:
                    LOG.warning("条件 %s 出现 offset(%s) 但无 onset，记为瞬时事件", cond, mark)
                    singles.setdefault(mark, []).append(float(t))
                else:
                    ev.duration = float(t) - ev.onset
                    events.append(ev)
        else:  # ST/ED/数字/其他：瞬时条件
            singles.setdefault(mark, []).append(float(t))

    for cond, ev in open_ev.items():  # EOF 未闭合
        ev.duration = stim_time.get(cond, 0.0)
        events.append(ev)
        LOG.warning("条件 %s 到记录结束未闭合，按默认时长 %.2gs 补齐", cond, ev.duration)
    for mark, onsets in singles.items():
        name = mark if mark.upper() not in ("ST", "ED") else mark.upper()
        for t in onsets:
            events.append(StimEvent(name, t, 0.0))

    order = {n: i for i, n in enumerate(dict.fromkeys(e.name for e in events))}
    events.sort(key=lambda e: (order[e.name], e.onset))
    return events


def _normalize_marks(rc: RawCsv) -> list[str]:
    """取 Mark 列并归一（"1.0"→"1"，0 → 空），字符串/数值列统一处理。"""
    mark_col = rc.named_cols.get("Mark")
    if mark_col is not None and mark_col in rc.str_cols:
        marks = [m.strip() for m in rc.str_cols[mark_col]]
    elif mark_col is not None:
        marks = [str(v) for v in rc.data[:, mark_col]]
    else:
        marks = []
    norm = []
    for mk in marks:
        fv = _to_float(mk) if mk else 0.0
        if fv is None:
            norm.append(mk)
        elif fv == 0:
            norm.append("")
        else:
            norm.append("%d" % fv if fv == int(fv) else mk)
    return norm or [""] * (0 if rc.data is None else len(rc.data))


def parse_datetime(rc: RawCsv) -> datetime | None:
    raw = rc.meta.get("Study Date / Time") or rc.meta.get("Date") or ""
    raw = raw.strip()
    if not raw or raw.lower() == "unknown":
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M",
                "%Y/%m/%d %H:%M", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    LOG.warning("无法解析测量时间 %r，MeasurementDate/Time 记为 unknown", raw)
    return None


def _build_tags(rc: RawCsv, probe: Probe, period: float) -> dict:
    """metaDataTags 构造（write_snirf 与 build_recording 共用同一份）。"""
    dt = parse_datetime(rc)
    subj = rc.meta.get("ID") or rc.path.stem
    if dt is not None:
        # 设备时钟为站点本地时间且未记录时区；MeasurementTime 写朴素时间以兼容
        # 主流读取器（MNE 不支持 +hh:mm 后缀），偏移量单独存 MeasurementTimezone
        dt_local = dt.astimezone()
        date_s, time_s = dt_local.strftime("%Y-%m-%d"), dt_local.strftime("%H:%M:%S")
        tz_s = dt_local.strftime("%z")
        tz_s = tz_s[:-2] + ":" + tz_s[-2:] if tz_s else "unknown"
    else:
        date_s = time_s = "unknown"
        tz_s = "unknown"
    task = next((s for s in rc.path.parts[::-1] if s.lower() in TASK_NAMES), "")
    tags = {
        "SubjectID": str(subj),
        "MeasurementDate": date_s,
        "MeasurementTime": time_s,
        "LengthUnit": "mm",
        "TimeUnit": "s",
        "FrequencyUnit": "Hz",
        "ManufacturerName": MANUFACTURER,
        "SourceFile": rc.path.name,
        "SourceFileVersion": rc.meta.get("File Version", "unknown"),
        "DeviceName": rc.meta.get("Header", "") or "unknown",
        "ProbeMode": rc.meta.get("Mode", "unknown"),
        "SubjectName": rc.meta.get("Name") or " ".join(
            filter(None, [rc.meta.get("Family Name", ""), rc.meta.get("First Name", "")])),
        "Sex": rc.meta.get("Sex", "unknown"),
        "Age": rc.meta.get("Age", "unknown"),
        "TaskComment": rc.meta.get("Comment") or rc.meta.get("Study Comment") or "",
        "CapLayout": probe.layout_name or "",
        "PosFile": probe.path.name,
        "SamplingPeriod": repr(period),
    }
    if task:
        tags["Task"] = task.lower()
    tags["MeasurementTimezone"] = tz_s  # MeasurementTime 为朴素本地时间的时区注记
    exc = rc.meta_list.get("Exception Ch") or []
    # 0/1 语义厂商未说明（不同批次全 0/全 1 都有），原样保留不做解释
    if exc:
        tags["ExceptionChRaw"] = ",".join(v.strip() for v in exc)
    return tags


def _resolve_columns(rc: RawCsv) -> tuple[list[float], list[int], dict[tuple[int, int], int]]:
    """波长列表、通道号升序列表与 (波长序号, 通道号) → CSV 列 映射。"""
    wave_tokens = rc.meta_list.get("Wave[nm]") or [str(w) for _, _, w in rc.ch_cols[:1]]
    wavelengths = [float(w) for w in wave_tokens]
    wl_index = {int(round(w)): i for i, w in enumerate(wavelengths)}
    col_of: dict[tuple[int, int], int] = {}
    for col, ch, wl in rc.ch_cols:
        key = int(round(wl))
        if key not in wl_index:
            raise ValueError(
                f"通道 {ch} 波长 {wl}nm 不在 Wave[nm]={wavelengths} 中: {rc.path.name}"
            )
        k = (wl_index[key], ch)
        if k in col_of:
            raise ValueError(f"通道 {ch} 波长 {wl} 重复出现: {rc.path.name}")
        col_of[k] = col
    ch_nums = sorted({ch for _, ch, _ in rc.ch_cols})
    for ch in ch_nums:
        for wi in range(len(wavelengths)):
            if (wi, ch) not in col_of:
                raise ValueError(
                    f"通道 {ch} 缺少第 {wi + 1} 个波长（{wavelengths[wi]}nm）的数据列: {rc.path.name}"
                )
    return wavelengths, ch_nums, col_of


# --------------------------------------------------------------------------- #
# 内存直建 cedalion Recording（不落盘）
# 逐件复刻 cedalion.io.snirf.read_snirf 的产物（read_nirs_element /
# read_data_elements / geometry_from_probe / stim_to_dataframe 四个构造件
# 的等价实现），输入全部来自进程内的 rc(RawCsv) + probe(Probe)，不经
# tmpfile、不做 ASCII 净化，中文 meta 直接保留。
# --------------------------------------------------------------------------- #
def build_recording(rc: RawCsv, probe: Probe):
    """rc + probe → (cedalion.Recording, marks_frames)。等价于
    write_snirf→tmpfile→cedalion.io.snirf.read_snirf 的产物。

    marks_frames: {stim 名: [帧号,...]}（原始采样率下）。
    """
    wavelengths, ch_nums, col_of = _resolve_columns(rc)
    n_ch, n_wl = len(ch_nums), len(wavelengths)
    order = [(wi, ch) for wi in range(n_wl) for ch in ch_nums]
    data = rc.data[:, [col_of[k] for k in order]]
    n_t = data.shape[0]
    period = float(rc.meta.get("Sampling Period[s]") or 0)
    if period <= 0:
        raise ValueError(f"Sampling Period[s] 缺失或非法: {rc.path.name}")
    tvec = np.arange(n_t, dtype=np.float64) * period

    # ---- marks → stim 事件 + 帧号 ----
    events = marks_to_events(_normalize_marks(rc), tvec, rc.stim_time)
    stim = pd.DataFrame(
        [{"onset": e.onset, "duration": e.duration, "value": 1.0, "trial_type": e.name}
         for e in events],
        columns=["onset", "duration", "value", "trial_type"]) if events else cdc.build_stim_dataframe()
    fs0 = 1.0 / period
    marks_frames = {}
    for e in events:
        marks_frames.setdefault(e.name, []).append(int(round(e.onset * fs0)))

    # ---- amp：等价 read_data_elements（channel, wavelength, time）----
    # 通道标签 S{n}D{m}（测量列表顺序 = 通道号升序，同 read_snirf）；
    # vendor_channel 保留设备通道号，供上层改 CHn 命名时使用。
    missing = [ch for ch in ch_nums if ch not in probe.ch_map]
    if missing:
        raise ValueError(
            f"通道 {missing} 不在 pos 文件 {probe.path.name} 中，请核对光极排布"
        )
    pairs = [probe.ch_map[ch] for ch in ch_nums]
    ts3d = np.zeros((n_ch, n_wl, n_t), dtype=np.float64)
    for i, (wi, ch) in enumerate(order):
        ts3d[ch_nums.index(ch), wi, :] = data[:, i]
    amp = xr.DataArray(
        ts3d,
        dims=["channel", "wavelength", "time"],
        coords={
            "time": ("time", tvec),
            "samples": ("time", np.arange(n_t)),
            "channel": [f"S{s}D{d}" for s, d in pairs],
            "vendor_channel": ("channel", list(ch_nums)),
            "source": ("channel", [f"S{s}" for s, d in pairs]),
            "detector": ("channel", [f"D{d}" for s, d in pairs]),
            "wavelength": wavelengths,
        },
        attrs={"units": "1"},
    ).pint.quantify().pint.quantify({"time": "s"})

    # ---- geo3d：等价 geometry_from_probe（光源/探测器/landmark，mm，crs="pos"）----
    src, det = probe.src_pos, probe.det_pos
    labels = ([f"S{i + 1}" for i in range(len(src))]
              + [f"D{i + 1}" for i in range(len(det))]
              + [n for n, _ in probe.landmarks])
    types = ([PointType.SOURCE] * len(src) + [PointType.DETECTOR] * len(det)
             + [PointType.LANDMARK] * len(probe.landmarks))
    positions = np.vstack([src, det, np.array([xyz for _, xyz in probe.landmarks] or np.zeros((0, 3)))])
    geo3d = xr.DataArray(
        positions, dims=["label", "pos"],
        coords={"label": labels, "type": ("label", types)},
        attrs={"units": "mm"},
    ).set_xindex("type").pint.quantify()   # 同 geometry_from_probe 结尾两步

    rec = cdc.Recording()
    rec["amp"] = amp
    rec.geo3d = geo3d
    rec.stim = stim
    rec.meta_data.update(_build_tags(rc, probe, period))
    rec.meta_data["fs"] = fs0
    return rec, marks_frames


def read_yiruide(
    csv_path: Union[str, Path],
    pos_path: Union[str, Path],
    fix_zeros: bool = False,
) -> cdc.Recording:
    """Read a Yiruide zlhk CSV export into a :class:`Recording`.

    The pos file must be passed explicitly: several cap generations can
    share the same channel count (e.g. the 49-ch 1st/2nd generation), so
    automatic matching is deliberately not offered.

    Args:
        csv_path: path to the device-exported CSV file (GBK, headerless
            data rows)
        pos_path: path to the pos CSV with optode coordinates
        fix_zeros: apply the MATLAB csv2nirs zero repair (first zero →
            nearest non-zero, later zeros → previous sample). Default off.

    Returns:
        A :class:`Recording` with an ``amp`` timeseries
        (channel, wavelength, time) carrying ``S#D#`` channel labels and a
        ``vendor_channel`` coord with the device channel numbers, at the
        original sampling rate (``meta_data['fs']``); a stim dataframe built
        from the Mark column; geo3d points from the pos file; and vendor
        metadata tags (Chinese values preserved).
    """
    rc = parse_csv(csv_path)
    probe = parse_pos(pos_path)
    if fix_zeros:
        wavelengths, ch_nums, col_of = _resolve_columns(rc)
        order = [(wi, ch) for wi in range(len(wavelengths)) for ch in ch_nums]
        cols = [col_of[k] for k in order]
        rc.data[:, cols] = fix_zeros(rc.data[:, cols], rc.path.name)
    rec, _marks = build_recording(rc, probe)
    # 原始头段键值原样打包（中文保留），供上层实验室语义使用；SNIRF 落盘不含
    rec.meta_data["SourceHeader"] = json.dumps(
        {"meta_list": rc.meta_list, "stim_time": rc.stim_time}, ensure_ascii=False)
    return rec


def detect_version(amplitudes) -> str:
    """Distinguish normalized ('v1') from raw-intensity ('v2') exports.

    v1 exports (normalized intensity) have a mean amplitude below 100.
    Accepts an NDTimeSeries (units are stripped) or an ndarray.
    """
    values = np.asarray(getattr(amplitudes, "data", amplitudes), dtype=np.float64)
    return "v1" if float(np.nanmean(values)) < 100 else "v2"


# --------------------------------------------------------------------------- #
# SNIRF 写入（设备 CSV → 厂商风格 SNIRF 文件）
# --------------------------------------------------------------------------- #
def _meta_group(h: h5py.File, tags: dict[str, str]) -> None:
    g = h.require_group("/nirs/metaDataTags")
    for k, v in tags.items():
        g.create_dataset(k, data=v, dtype=_STR)


def write_snirf(out_path: Path, rc: RawCsv, probe: Probe, *,
                apply_fix_zeros: bool = False) -> dict:
    """rc + probe → 写 SNIRF 文件；返回摘要信息（用于日志/批量汇总）。"""
    wavelengths, ch_nums, col_of = _resolve_columns(rc)

    missing = [ch for ch in ch_nums if ch not in probe.ch_map]
    if missing:
        raise ValueError(
            f"通道 {missing} 不在 pos 文件 {probe.path.name} 中，请核对光极排布"
        )

    # ---- 数据矩阵：波长主序 [ch1..chN@λ1, ch1..chN@λ2, ...] ----
    order = [(wi, ch) for wi in range(len(wavelengths)) for ch in ch_nums]
    cols = [col_of[k] for k in order]
    intensity = rc.data[:, cols].astype(np.float64)
    n_t = intensity.shape[0]
    n_zero = int(np.count_nonzero(intensity == 0.0))
    if apply_fix_zeros:
        intensity = fix_zeros(intensity, rc.path.name)

    # ---- 时间轴：优先规则网格；设备 Time 列若偏离则改用之 ----
    period = float(rc.meta.get("Sampling Period[s]") or 0)
    if period <= 0:
        raise ValueError(f"Sampling Period[s] 缺失或非法: {rc.path.name}")
    tvec = np.arange(n_t, dtype=np.float64) * period
    if "Time" in rc.named_cols:
        tdev = rc.data[:, rc.named_cols["Time"]]
        if np.all(np.isfinite(tdev)) and tdev.size >= 2:
            if np.max(np.abs(tdev - tvec)) > 0.51 * period:
                if np.all(np.diff(tdev) > 0):
                    LOG.warning("设备 Time 列与采样周期推算不一致，改用设备 Time: %s",
                                rc.path.name)
                    tvec = tdev
                else:
                    LOG.warning("设备 Time 列非单调，忽略之: %s", rc.path.name)

    # ---- mark / stim / meta ----
    marks = _normalize_marks(rc)
    events = marks_to_events(marks, tvec, rc.stim_time)
    tags = _build_tags(rc, probe, period)

    # ---- aux：原始 mark 编码 + 有信息的设备标注列 ----
    aux_channels: list[tuple[str, np.ndarray]] = []
    uniq = [m for m in dict.fromkeys(marks) if m != ""]
    code = {m: i + 1 for i, m in enumerate(uniq)}
    mark_vec = np.array([[float(code.get(m, 0))] for m in marks], dtype=np.float64)
    aux_channels.append(("Mark", mark_vec))
    if uniq:
        tags["MarkCodeMap"] = ";".join(f"{m}={c}" for m, c in code.items())
    for nm in ("BodyMovement", "RemovalMark", "PreScan"):
        if nm in rc.named_cols:
            v = rc.data[:, rc.named_cols[nm]]
            if np.any(v != 0):
                aux_channels.append((nm, v[:, None].astype(np.float64)))

    # ---- 写 HDF5 ----
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    with h5py.File(tmp, "w") as h:
        h.create_dataset("formatVersion", data=FORMAT_VERSION, dtype=_STR)
        _meta_group(h, tags)

        dg = h.require_group("/nirs/data1")
        dg.create_dataset("name", data="Raw CW intensity (wavelength-major)",
                          dtype=_STR)
        dg.create_dataset("dataTimeSeries", data=intensity,
                          compression="gzip", compression_opts=4, shuffle=True)
        dg.create_dataset("time", data=tvec, compression="gzip",
                          compression_opts=4, shuffle=True)
        for k, (wi, ch) in enumerate(order, start=1):
            s_new, d_new = probe.ch_map[ch]
            ml = dg.create_group(f"measurementList{k}")
            ml.create_dataset("sourceIndex", data=np.int32(s_new))
            ml.create_dataset("detectorIndex", data=np.int32(d_new))
            ml.create_dataset("wavelengthIndex", data=np.int32(wi + 1))
            ml.create_dataset("dataType", data=np.int32(1))       # Raw CW
            ml.create_dataset("dataTypeIndex", data=np.int32(1))

        by_name: dict[str, list[StimEvent]] = {}
        for ev in events:
            by_name.setdefault(ev.name, []).append(ev)
        for j, (name, evs) in enumerate(by_name.items(), start=1):
            sg = h.create_group(f"/nirs/stim{j}")
            sg.create_dataset("name", data=name, dtype=_STR)
            sg.create_dataset("data", data=np.array(
                [[e.onset, e.duration, 1.0] for e in evs], dtype=np.float64))
            sg.create_dataset(
                "dataLabels",
                data=np.array(["starttime", "duration", "value"], dtype=object),
                dtype=_STR)

        for j, (name, ts) in enumerate(aux_channels, start=1):
            ag = h.create_group(f"/nirs/aux{j}")
            ag.create_dataset("name", data=name, dtype=_STR)
            ag.create_dataset("dataTimeSeries", data=ts)
            ag.create_dataset("time", data=tvec)

        pg = h.require_group("/nirs/probe")
        pg.create_dataset("wavelengths", data=np.array(wavelengths, dtype=np.float64))
        pg.create_dataset("sourcePos3D", data=probe.src_pos)
        pg.create_dataset("detectorPos3D", data=probe.det_pos)
        pg.create_dataset("sourceLabels", data=np.array(probe.src_labels, dtype=object),
                          dtype=_STR)
        pg.create_dataset("detectorLabels", data=np.array(probe.det_labels, dtype=object),
                          dtype=_STR)
        if probe.landmarks:
            pg.create_dataset("landmarkPos3D", data=np.array(
                [xyz for _, xyz in probe.landmarks], dtype=np.float64))
            pg.create_dataset(
                "landmarkLabels",
                data=np.array([n for n, _ in probe.landmarks], dtype=object),
                dtype=_STR)
        pg.create_dataset("coordinateSystem", data="Other", dtype=_STR)
        pg.create_dataset(
            "coordinateSystemDescription",
            data=("Vendor (zlhk) flattened cap-surface layout in mm. "
                  "X: left-positive; Y: posterior-positive; Z: superior "
                  "(landmarks only, optodes lie on Z=0 plane). Not an "
                  "anatomical 3-D space; optode labels keep vendor numbering."),
            dtype=_STR)

    # ---- 回读校验 ----
    with h5py.File(tmp, "r") as h:
        dts = h["/nirs/data1/dataTimeSeries"]
        assert dts.shape == (n_t, len(order)), "回读形状不符"
        chk = np.linspace(0, n_t - 1, min(64, n_t)).astype(int)
        assert np.array_equal(dts[chk, :], intensity[chk, :]), "回读数值不符"
        assert h["/nirs/data1/time"].shape[0] == n_t
    tmp.replace(out_path)

    return {
        "out": str(out_path),
        "subject": str(rc.meta.get("ID") or rc.path.stem),
        "n_times": n_t,
        "n_channels": len(ch_nums),
        "n_measurements": len(order),
        "wavelengths": wavelengths,
        "sfreq": 1.0 / period,
        "stim": {k: len(v) for k, v in by_name.items()},
        "duration_s": float(tvec[-1]) if n_t else 0.0,
        "zeros_repaired": (n_zero if apply_fix_zeros else 0),
        "probe": probe.path.name,
    }
