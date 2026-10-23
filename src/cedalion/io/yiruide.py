"""Reading Yiruide (依瑞德) zlhk CSV exports.

Yiruide devices export continuous-wave data as CSV files (GBK encoded,
key-value header rows, a ``Probe1`` row describing the channel layout, and
trailing Mark/Time/BodyMovement/RemovalMark columns) plus a separate pos CSV
with optode coordinates.

This module is format-only, mirroring the role of :mod:`cedalion.io.nirs`:
it produces ``S#D#`` channel labels and keeps the original sampling rate.
Lab-specific conventions (CHn relabeling, decimation to 10 Hz, Chinese
metadata passthrough) live in the separate ``cedalion_ext`` package.

The implementation will be migrated from the validated ``zlhk2snirf.py``
reader (chongyi project, 869 lines, batch-validated on 1272 files); see the
cedalion_ext plan, task T02.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cedalion.dataclasses as cdc


def read_yiruide(
    csv_path: Union[str, Path], pos_path: Union[str, Path, None] = None
) -> cdc.Recording:
    """Read a Yiruide zlhk CSV export into a :class:`Recording`.

    Args:
        csv_path: path to the device-exported CSV file (GBK, headerless
            data rows)
        pos_path: path to the pos CSV with optode coordinates. If None the
            recording is built without geo3d coordinates.

    Returns:
        A :class:`Recording` with an ``amp`` timeseries
        (channel, wavelength, time) carrying ``S#D#`` channel labels at the
        original sampling rate, a stim dataframe built from the Mark column,
        and geo3d points if a pos file was given.

    Raises:
        NotImplementedError: skeleton only; the implementation lands with
            task T02 of the cedalion_ext plan.
    """
    raise NotImplementedError(
        "read_yiruide is a skeleton; migrate parse_csv/build_recording from "
        "zlhk2snirf.py here (cedalion_ext plan, task T02)"
    )


def detect_version(amplitudes) -> str:
    """Distinguish normalized ('v1') from raw-intensity ('v2') exports.

    v1 exports (normalized intensity) have a mean amplitude below 100.
    This is the single place where this distinction is made.

    Raises:
        NotImplementedError: skeleton only; lands with task T02.
    """
    raise NotImplementedError("land with task T02")
