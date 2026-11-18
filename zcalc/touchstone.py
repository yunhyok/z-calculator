"""Touchstone 1.0 S-parameter files (.s2p for a single line, .s4p for a pair).

Layout follows the Touchstone File Format Specification, Version 2.0 (IBIS Open
Forum, 2009), rules for Version 1.0 files,
https://ibis.org/touchstone_ver2.0/touchstone_ver2_0.pdf : an option line
"# Hz S RI R <ohm>", one data line "f S11 S21 S12 S22" per frequency for two
ports, and the matrix row by row, four pairs per line, for four ports.

Tools disagree on how the four ports of a pair are numbered, so the file
states it in its comments and the caller chooses between the two usual maps.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np

# Position in the file -> port of engine.Result.s_parameters (1/2 are the near/far
# end of one line and 3/4 of the other).
PORT_ORDERS = {
    "1-2": (0, 1, 2, 3),  # through paths 1-2 and 3-4; near end is ports 1 and 3
    "1-3": (0, 2, 1, 3),  # through paths 1-3 and 2-4; near end is ports 1 and 2
}


def port_map(ports: int, order: str = "1-2") -> list[str]:
    """One comment line per port describing where it sits on the line."""
    if ports == 2:
        return ["Port 1: near end", "Port 2: far end"]
    ends = {0: "line + (near end)", 1: "line + (far end)", 2: "line - (near end)", 3: "line - (far end)"}
    return [f"Port {number}: {ends[source]}" for number, source in enumerate(PORT_ORDERS[order], start=1)]


def format_touchstone(freq_hz, s_parameters, port_ohm: float, comments=(), order: str = "1-2") -> str:
    """Text of a Touchstone 1.0 file in real/imaginary form.

    s_parameters has shape (frequencies, n, n) with n = 2 or 4, ports as in
    engine.Result. order renumbers the four ports of a pair (see PORT_ORDERS).
    Comments are written as ASCII; other characters become "?".
    """
    frequency = np.asarray(freq_hz, dtype=float)
    s = np.asarray(s_parameters, dtype=complex)
    ports = s.shape[-1]
    if s.shape != (frequency.size, ports, ports) or ports not in (2, 4):
        raise ValueError("Touchstone export needs a 2-port or 4-port matrix per frequency.")
    if not np.all(np.isfinite(s)) or not np.all(np.isfinite(frequency)) or np.any(np.diff(frequency) <= 0):
        raise ValueError("Touchstone export needs finite data at increasing frequencies.")
    if order not in PORT_ORDERS:
        raise ValueError(f"Unknown port order: {order}")
    if ports == 4:
        index = list(PORT_ORDERS[order])
        s = s[:, index][:, :, index]
    lines = [f"! {str(text).encode('ascii', 'replace').decode()}" for text in (*comments, *port_map(ports, order))]
    lines.append(f"# Hz S RI R {port_ohm:.12g}")
    pair = lambda value: f"{value.real:.12e} {value.imag:.12e}"
    if ports == 2:
        lines.append("! frequency  S11  S21  S12  S22  (real imaginary)")
        for f, m in zip(frequency, s):
            # Two-port files list the first column before the second.
            lines.append(" ".join([f"{f:.12e}", pair(m[0, 0]), pair(m[1, 0]), pair(m[0, 1]), pair(m[1, 1])]))
    else:
        lines.append("! frequency  S11 S12 S13 S14 / S21 ... S24 / S31 ... S34 / S41 ... S44  (real imaginary)")
        for f, m in zip(frequency, s):
            for row in range(ports):
                start = f"{f:.12e}" if row == 0 else " " * 18
                lines.append(" ".join([start, *(pair(value) for value in m[row])]))
    return "\n".join(lines) + "\n"


def save_touchstone(path: str | Path, freq_hz, s_parameters, port_ohm: float, comments=(), order: str = "1-2"):
    """Write a Touchstone file, preserving an existing file on failure."""
    text = format_touchstone(freq_hz, s_parameters, port_ohm, comments, order)
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{destination.stem}-", suffix=destination.suffix,
                                    dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(handle, "w", encoding="ascii", newline="\n") as stream:
            stream.write(text)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
