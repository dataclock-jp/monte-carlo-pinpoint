"""Shared Phase 2 primitives for the Pinpoint algorithm.

This module provides the deterministic hexagonal half-net dot stencil used by
both the public MCP tool (`mcp_server.py`) and the benchmark harness
(`benchmark/run_benchmark.py`). Centralizing it here prevents the kind of
drift that produced the bundle 2.14 → 2.15 paper-vs-code algorithmic gap, in
which the benchmark and the MCP entrypoint had diverged from each other and
from the paper's stated algorithm.

The hex stencil yields a deterministic geometric `s/2`-cover of the search
disk: any point inside the radius-`s` disk is within `s/2` of at least one
stencil dot. Combined with halving the search radius after each majority hit
in the caller's Phase 2 loop, this preserves the inductive containment
guarantee asserted by Theorem 1 of the paper.

Pure-function module: no I/O, no LLM calls, no configuration state. Both
callers retain ownership of the Phase 2 loop structure (current-center,
recentering, halving, max_rounds, majority voting).
"""

from __future__ import annotations

import math


# Outer-ring radius factor for the hex half-net: (sqrt(3) / 2).
# Rationale: the maximum distance from any point inside the unit disk to the
# nearest stencil dot is exactly 1/2, achieved at the angular midpoint
# between adjacent outer dots on the disk boundary.
_HEX_OUTER_FACTOR = math.sqrt(3.0) / 2.0


def hex_stencil_dots(
    center_x: float,
    center_y: float,
    radius: float,
) -> list[tuple[int, int]]:
    """Return the 7-point hexagonal half-net for a disk of the given radius.

    Layout: one center dot at (center_x, center_y), plus six outer dots at
    radius (sqrt(3) / 2) * `radius`, placed at 60-degree increments starting
    at angle 0 (pointing in the +x direction).

    All seven points lie inside the closed disk of radius `radius` centered
    at (center_x, center_y). Coordinates are rounded to the nearest integer
    for compatibility with screenshot-coordinate sampling, which is the
    convention used by the calling Phase 2 loops in mcp_server.py and
    benchmark/run_benchmark.py.

    Args:
        center_x: Disk center x-coordinate (screenshot coordinates).
        center_y: Disk center y-coordinate (screenshot coordinates).
        radius:   Disk radius. Caller is responsible for ensuring radius > 0.

    Returns:
        A list of seven (x, y) integer tuples in deterministic order:
        index 0 is the center; indices 1..6 are the outer ring in
        counter-clockwise order from angle 0.
    """
    outer_r = _HEX_OUTER_FACTOR * radius
    dots: list[tuple[int, int]] = [
        (int(round(center_x)), int(round(center_y)))
    ]
    for i in range(6):
        angle = math.radians(60.0 * i)
        dx = outer_r * math.cos(angle)
        dy = outer_r * math.sin(angle)
        dots.append((
            int(round(center_x + dx)),
            int(round(center_y + dy)),
        ))
    return dots


def clip_dots_to_bounds(
    dots: list[tuple[int, int]],
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
) -> list[tuple[int, int]]:
    """Clip dot coordinates to an inclusive-min, exclusive-max rectangle.

    Used by the Phase 2 callers to keep dots inside the screenshot, since
    the hex stencil may place outer dots outside the image when the center
    is near an edge. Dots whose clipped position duplicates an earlier dot
    are preserved as-is (de-duplication is the caller's responsibility, if
    any). The returned list has the same length and ordering as the input.
    """
    clipped: list[tuple[int, int]] = []
    for x, y in dots:
        cx = max(x_min, min(x_max - 1, int(x)))
        cy = max(y_min, min(y_max - 1, int(y)))
        clipped.append((cx, cy))
    return clipped
