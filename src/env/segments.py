"""Segment checkpoints and per-segment enemy counts parsed from level text."""

from __future__ import annotations

from math import ceil

# Fixed by the level format (must equal ``smb.level.MarioLevel.seg_width`` and
# ``smb.level.LevelRender.tex_size``): a segment is 16 tiles wide at 16 px per
# tile, so checkpoints sit at x = 256, 512, 768, ... px.
SEGMENT_WIDTH_TILES = 16
TILE_SIZE_PX = 16

# Width assumed for empty/blank level text.
_FALLBACK_LEVEL_WIDTH_TILES = 100

# Tile characters counted as enemies: goomba, red/green koopa, spiky variants,
# generic enemy, bullet-bill.
_ENEMY_TILE_CHARS = frozenset({"g", "G", "r", "R", "k", "K", "y", "Y", "E", "*"})


def enemy_counts_per_segment_from_level_text(level_text: str) -> list[int]:
    """Count enemy tiles per 16-tile segment of ``level_text`` (at least one entry).
    Segment ``i`` spans columns ``[i * 16, min((i + 1) * 16, width))``; lines
    are only ``\\r``-stripped and short lines contribute what they have."""
    width = level_width_tiles_from_text(level_text)
    seg_width = SEGMENT_WIDTH_TILES
    num_segments = max(1, ceil(width / seg_width))
    lines = level_text.splitlines()
    counts: list[int] = []
    for seg_i in range(num_segments):
        col_start = seg_i * seg_width
        col_end = min((seg_i + 1) * seg_width, width)
        count = 0
        for line in lines:
            row = line.rstrip("\r")
            for col in range(col_start, min(col_end, len(row))):
                if row[col] in _ENEMY_TILE_CHARS:
                    count += 1
        counts.append(count)
    return counts


def level_width_tiles_from_text(level_text: str) -> int:
    """Return the level width in tiles from the first line of ``level_text``.
    Empty or whitespace-only text (or a blank first line) falls back to
    ``_FALLBACK_LEVEL_WIDTH_TILES``."""
    if not level_text.strip():
        return _FALLBACK_LEVEL_WIDTH_TILES
    first = level_text.splitlines()[0].strip()
    return len(first) if first else _FALLBACK_LEVEL_WIDTH_TILES


def segments_and_checkpoints_from_width(level_width_tiles: int) -> tuple[int, list[int]]:
    """Return ``(num_segments, checkpoint_x_positions_px)`` for a level width.
    The last segment's far end is also a checkpoint, so finishing a level can
    yield the final segment reward plus the win reward in the same step."""
    num_segments = max(1, ceil(level_width_tiles / SEGMENT_WIDTH_TILES))
    checkpoints_px = [TILE_SIZE_PX * SEGMENT_WIDTH_TILES * (i + 1) for i in range(num_segments)]
    return num_segments, checkpoints_px


def segments_and_checkpoints_from_level_text(level_text: str) -> tuple[int, list[int]]:
    """Return ``(num_segments, checkpoint_x_positions_px)`` for ``level_text``."""
    return segments_and_checkpoints_from_width(level_width_tiles_from_text(level_text))
