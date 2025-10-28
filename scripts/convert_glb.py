#!/usr/bin/env python3
"""Render a single GLB file from multiple viewpoints."""

from __future__ import annotations

import pathlib
import sys
from typing import Sequence

from glb_multipoint import renderer


DEFAULT_INPUT_DIR = pathlib.Path("input_glbs")


def resolve_default_glb() -> pathlib.Path:
    """Return the only GLB file inside :data:`DEFAULT_INPUT_DIR`.

    The helper mirrors the new repository layout where users upload their GLB
    assets to ``input_glbs``. Helpful errors guide them if the directory is
    missing, empty, or contains multiple files.
    """

    if not DEFAULT_INPUT_DIR.exists():
        raise SystemExit(
            f"Default GLB directory '{DEFAULT_INPUT_DIR}' does not exist. "
            "Create it or provide a GLB path explicitly."
        )

    candidates = sorted(p for p in DEFAULT_INPUT_DIR.glob("*.glb") if p.is_file())
    if not candidates:
        raise SystemExit(
            "No GLB files were found in the default directory. Place your model "
            f"inside '{DEFAULT_INPUT_DIR}' or pass the GLB path explicitly."
        )

    if len(candidates) > 1:
        joined = "\n - ".join(str(p) for p in candidates)
        raise SystemExit(
            "Multiple GLB files were found in the default directory. "
            "Specify which one to render explicitly. Found:\n - " + joined
        )

    return candidates[0]


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])

    if (not argv or argv[0].startswith("-")) and "-h" not in argv and "--help" not in argv:
        # No positional GLB argument was provided, so fall back to the
        # repository's default upload directory.
        glb_path = resolve_default_glb()
        argv = [str(glb_path), *argv]

    return renderer.main(argv)


if __name__ == "__main__":
    sys.exit(main())
