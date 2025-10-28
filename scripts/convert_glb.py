#!/usr/bin/env python3
"""Render a single GLB file from multiple viewpoints."""

from __future__ import annotations

import sys

from glb_multipoint.renderer import main


if __name__ == "__main__":
    sys.exit(main())
