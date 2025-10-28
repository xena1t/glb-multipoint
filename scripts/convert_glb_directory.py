#!/usr/bin/env python3
"""Render every GLB file in a directory from multiple viewpoints."""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Iterable

from glb_multipoint.renderer import RenderConfig, render_glb_views


def iter_glb_files(root: pathlib.Path) -> Iterable[pathlib.Path]:
    for path in root.rglob("*.glb"):
        if path.is_file():
            yield path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=pathlib.Path, help="Directory containing GLB files.")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("renders"),
        help="Directory where sub-folders of rendered images will be written.",
    )
    parser.add_argument("--views", type=int, default=8, help="Number of viewpoints to render.")
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=(512, 512),
        metavar=("WIDTH", "HEIGHT"),
        help="Pixel dimensions of the output images.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=3.0,
        help="Radius of the virtual camera orbit around the model.",
    )
    parser.add_argument(
        "--background",
        type=float,
        nargs=4,
        default=(1.0, 1.0, 1.0, 0.0),
        metavar=("R", "G", "B", "A"),
        help="Background color specified as RGBA floats between 0 and 1.",
    )
    parser.add_argument(
        "--light-intensity",
        type=float,
        default=3.0,
        help="Intensity of the directional light used in the scene.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.root.exists():
        raise SystemExit(f"Directory {args.root} does not exist")

    config = RenderConfig(
        views=args.views,
        image_size=tuple(args.image_size),
        radius=args.radius,
        background=tuple(args.background),
        light_intensity=args.light_intensity,
    )

    for glb_file in iter_glb_files(args.root):
        relative_dir = glb_file.relative_to(args.root).with_suffix("")
        output_dir = args.output_dir / relative_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        render_glb_views(glb_file, output_dir, config=config)

    return 0


if __name__ == "__main__":
    sys.exit(main())
