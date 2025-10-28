"""Rendering helpers for creating multi-point images from GLB models."""

from __future__ import annotations

import argparse
import dataclasses
import math
import pathlib
from typing import List, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
import pyrender
import trimesh


@dataclasses.dataclass
class RenderConfig:
    """Configuration options for rendering a GLB model.

    Attributes:
        views: Number of viewpoints to render around the model.
        image_size: Tuple containing the desired render width and height.
        radius: Distance of the camera from the scene origin.
        background: RGBA background color tuple with values in the range [0, 1].
        light_intensity: Strength of the directional lights in the scene.
    """

    views: int = 8
    image_size: Tuple[int, int] = (512, 512)
    radius: float = 3.0
    background: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.0)
    light_intensity: float = 3.0


Vector = Sequence[float]


def fibonacci_sphere(samples: int) -> List[np.ndarray]:
    """Generate `samples` points that are approximately evenly spaced on a sphere.

    The first view is always positioned on the positive Z axis so that the
    function produces deterministic camera arrangements.
    """

    if samples <= 0:
        raise ValueError("samples must be positive")

    points: List[np.ndarray] = []
    golden_angle = math.pi * (3 - math.sqrt(5.0))
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2 if samples > 1 else 0.0
        radius = math.sqrt(max(0.0, 1 - y * y))
        theta = golden_angle * i
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append(np.array([x, y, z], dtype=np.float32))

    if samples > 1:
        points[0] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    return points


def look_at(eye: Vector, target: Vector, up: Vector = (0.0, 1.0, 0.0)) -> np.ndarray:
    """Construct a camera pose matrix that looks from ``eye`` to ``target``."""

    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    forward = target - eye
    forward /= np.linalg.norm(forward)
    side = np.cross(forward, up)
    if np.linalg.norm(side) == 0:
        side = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        side /= np.linalg.norm(side)
    up_corrected = np.cross(side, forward)

    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 0] = side
    matrix[:3, 1] = up_corrected
    matrix[:3, 2] = -forward
    matrix[:3, 3] = eye
    return matrix


def _load_mesh(path: pathlib.Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Failed to load a mesh from {path}")
    return mesh


def render_glb_views(
    glb_path: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    config: RenderConfig | None = None,
) -> List[pathlib.Path]:
    """Render ``glb_path`` from multiple viewpoints and save the resulting images.

    Args:
        glb_path: Path to the GLB model that will be rendered.
        output_dir: Directory in which the rendered PNG files will be stored.
        config: Optional :class:`RenderConfig` instance to control the renders.

    Returns:
        A list containing the paths of the generated images in render order.
    """

    config = config or RenderConfig()
    glb_path = pathlib.Path(glb_path)
    output_dir = pathlib.Path(output_dir)

    if not glb_path.exists():
        raise FileNotFoundError(glb_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    mesh = _load_mesh(glb_path)
    scene = pyrender.Scene(bg_color=np.array(config.background, dtype=np.float32))
    mesh_node = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene.add(mesh_node)

    # Add simple lighting to give the renders some depth.
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=config.light_intensity)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([2.0, 4.0, 2.0])
    scene.add(light, pose=light_pose)

    renderer = pyrender.OffscreenRenderer(*config.image_size)

    camera_positions = fibonacci_sphere(config.views)
    generated: List[pathlib.Path] = []
    for index, direction in enumerate(camera_positions):
        eye = direction / np.linalg.norm(direction) * config.radius
        camera_pose = look_at(eye, target=(0.0, 0.0, 0.0))
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        cam_node = scene.add(camera, pose=camera_pose)
        try:
            color, _ = renderer.render(scene)
        finally:
            scene.remove_node(cam_node)
        output_path = output_dir / f"{glb_path.stem}_view_{index:03d}.png"
        imageio.imwrite(output_path, color)
        generated.append(output_path)

    renderer.delete()
    return generated


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("glb", type=pathlib.Path, help="Path to the input GLB file")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("renders"),
        help="Directory where rendered images will be written.",
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config = RenderConfig(
        views=args.views,
        image_size=tuple(args.image_size),
        radius=args.radius,
        background=tuple(args.background),
        light_intensity=args.light_intensity,
    )

    render_glb_views(args.glb, args.output_dir, config=config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
