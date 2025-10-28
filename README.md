# glb-multipoint

Scripts and utilities for rendering GLB models from multiple viewpoints.

## Installation

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Upload your GLB files into the repository's `input_glbs/` directory. The helper
scripts read from this folder by default so you do not need to pass the file or
directory paths manually.

Render a single GLB file into a set of PNG images:

```bash
scripts/convert_glb.py --views 12 --image-size 1024 1024
```

You can still render a specific file by providing its path explicitly:

```bash
scripts/convert_glb.py path/to/model.glb --views 12 --image-size 1024 1024
```

Render every GLB file under a directory into a mirrored folder structure of
renders:

```bash
scripts/convert_glb_directory.py --output-dir renders --views 16
```

The renderer distributes cameras across a sphere using a Fibonacci spiral,
which provides good coverage of the model without large gaps between views.
Use the CLI flags to customise the render size, number of views, background
colour, camera radius, and lighting intensity.
