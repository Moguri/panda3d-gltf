![Pipeline](https://github.com/Moguri/panda3d-gltf/workflows/Pipeline/badge.svg)
[![License](https://img.shields.io/github/license/Moguri/panda3d-gltf.svg)](https://choosealicense.com/licenses/bsd-3-clause/)

# panda3d-gltf
This project adds glTF loading capabilities to Panda3D.
One long-term goal for this project is to be used as a reference for adding a builtin, C++ glTF loader to Panda3D.
If and when Panda3D gets builtin support for glTF, this module will go into maintenance mode and be used to backport glTF support to older versions of Panda3D.

## Features
* Adds support for native loading of glTF files
* Supports glTF 2.0
* Supports binary glTF
* Includes support for the following extensions:
  * KHR_lights (deprecated in favor of KHR_lights_punctual)
  * KHR_lights_punctual
  * BLENDER_physics
* Ships with a `gltf2bam` cli-tool for converting glTF files to BAM
* Ships with `gltf-viewer` for viewing files (including glTF) with a simple PBR renderer

## Installation

Use pip to install the `panda3d-gltf` package:

```bash
pip install panda3d-gltf
```

To grab the latest development build, use:

```bash
pip install git+https://github.com/Moguri/panda3d-gltf.git

```

## Usage

### Configuration

`panda3d-gltf` has the following configuration options.
See below for information on setting these options for the native loader and the CLI.

* `collision_shapes` - the type of collision shapes to build.
  Either `builtin` for `ColisionSolids` or `bullet` for `BulletRigidBodyNodes`.
  Defaults to `builtin`.
* `flatten_nodes` - attempt to flatten resulting scene graph, defaults to `False`
* `legacy_materials` - convert imported PBR materials to legacy materials, defaults to `False`
* `no_srgb` - do not load textures as sRGB textures, defaults to `False`
* `skip_animations` - do not convert animation data found in the glTF file, defaults to `False`
* `skip_axis_conversion` - do not perform axis-conversion (useful if glTF data is already non-standard and already Z-Up), defaults to `False`

### Native loading

`panda3d-gltf` ships with a Python file loader (requires Panda3D 1.10.4+), which seamlessly adds glTF support to Panda3D's `Loader` classes.
This *does not* add support to `pview`, which is a C++ application that does not support loading Python file loaders.
Instead of `pview`, use the `gltf-viewer` that ships with `panda3d-gltf`.

The loader can be configured via PRC variables.
These PRC variables are prefixed with `gltf-` but otherwise match the names above.
For example, use `gltf-collision-shapes bullet` to have the loader load Bullet shapes instead of CollisionSolids.

### Command Line

To convert glTF files to BAM via the command line, use the supplied `gltf2bam` tool:

```bash
gltf2bam source.gltf output.bam
```

See `gltf2bam -h` for more information on usage and available CLI flags.

### Viewer

`panda3d-gltf` ships with `gltf-viewer`.
This is a simple viewer (like `pview`) to view glTF (or any other file format support by Panda3D) with a simple, PBR renderer.

## Running Tests

First install `panda3d-gltf` in editable mode along with `test` extras:

```bash
pip install -e .[test]
```

Then run the test suite with `pytest`:

```bash
pytest
```

## Building Wheels

Install `build`:

```bash
pip install --upgrade build
```

and run:

```bash
python -m build
```

## License
[B3D 3-Clause](https://choosealicense.com/licenses/bsd-3-clause/)
