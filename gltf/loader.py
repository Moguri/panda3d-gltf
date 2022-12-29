import panda3d.core as p3d

from .converter import GltfSettings, Converter
from .parseutils import parse_gltf_file


def load_model(file_path, gltf_settings=None):
    '''Load a glTF file from file_path and return a ModelRoot'''
    if gltf_settings is None:
        gltf_settings = GltfSettings()

    if not isinstance(file_path, p3d.Filename):
        file_path = p3d.Filename.from_os_specific(file_path)

    workdir = p3d.Filename(file_path.get_dirname())

    p3d.get_model_path().prepend_directory(workdir)
    converter = Converter(indir=workdir, outdir=workdir, settings=gltf_settings)

    gltf_data = parse_gltf_file(file_path)
    converter.update(gltf_data)

    return converter.active_scene.node()


class GltfLoader:
    # Loader metadata
    name = 'glTF'
    extensions = ['gltf', 'glb']
    supports_compressed = False

    # Global loader options
    global_settings = GltfSettings()

    @staticmethod
    def load_file(path, _options, _record=None):
        return load_model(
            path,
            gltf_settings=GltfLoader.global_settings,
        )
