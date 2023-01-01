from .converter import GltfSettings, Converter
from .parseutils import parse_gltf_file


def load_model(file_path, gltf_settings=None):
    '''Load a glTF file from file_path and return a ModelRoot'''
    converter = Converter(file_path, settings=gltf_settings)
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
