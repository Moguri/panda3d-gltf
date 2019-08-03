import panda3d.core as p3d

from .converter import load_model, GltfSettings

class GltfLoader:
    # Loader metadata
    name = 'glTF'
    extensions = ['gltf', 'glb']
    supports_compressed = False

    # Global loader options
    global_settings = GltfSettings()

    @staticmethod
    def load_file(path, options, _record=None):
        loader = p3d.Loader.get_global_ptr()
        return load_model(
            loader,
            path,
            gltf_settings=GltfLoader.global_settings,
            options=options
        )
