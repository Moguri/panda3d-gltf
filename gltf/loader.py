from .converter import load_model, GltfSettings

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
