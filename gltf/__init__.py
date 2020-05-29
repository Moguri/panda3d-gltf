import types

import panda3d.core as p3d

from .converter import load_model, GltfSettings
from .loader import GltfLoader
from .version import __version__


def patch_loader(loader, gltf_settings=None):
    '''Monkey patch the supplied Loader to add glTF support'''
    if gltf_settings is None:
        gltf_settings = GltfLoader.global_settings
    else:
        GltfLoader.global_settings = gltf_settings

    registry = p3d.LoaderFileTypeRegistry.get_global_ptr()
    if not hasattr(registry, 'register_type'):
        _load_model = loader.load_model

        def new_load_model(self, model_path, **kwargs):
            if not isinstance(model_path, (tuple, list, set)):
                model_path = [model_path]
            for model in model_path:
                fname = p3d.Filename(model)
                if fname.get_extension() in ('gltf', 'glb'):
                    return load_model(self, model, gltf_settings=gltf_settings, **kwargs)
                else:
                    return _load_model(model_path, **kwargs)
        loader.load_model = loader.loadModel = types.MethodType(new_load_model, loader)
    else:
        # Ensure that our loader is the preferred choice for .gltf files.
        ftype = registry.get_type_from_extension("gltf")
        if not ftype or ftype.type.name != 'PythonLoaderFileType':
            registry.register_type(GltfLoader)


__all__ = [
    'load_model',
]
