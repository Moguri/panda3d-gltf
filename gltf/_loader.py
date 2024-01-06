import dataclasses

import panda3d.core as p3d

from ._converter import GltfSettings, Converter
from .parseutils import parse_gltf_file


def load_model(file_path, gltf_settings=None):
    '''Load a glTF file from file_path and return a ModelRoot'''
    converter = Converter(file_path, settings=gltf_settings)
    gltf_data = parse_gltf_file(file_path)
    converter.update(gltf_data)
    return converter.active_scene.node()


def _config_var_for_type(var_type):
    return {
        'str': p3d.ConfigVariableString,
        'bool': p3d.ConfigVariableBool,
        'int': p3d.ConfigVariableInt,
    }.get(var_type, None)


class GltfLoader:
    # Loader metadata
    name = 'glTF'
    extensions = ['gltf', 'glb']
    supports_compressed = False

    @staticmethod
    def load_file(path, _options, _record=None):
        settings = GltfSettings()
        for field in dataclasses.fields(settings):
            fname = 'gltf-' + field.name.replace('_', '-')
            config_type = _config_var_for_type(field.type)
            if config_type is None:
                raise RuntimeError(
                    f'Unknown type ({field.type}) for {fname}'
                )
            default_value = getattr(settings, field.name)
            setattr(
                settings,
                field.name,
                config_type(fname, default_value).get_value()
            )

        return load_model(
            path,
            settings,
        )
