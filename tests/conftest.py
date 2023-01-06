import os

import pytest
import panda3d.core as p3d
from direct.showbase.ShowBase import ShowBase

@pytest.fixture(scope='session')
def showbase():
    prcdata = (
        'window-type none\n'
        'audio-library-name null\n'
        'model-cache-dir\n'
    )
    p3d.load_prc_file_data('', prcdata)
    base = ShowBase()

    type_registry = p3d.LoaderFileTypeRegistry.get_global_ptr()
    ftype = type_registry.get_type_from_extension('gltf')
    assert ftype.getName() == 'Python loader'

    return base


@pytest.fixture
def modelroot():
    return p3d.Filename.from_os_specific(
        os.path.join(
            os.path.dirname(__file__),
            'models',
        )
    )
