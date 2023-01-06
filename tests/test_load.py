import panda3d.core as p3d
from direct.showbase.ShowBase import ShowBase
import pytest #pylint:disable=wrong-import-order

#pylint:disable=redefined-outer-name


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
def modelpath(modelroot):
    return p3d.Filename(modelroot, 'test.gltf')

def test_load_single(showbase, modelpath):
    showbase.loader.load_model(modelpath)

def test_load_multiple(showbase, modelpath):
    showbase.loader.load_model([modelpath, modelpath])
    showbase.loader.load_model({modelpath, modelpath})
    # doesn't work on Panda3D 1.10.4+
    # showbase.loader.load_model((modelpath, modelpath))

def test_load_prc(showbase, modelpath):
    page = p3d.load_prc_file_data('', 'gltf-collision-shapes builtin')
    scene = showbase.loader.load_model(modelpath, noCache=True)
    p3d.unload_prc_file(page)
    assert scene.find('**/+CollisionNode')

    page = p3d.load_prc_file_data('', 'gltf-collision-shapes bullet')
    scene = showbase.loader.load_model(modelpath, noCache=True)
    p3d.unload_prc_file(page)
    assert scene.find('**/+BulletRigidBodyNode')
