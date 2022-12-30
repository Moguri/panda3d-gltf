import panda3d.core as p3d

import gltf

def test_load_glb(modelroot):
    model = gltf.load_model(p3d.Filename(modelroot, 'Fox.glb'))

    assert model
