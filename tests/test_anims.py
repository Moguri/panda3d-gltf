import panda3d.core as p3d
from direct.actor.Actor import Actor

import gltf

def test_simple_anim(modelroot):
    model = gltf.load_model(p3d.Filename(modelroot, 'Fox.glb'))
    assert model

    actor = Actor(p3d.NodePath(model))

    assert len(actor.get_anim_names()) == 3
