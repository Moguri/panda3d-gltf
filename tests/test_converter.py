import panda3d.core as p3d
from direct.actor.Actor import Actor

import gltf


def load_test_asset(modelroot, assetname) -> p3d.NodePath:
    return p3d.NodePath(gltf.load_model(p3d.Filename(modelroot, assetname)))

def test_texture_external(modelroot):
    model = load_test_asset(modelroot, 'BoxTextured.gltf')

    textures = model.find_all_textures('CesiumLogoFlat')
    assert textures

    texture = textures[0]
    assert texture.filename == 'CesiumLogoFlat.png'

def test_texture_embedded(modelroot):
    model = load_test_asset(modelroot, 'BoxTexturedEmbed.gltf')

    textures = model.find_all_textures('gltf-embedded-0')
    assert textures

def test_simple_anim(modelroot):
    model = load_test_asset(modelroot, 'Fox.glb')
    assert model

    actor = Actor(p3d.NodePath(model))

    assert len(actor.get_anim_names()) == 3
