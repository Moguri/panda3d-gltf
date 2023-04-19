import panda3d.core as p3d
from direct.actor.Actor import Actor

import gltf


def load_test_asset(modelroot, assetname) -> p3d.NodePath:
    model = p3d.NodePath(gltf.load_model(p3d.Filename(modelroot, assetname)))
    assert model
    return model

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

def test_anim_simple(modelroot):
    model = load_test_asset(modelroot, 'Fox.glb')

    actor = Actor(p3d.NodePath(model))

    assert len(actor.get_anim_names()) == 3

def test_skin_no_joint_nodes(modelroot):
    model = load_test_asset(modelroot, 'Fox.glb')

    model.ls()
    assert not model.find_all_matches('**/_rootJoint')

def test_skin_char_root(modelroot):
    model = load_test_asset(modelroot, 'Fox.glb')
    model.ls()
    assert model.find_all_matches('**/+Character/+GeomNode')
    assert model.find_all_matches('**/+Character/+AnimBundleNode')
