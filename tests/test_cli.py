import os
import subprocess


import panda3d.core as p3d


def test_cli_basic(modelroot, tmp_path):
    src = (modelroot / 'BoxTextured.gltf').to_os_specific()
    dst = tmp_path / 'tmp.bam'
    subprocess.check_call([
        'gltf2bam',
        src,
        dst,
    ])

    assert os.path.exists(dst)


def test_cli_copy_textures(modelroot, tmp_path):
    src = (modelroot / 'BoxTextured.gltf').to_os_specific()
    dst = tmp_path / 'tmp.bam'
    subprocess.check_call([
        'gltf2bam',
        '--textures', 'copy',
        src,
        dst,
    ])

    assert os.path.exists(dst)
    assert os.path.exists(tmp_path / 'CesiumLogoFlat.png')

def test_cli_flatten_nodes(modelroot, tmp_path, showbase):
    # Load showbase to ensure using the loader does not mess with the type registry
    assert showbase

    src = (modelroot / 'Fox.glb').to_os_specific()
    dst = tmp_path / 'tmp.bam'
    subprocess.check_call([
        'gltf2bam',
        '--flatten-nodes',
        src,
        dst,
    ])

    loader = p3d.Loader.get_global_ptr()
    scene = p3d.NodePath(loader.load_sync(dst, p3d.LoaderOptions.LF_no_cache))

    panda_nodes = scene.find_all_matches('**/*/-PandaNode')
    assert not panda_nodes
