import os
import subprocess


def test_cli_basic(modelroot, tmp_path):
    src = (modelroot / 'BoxTextured.gltf').to_os_specific()
    dst = tmp_path / 'tmp.bam'
    subprocess.check_call([
        'gltf2bam',
        src,
        dst,
    ])

    assert os.path.exists(dst)
