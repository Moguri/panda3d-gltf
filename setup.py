from setuptools import setup

__version__ = ''
#pylint: disable=exec-used
exec(open('gltf/version.py').read())

setup(
    version=__version__,
    entry_points={
        'console_scripts': [
            'gltf2bam=gltf.cli:main'
        ],
        'gui_scripts': [
            'gltf-viewer=gltf.viewer:main'
        ],
        'panda3d.loaders': [
            'gltf=gltf.loader:GltfLoader',
            'glb=gltf.loader:GltfLoader',
        ],
    },
)
