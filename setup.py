from setuptools import setup

__version__ = ''
#pylint: disable=exec-used
exec(open('gltf/version.py').read())

setup(
    version=__version__,
    keywords='panda3d gltf',
    packages=['gltf'],
    python_requires='>=3.6',
    install_requires=[
        'panda3d',
        'panda3d-simplepbr>=0.6',
    ],
    setup_requires=[
        'pytest-runner'
    ],
    tests_require=[
        'pytest',
        'pylint~=2.5.0',
        'pytest-pylint',
    ],
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
