from setuptools import setup

__version__ = ''
#pylint: disable=exec-used
exec(open('gltf/version.py').read())

setup(
    version=__version__
)
