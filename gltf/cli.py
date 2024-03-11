import argparse
import os
import shutil

import panda3d.core as p3d

from . import GltfSettings
from ._converter import Converter
from .version import __version__
from .parseutils import parse_gltf_file


def main():
    parser = argparse.ArgumentParser(
        description='CLI tool to convert glTF files to Panda3D BAM files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        'src',
        type=str,
        help='source file'
    )
    parser.add_argument(
        'dst',
        type=str,
        nargs='?',
        default='',
        help='destination file',
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )

    parser.add_argument(
        '--collision-shapes',
        choices=[
            'builtin',
            'bullet',
        ],
        default='builtin',
        help='the collision system to build shapes for'
    )

    parser.add_argument(
        '--print-scene',
        action='store_true',
        help='print the converted scene graph to stdout'
    )

    parser.add_argument(
        '--skip-axis-conversion',
        action='store_true',
        help='do not perform axis-conversion (useful if glTF data is already Z-Up)'
    )

    parser.add_argument(
        '--no-srgb',
        action='store_true',
        help='do not load textures as sRGB textures'
    )

    parser.add_argument(
        '--textures',
        choices=[
            'ref',
            'copy',
        ],
        default='ref',
        help='control what to do with external textures (embedded textures will remain embedded)'
    )

    parser.add_argument(
        '--legacy-materials',
        action='store_true',
        help='convert imported PBR materials to legacy materials'
    )

    parser.add_argument(
        '--animations',
        choices=[
            'embed',
            'separate',
            'skip',
        ],
        default='embed',
        help='control what to do with animation data'
    )

    parser.add_argument(
        '--flatten-nodes',
        action='store_true',
        help='attempt to flatten resulting node structure'
    )

    args = parser.parse_args()

    settings = GltfSettings(
        collision_shapes=args.collision_shapes,
        skip_axis_conversion=args.skip_axis_conversion,
        no_srgb=args.no_srgb,
        legacy_materials=args.legacy_materials,
        skip_animations=args.animations == 'skip',
        flatten_nodes=args.flatten_nodes,
    )

    src = p3d.Filename.from_os_specific(args.src)
    src.make_absolute()

    if not args.dst:
        args.dst = args.src.rsplit('.', 1)[0] + '.bam'
    dst = p3d.Filename.from_os_specific(args.dst)
    dst.make_absolute()

    indir = p3d.Filename(src.get_dirname())
    outdir = p3d.Filename(dst.get_dirname())

    converter = Converter(src, settings=settings)
    gltf_data = parse_gltf_file(src)
    converter.update(gltf_data)

    os.makedirs(outdir, exist_ok=True)

    if args.print_scene:
        converter.active_scene.ls()

    if args.textures == 'copy':
        textures = [
            texture
            for scene in converter.scenes.values()
            for texture in scene.find_all_textures()
            if texture.filename
        ]

        for texture in textures:
            fname = texture.filename
            texsrc = os.path.join(indir.to_os_specific(), fname)
            texdst = os.path.join(outdir.to_os_specific(), fname)

            texture.fullpath = fname
            os.makedirs(os.path.dirname(texdst), exist_ok=True)
            shutil.copy(texsrc, texdst)

    if args.animations == 'separate':
        for bundlenode in converter.active_scene.find_all_matches('**/+AnimBundleNode'):
            anim_name = bundlenode.node().bundle.name
            anim_dst = dst.get_fullpath_wo_extension() \
                + f'_{anim_name}.' \
                + dst.get_extension()
            bundlenode.write_bam_file(anim_dst)

    converter.active_scene.write_bam_file(dst)


if __name__ == '__main__':
    main()
