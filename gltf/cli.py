import argparse

import panda3d.core as p3d

import gltf.converter
from gltf.version import __version__
from gltf.parseutils import parse_gltf_file


def main():
    parser = argparse.ArgumentParser(
        description='CLI tool to convert glTF files to Panda3D BAM files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('src', type=str, help='source file')
    parser.add_argument('dst', type=str, help='destination file')

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s {}'.format(__version__)
    )

    parser.add_argument(
        '--physics-engine',
        choices=[
            'builtin',
            'bullet',
        ],
        default='builtin',
        help='the physics engine to build collision solids for'
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

    args = parser.parse_args()

    settings = gltf.GltfSettings(
        physics_engine=args.physics_engine,
        print_scene=args.print_scene,
        skip_axis_conversion=args.skip_axis_conversion,
        no_srgb=args.no_srgb,
        textures=args.textures,
        legacy_materials=args.legacy_materials,
        skip_animations=args.animations == 'skip',
    )

    src = p3d.Filename(args.src)
    dst = p3d.Filename(args.dst)

    indir = p3d.Filename(src.get_dirname())
    outdir = p3d.Filename(dst.get_dirname())

    converter = Converter(indir=indir, outdir=outdir, settings=settings)
    gltf_data = parse_gltf_file(src)
    converter.update(gltf_data)

    if args.print_scene:
        converter.active_scene.ls()

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
