from __future__ import annotations

import base64
import collections
import itertools
import os
import math
import struct
import urllib.parse
import pprint # pylint: disable=unused-import

from dataclasses import dataclass

from panda3d.core import * # pylint: disable=wildcard-import
import panda3d.core as p3d
try:
    from panda3d import bullet
    HAVE_BULLET = True
except ImportError:
    HAVE_BULLET = False
from direct.stdpy.file import open # pylint: disable=redefined-builtin

from ._converter_maps import (
    ATTRIB_CONTENT_MAP,
    ATTRIB_NAME_MAP,
    COMPONENT_FORMT_STR_MAP,
    COMPONENT_NUM_MAP,
    COMPONENT_SIZE_MAP,
    COMPONENT_TYPE_MAP,
    MAG_FILTER_MAP,
    MIN_FILTER_MAP,
    PRIMITIVE_MODE_MAP,
    WRAP_MODE_MAP,
)


# Taken from Blender glTF exporter
PBR_WATTS_TO_LUMENS = 683

if LVector3 is LVector3f:
    CPTA_stdfloat = CPTA_float
    PTA_stdfloat = PTA_float
else:
    CPTA_stdfloat = CPTA_double
    PTA_stdfloat = PTA_double


@dataclass
class GltfSettings:
    collision_shapes: str = 'builtin'
    skip_axis_conversion: bool = False
    no_srgb: bool = False
    legacy_materials: bool = False
    skip_animations: bool = False
    flatten_nodes: bool = False
    animation_fps: int = 30


def get_extras(gltf_data):
    extras = gltf_data.get('extras', {})
    if not isinstance(extras, dict):
        # weird, but legal; fail silently for now
        return {}
    return extras

def vlerp(veca: p3d.LVector3, vecb: p3d.LVector3, factor: float) -> p3d.LVector3:
    return veca * (1.0 - factor) + vecb * factor


def slerp(quata: p3d.LQuaternion, quatb: p3d.LQuaternion, factor: float) -> p3d.LQuaternion:
    dot_product = quata.dot(quatb)

    if dot_product < 0.0:
        # take shorted path for negative dot product
        quata = -quata
        dot_product = -dot_product

    if dot_product > 0.9995:
        # close enough; just lerp
        retval = quata * (1.0 - factor) + quatb * factor
        retval.normalize()
        return retval

    # spherical linear interpolation
    theta0 = math.acos(dot_product)
    theta = factor * theta0
    sin_theta = math.sin(theta)
    sin_theta0 = math.sin(theta0)

    scale_quata = math.cos(theta) - dot_product * sin_theta / sin_theta0
    scale_quatb = sin_theta / sin_theta0

    return quata * scale_quata + quatb * scale_quatb


def get_next_time_index(currtime: float, time_buffer: 'list[float]') -> int:
    nextidx = 1
    nexttime = time_buffer[nextidx]
    while currtime > nexttime and nextidx < len(time_buffer):
        nextidx += 1
        nexttime = time_buffer[nextidx]

    return nextidx


def get_lerp_factor(currtime: float, lasttime: float, nexttime: float) -> float:
    return max((currtime - lasttime) / (nexttime - lasttime), 1)


@dataclass
class CharInfo:
    character: p3d.Character
    nodepath: p3d.NodePath
    jvtmap: 'dict[int, p3d.JointVertexTransform]'
    cvsmap: 'dict[tuple[int, str], p3d.CharacterVertexSlider]'

    def __init__(self, name: str):
        self.character = p3d.Character(name)
        self.nodepath = p3d.NodePath(self.character)
        self.jvtmap = {}
        self.cvsmap = {}


class Converter():


    def __init__(
            self,
            filepath,
            settings=None
    ):
        if not isinstance(filepath, Filename):
            filepath = Filename.from_os_specific(filepath)
        if settings is None:
            settings = GltfSettings()
        self.filepath = filepath
        self.filedir = Filename(filepath.get_dirname())
        self.settings = settings
        self.cameras = {}
        self.buffers = {}
        self.lights = {}
        self.textures = {}
        self.texture_stages = {}
        self.mat_states = {}
        self.mat_mesh_map = {}
        self.meshes = {}
        self.scenes = {}
        self.skeletons = {}
        self.joint_parents = {}
        self.characters = {}

        # Coordinate system transform matrix
        self.csxform = LMatrix4.convert_mat(CS_yup_right, CS_default)
        self.csxform_inv = LMatrix4.convert_mat(CS_default, CS_yup_right)
        self.compose_cs = CS_yup_right

        self._joint_nodes = set()

        # Scene props
        self.active_scene = NodePath(ModelRoot('default'))
        self.background_color = (0, 0, 0)
        self.active_camera = None

    def update(self, gltf_data):
        #pprint.pprint(gltf_data)

        skip_axis_conversion = (
            'extensionsUsed' in gltf_data and 'BP_zup' in gltf_data['extensionsUsed'] or
            self.settings.skip_axis_conversion
        )

        if skip_axis_conversion:
            self.csxform = LMatrix4.ident_mat()
            self.csxform_inv = LMatrix4.ident_mat()
            self.compose_cs = CS_zup_right

        # Convert data
        for buffid, gltf_buffer in enumerate(gltf_data.get('buffers', [])):
            self.load_buffer(buffid, gltf_buffer)

        for camid, gltf_cam in enumerate(gltf_data.get('cameras', [])):
            self.load_camera(camid, gltf_cam)

        if 'extensions' in gltf_data and 'KHR_lights' in gltf_data['extensions']:
            lights = gltf_data['extensions']['KHR_lights'].get('lights', [])
            for lightid, gltf_light in enumerate(lights):
                self.load_light(lightid, gltf_light)

        if 'extensions' in gltf_data and 'KHR_lights_punctual' in gltf_data['extensions']:
            lights = gltf_data['extensions']['KHR_lights_punctual'].get('lights', [])
            for lightid, gltf_light in enumerate(lights):
                self.load_light(lightid, gltf_light, punctual=True)

        for texid, gltf_tex in enumerate(gltf_data.get('textures', [])):
            self.load_texture(texid, gltf_tex, gltf_data)

        for matid, gltf_mat in enumerate(gltf_data.get('materials', [])):
            self.load_material(matid, gltf_mat)

        for skinid, gltf_skin in enumerate(gltf_data.get('skins', [])):
            self.load_skin(skinid, gltf_skin, gltf_data)

        for meshid, gltf_mesh in enumerate(gltf_data.get('meshes', [])):
            self.load_mesh(meshid, gltf_mesh, gltf_data)

        def get_node_transform(gltf_node):
            if 'matrix' in gltf_node:
                gltf_mat = LMatrix4(*gltf_node.get('matrix'))
            else:
                gltf_mat = LMatrix4(LMatrix4.ident_mat())
                if 'scale' in gltf_node:
                    gltf_mat.set_scale_mat(tuple(gltf_node['scale']))

                if 'rotation' in gltf_node:
                    rot_mat = LMatrix4()
                    rot = gltf_node['rotation']
                    quat = LQuaternion(rot[3], rot[0], rot[1], rot[2])
                    quat.extract_to_matrix(rot_mat)
                    gltf_mat *= rot_mat

                if 'translation' in gltf_node:
                    gltf_mat *= LMatrix4.translate_mat(*gltf_node['translation'])

            return TransformState.make_mat(self.csxform_inv * gltf_mat * self.csxform)

        def build_characters(nodeid):
            try:
                gltf_node = gltf_data['nodes'][nodeid]
            except IndexError:
                print(f"Could not find node with index: {nodeid}")
                return
            node_name = gltf_node.get('name', 'node'+str(nodeid))

            if nodeid in self.skeletons:
                skinid = self.skeletons[nodeid]
                charinfo = CharInfo(node_name)
                charinfo.character.set_transform(get_node_transform(gltf_node))
                self.build_character(charinfo, nodeid, gltf_data, recurse=True)
                self.characters[skinid] = charinfo

            for child_nodeid in gltf_node.get('children', []):
                build_characters(child_nodeid)

        # Build scenegraphs
        def add_node(root, gltf_scene, nodeid):
            try:
                gltf_node = gltf_data['nodes'][nodeid]
            except IndexError:
                print(f"Could not find node with index: {nodeid}")
                return

            skinid = self.skeletons.get(nodeid, None)
            charinfo = self.characters.get(skinid, None)
            scene_extras = get_extras(gltf_scene)
            node_name = gltf_node.get('name', 'node'+str(nodeid))
            if nodeid in self._joint_nodes and not nodeid in self.skeletons:
                # Handle non-joint children of joints, but don't add joints themselves
                for child_nodeid in gltf_node.get('children', []):
                    add_node(root, gltf_scene, child_nodeid)
                return

            if charinfo:
                # This node is the root of an animated character.
                panda_node = charinfo.character
                np = charinfo.nodepath
                np.reparent_to(root)
            else:
                panda_node = PandaNode(node_name)
                panda_node.set_transform(get_node_transform(gltf_node))
                np = root.attach_new_node(panda_node)

            if 'hidden_nodes' in scene_extras:
                if nodeid in scene_extras['hidden_nodes']:
                    panda_node = panda_node.make_copy()

            if 'mesh' in gltf_node:
                meshid = gltf_node['mesh']
                gltf_mesh = gltf_data['meshes'][meshid]
                mesh = self.meshes[meshid]

                charinfo = None
                if "skin" in gltf_node:
                    skinid = gltf_node["skin"]
                    charinfo = self.characters[skinid]

                # Does this mesh have weights, but are we not under a character?
                # If so, create a character just for this mesh.
                if gltf_mesh.get('weights') and not charinfo:
                    mesh_name = gltf_mesh.get('name', 'mesh'+str(meshid))
                    charinfo = CharInfo(mesh_name)
                    self.build_character(charinfo, nodeid, gltf_data, recurse=False)
                    self.combine_mesh_morphs(mesh, meshid, charinfo)
                    charinfo.nodepath.reparent_to(np)
                    charinfo.nodepath.attach_new_node(mesh)
                else:
                    np.attach_new_node(mesh)
                    if charinfo:
                        self.combine_mesh_skin(mesh, charinfo)
                        self.combine_mesh_morphs(mesh, meshid, charinfo)

            if 'camera' in gltf_node:
                camid = gltf_node['camera']
                cam = self.cameras[camid]
                np.attach_new_node(cam)
            if 'extensions' in gltf_node:
                light_ext = None
                has_light_ext = False
                if 'KHR_lights_punctual' in gltf_node['extensions']:
                    light_ext = 'KHR_lights_punctual'
                    has_light_ext = True
                elif 'KHR_lights' in gltf_node['extensions']:
                    light_ext = 'KHR_lights'
                    has_light_ext = True
                if has_light_ext:
                    lightid = gltf_node['extensions'][light_ext]['light']
                    light = self.lights[lightid]
                    lnp = np.attach_new_node(light)
                    if self.compose_cs == CS_zup_right:
                        lnp.set_p(lnp.get_p() - 90)
                    lnp.set_r(lnp.get_r() - 90)
                    if isinstance(light, Light):
                        root.set_light(lnp)

                has_physics = (
                    'BLENDER_physics' in gltf_node['extensions'] or
                    'PANDA3D_physics_collision_shapes' in gltf_node['extensions']
                )
                if has_physics:
                    gltf_collisions = gltf_node['extensions'].get(
                        'PANDA3D_physics_collision_shapes',
                        gltf_node['extensions']['BLENDER_physics']
                    )
                    gltf_rigidbody = gltf_node['extensions'].get('BLENDER_physics', None)
                    if 'PANDA3D_physics_collision_shapes' in gltf_node['extensions']:
                        collision_shape = gltf_collisions['shapes'][0]
                        shape_type = collision_shape['type']
                    else:
                        collision_shape = gltf_collisions['collisionShapes'][0]
                        shape_type = collision_shape['shapeType']
                    bounding_box = [
                        max(0.00001, i)
                        for i in collision_shape['boundingBox']
                    ]
                    radius = max(bounding_box[0], bounding_box[1]) / 2.0
                    height = bounding_box[2]
                    geomnode = None
                    intangible = gltf_collisions.get('intangible', False)
                    if 'mesh' in collision_shape:
                        try:
                            geomnode = self.meshes[collision_shape['mesh']]
                        except KeyError:
                            print(
                                f"Could not find physics mesh ({collision_shape['mesh']}) for object ({nodeid})"
                            )
                    if 'extensions' in gltf_data and 'BP_physics_engine' in gltf_data['extensions']:
                        use_bullet = (
                            gltf_data['extensions']['BP_physics_engine']['engine'] == 'bullet'
                        )
                    else:
                        use_bullet = self.settings.collision_shapes == 'bullet'
                    if use_bullet and not HAVE_BULLET:
                        print(
                            'Warning: attempted to export for Bullet, which is unavailable, falling back to builtin'
                        )
                        use_bullet = False

                    if use_bullet:
                        phynode = self.load_physics_bullet(
                            node_name,
                            geomnode,
                            shape_type,
                            bounding_box,
                            radius,
                            height,
                            intangible,
                            gltf_rigidbody
                        )
                    else:
                        phynode = self.load_physics_builtin(
                            node_name,
                            geomnode,
                            shape_type,
                            bounding_box,
                            radius,
                            height,
                            intangible
                        )
                    if phynode is not None:
                        phynp = np.attach_new_node(phynode)
                        for geomnode in np.find_all_matches('+GeomNode'):
                            geomnode.reparent_to(phynp)

            for key, value in get_extras(gltf_node).items():
                np.set_tag(key, str(value))

            for child_nodeid in gltf_node.get('children', []):
                add_node(np, gltf_scene, child_nodeid)

            # Handle visibility after children are loaded
            def visible_recursive(node, visible):
                if visible:
                    node.show()
                else:
                    node.hide()
                for child in node.get_children():
                    visible_recursive(child, visible)

            hidden_nodes = scene_extras.get('hidden_nodes', [])
            if nodeid in hidden_nodes:
                #print('Hiding', np)
                visible_recursive(np, False)
            else:
                #print('Showing', np)
                visible_recursive(np, True)

            # Check if we need to deal with negative scale values
            scale = panda_node.get_transform().get_scale()
            negscale = scale.x * scale.y * scale.z < 0
            if negscale:
                for geomnode in np.find_all_matches('**/+GeomNode'):
                    tmp = geomnode.get_parent().attach_new_node(PandaNode('ReverseCulling'))
                    tmp.set_attrib(CullFaceAttrib.make_reverse())
                    geomnode.reparent_to(tmp)

            # Handle parenting to joints
            joint = self.joint_parents.get(nodeid)
            if joint:
                xformnp = root.attach_new_node(PandaNode(f'{node_name}-parent'))
                np.reparent_to(xformnp)
                joint.add_net_transform(xformnp.node())

            # if the NodePath children were moved under a Character and has no other children,
            # then we can safely delete the NodePath
            if charinfo and not np.children:
                np.remove_node()

        for sceneid, gltf_scene in enumerate(gltf_data.get('scenes', [])):
            scene_name = gltf_scene.get('name', 'scene'+str(sceneid))
            scene_root = NodePath(ModelRoot(scene_name))

            node_list = gltf_scene['nodes']
            hidden_nodes = get_extras(gltf_scene).get('hidden_nodes', [])
            node_list += hidden_nodes

            # Run through and pre-build Characters
            for nodeid in node_list:
                build_characters(nodeid)

            # Now iterate again to build the scene graph
            for nodeid in node_list:
                add_node(scene_root, gltf_scene, nodeid)

            if self.settings.flatten_nodes:
                scene_root.flatten_medium()

            self.scenes[sceneid] = scene_root

        # Set the active scene
        sceneid = gltf_data.get('scene', 0)
        if sceneid in self.scenes:
            self.active_scene = self.scenes[sceneid]
        if 'scenes' in gltf_data:
            gltf_scene = gltf_data['scenes'][sceneid]
            scene_extras = get_extras(gltf_scene)
            if 'background_color' in scene_extras:
                self.background_color = scene_extras['background_color']
            if 'active_camera' in scene_extras:
                self.active_camera = scene_extras['active_camera']

    def load_matrix(self, mat):
        lmat = LMatrix4()

        for i in range(4):
            lmat.set_row(i, LVecBase4(*mat[i * 4: i * 4 + 4]))
        return lmat

    def load_buffer(self, buffid, gltf_buffer):
        if 'uri' not in gltf_buffer:
            assert self.buffers[buffid]
            return

        uri = gltf_buffer['uri']
        if uri == '_glb_bin' and buffid == 0:
            buff_data = gltf_buffer['_glb_bin']
        elif uri.startswith('data:application/octet-stream;base64') or \
           uri.startswith('data:application/gltf-buffer;base64'):
            buff_data = gltf_buffer['uri'].split(',')[1]
            buff_data = base64.b64decode(buff_data)
        elif uri.endswith('.bin'):
            buff_fname = os.path.join(self.filedir.to_os_specific(), uri)
            with open(buff_fname, 'rb') as buff_file:
                buff_data = buff_file.read(gltf_buffer['byteLength'])
        else:
            print(
                "Buffer {buffid} has an unsupported uri ({uri}), using a zero filled buffer instead"
            )
            buff_data = bytearray(gltf_buffer['byteLength'])
        self.buffers[buffid] = buff_data

    def get_buffer_view(self, gltf_data, view_id):
        buffview = gltf_data['bufferViews'][view_id]
        buff = self.buffers[buffview['buffer']]
        start = buffview.get('byteOffset', 0)
        end = start + buffview['byteLength']
        stride = buffview.get('byteStride', 1)
        return memoryview(buff)[start:end:stride]

    def get_buffer_from_accessor(self, gltf_data, accid, unpack=True):
        acc = gltf_data['accessors'][accid]
        viewid = acc['bufferView']
        buff_view = self.get_buffer_view(gltf_data, viewid)
        if 'byteOffset' in acc:
            buff_view = buff_view[acc['byteOffset']:]

        formatstr = (
            COMPONENT_FORMT_STR_MAP[acc['componentType']]
            * COMPONENT_NUM_MAP[acc['type']]
        )

        convertfn = lambda x: x

        acctype = acc['type']
        if acctype == 'SCALAR':
            convertfn = lambda x: x[0]
        elif acctype == 'VEC2':
            convertfn = lambda x: p3d.LVector2(*x)
        elif acctype == 'VEC3':
            convertfn = lambda x: p3d.LVector3(*x)
        elif acctype == 'VEC4':
            convertfn = lambda x: p3d.LVector4(*x)

        element_size = (
            COMPONENT_SIZE_MAP[acc['componentType']]
            * COMPONENT_NUM_MAP[acc['type']]
        )
        end = acc['count'] * element_size
        buffdata = buff_view[:end]

        if unpack:
            return list(map(convertfn, struct.iter_unpack(f'<{formatstr}', buffdata)))
        else:
            return buffdata

    def get_texture_stage(self, slot_name, texmode, texcoord):
        texcoord = str(texcoord)
        tshash = slot_name + str(texmode) + texcoord

        if tshash not in self.texture_stages:
            texstage = TextureStage(slot_name)
            texstage.set_texcoord_name(InternalName.get_texcoord_name(texcoord))
            texstage.set_mode(texmode)
            self.texture_stages[tshash] = texstage

        return self.texture_stages[tshash]

    def make_texture_srgb(self, texture):
        if self.settings.no_srgb:
            return

        if texture is None:
            return

        if texture.get_num_components() == 3:
            texture.set_format(Texture.F_srgb)
        elif texture.get_num_components() == 4:
            texture.set_format(Texture.F_srgb_alpha)

    def load_texture(self, texid, gltf_tex, gltf_data):
        if 'source' not in gltf_tex:
            print(f"Texture '{texid}' has no source, skipping")
            return

        def load_embedded_image(name, ext, data):
            if not name:
                name = f'gltf-embedded-{texid}'
            img_type_registry = PNMFileTypeRegistry.get_global_ptr()
            img_type = img_type_registry.get_type_from_extension(ext)

            img = PNMImage()
            img.read(StringStream(data), type=img_type)

            texture = Texture(name)
            texture.load(img)

            return texture

        source = gltf_data['images'][gltf_tex['source']]
        if 'uri' in source:
            uri = source['uri']
            if uri.startswith('data:'):
                info, b64data = uri.split(',')

                if not (info.startswith('data:image/') and info.endswith(';base64')):
                    raise RuntimeError(
                        f'Unknown data URI: {info}'
                    )

                name = source.get('name', '')
                ext = info.replace('data:image/', '').replace(';base64', '')
                data = base64.b64decode(b64data)

                texture = load_embedded_image(name, ext, data)
            else:
                uri = urllib.parse.unquote(uri)
                fname = Filename.from_os_specific(uri)
                if not os.path.isabs(uri):
                    fulluri = Filename(self.filedir, uri)
                else:
                    fulluri = fname
                texture = TexturePool.load_texture(fulluri, 0, False, LoaderOptions())
                if not texture:
                    raise RuntimeError(f'failed to load texture: {fulluri}')
                texture.filename = uri
        else:
            name = source.get('name', '')
            ext = source['mimeType'].split('/')[1]
            data = self.get_buffer_view(gltf_data, source['bufferView'])
            texture = load_embedded_image(name, ext, data)

        if 'sampler' in gltf_tex:
            gltf_sampler = gltf_data['samplers'][gltf_tex['sampler']]
            gltf_magfilter = gltf_sampler.get('magFilter', None)
            if gltf_magfilter:
                try:
                    magfilter = MAG_FILTER_MAP[gltf_magfilter]
                    texture.set_magfilter(magfilter)
                except KeyError:
                    print(
                        f"Sampler {gltf_tex['sampler']} has unsupported magFilter type {gltf_sampler['magFilter']}"
                    )

            gltf_minfilter = gltf_sampler.get('minFilter', None)
            if gltf_minfilter:
                try:
                    minfilter = MIN_FILTER_MAP[gltf_minfilter]
                    texture.set_minfilter(minfilter)
                except KeyError:
                    print(
                        f"Sampler {gltf_tex['sampler']} has unsupported minFilter type {gltf_sampler['minFilter']}"
                    )

            gltf_wraps = gltf_sampler.get('wrapS', 10497)
            try:
                wraps = WRAP_MODE_MAP[gltf_wraps]
                texture.set_wrap_u(wraps)
            except KeyError:
                print(
                    f"Sampler {gltf_tex['sampler']} has unsupported wrapS type {gltf_wraps}"
                )

            gltf_wrapt = gltf_sampler.get('wrapT', 10497)
            try:
                wrapt = WRAP_MODE_MAP[gltf_wrapt]
                texture.set_wrap_v(wrapt)
            except KeyError:
                print(
                    f"Sampler {gltf_tex['sampler']} has unsupported wrapT type {gltf_wrapt}"
                )

        self.textures[texid] = texture

    def load_material(self, matid, gltf_mat):
        matname = gltf_mat.get('name', 'mat'+str(matid))
        state = self.mat_states.get(matid, RenderState.make_empty())

        if matid not in self.mat_mesh_map:
            self.mat_mesh_map[matid] = []

        pmat = Material(matname)
        tex_attrib = TextureAttrib.make()
        tex_mat_attrib = None

        def add_texture(gltf_texture, slot_name, texmode, *, make_srgb=False):
            nonlocal tex_attrib, tex_mat_attrib

            if gltf_texture is None:
                return

            transform_ext = gltf_texture.get('extensions', {}).get('KHR_texture_transform')

            texdata = self.textures.get(gltf_texture['index'], None)
            if texdata is None:
                print(f"Could not find texture for key: {gltf_texture['index']}")
                return

            if make_srgb:
                self.make_texture_srgb(texdata)

            texcoord = gltf_texture.get('texCoord', 0)
            if transform_ext and 'texCoord' in transform_ext:
                # This overrides, if present.
                texcoord = transform_ext['texCoord']

            texstage = self.get_texture_stage(
                slot_name,
                texmode,
                texcoord
            )
            tex_attrib = tex_attrib.add_on_stage(texstage, texdata)

            if transform_ext:
                # glTF uses a transform origin of the upper-left corner of the
                # texture, whereas Panda uses the lower-left corner.
                mat = Mat3()
                scale = transform_ext.get('scale')
                if scale:
                    mat *= (Mat3.translate_mat(0, -1) *
                            Mat3.scale_mat(scale[0], scale[1]) *
                            Mat3.translate_mat(0, 1))

                rot = transform_ext.get('rotation')
                if rot:
                    mat *= (Mat3.translate_mat(0, -1) *
                            Mat3.rotate_mat(math.degrees(rot)) *
                            Mat3.translate_mat(0, 1))

                offset = transform_ext.get('offset', [0, 0])
                if offset:
                    mat *= Mat3.translate_mat(offset[0], -offset[1])

                transform = TransformState.make_mat3(mat)
                if not tex_mat_attrib:
                    tex_mat_attrib = TexMatrixAttrib.make(texstage, transform)
                else:
                    tex_mat_attrib = tex_mat_attrib.add_stage(texstage, transform)

        if self.settings.legacy_materials:
            if 'pbrMetallicRoughness' in gltf_mat:
                pbrsettings = gltf_mat['pbrMetallicRoughness']

                pmat.set_diffuse(LColor(*pbrsettings.get('baseColorFactor', [1.0, 1.0, 1.0, 1.0])))
                add_texture(
                    pbrsettings.get('baseColorTexture'),
                    'Base Color',
                    TextureStage.M_modulate,
                    make_srgb=True
                )

            add_texture(
                gltf_mat.get('normalTexture'),
                'Normal',
                TextureStage.M_normal
            )
        else:
            mat_extensions = gltf_mat.get('extensions', {})
            if 'BP_materials_legacy' in mat_extensions:
                matsettings = mat_extensions['BP_materials_legacy']['bpLegacy']
                pmat.set_shininess(matsettings['shininessFactor'])
                pmat.set_ambient(LColor(*matsettings['ambientFactor']))

                if 'diffuseTexture' in matsettings:
                    add_texture(
                        matsettings['diffuseTexture'],
                        'Diffuse',
                        TextureStage.M_modulate,
                        make_srgb=True
                    )
                else:
                    pmat.set_diffuse(LColor(*matsettings['diffuseFactor']))

                if 'emissionTexture' in matsettings:
                    add_texture(
                        matsettings['emissionTexture'],
                        'Emission',
                        TextureStage.M_emission,
                        make_srgb=True
                    )
                else:
                    pmat.set_emission(LColor(*matsettings['emissionFactor']))

                if 'specularTexture' in matsettings:
                    add_texture(
                        matsettings['specularTexture'],
                        'Specular',
                        TextureStage.M_modulate,
                        make_srgb=True
                    )
                else:
                    pmat.set_specular(LColor(*matsettings['specularFactor']))
            elif 'pbrMetallicRoughness' in gltf_mat:
                pbrsettings = gltf_mat['pbrMetallicRoughness']

                pmat.set_base_color(LColor(*pbrsettings.get('baseColorFactor', [1.0, 1.0, 1.0, 1.0])))
                add_texture(
                    pbrsettings.get('baseColorTexture'),
                    'Base Color',
                    TextureStage.M_modulate,
                    make_srgb=True
                )

                pmat.set_metallic(pbrsettings.get('metallicFactor', 1.0))
                pmat.set_roughness(pbrsettings.get('roughnessFactor', 1.0))
                add_texture(
                    pbrsettings.get('metallicRoughnessTexture'),
                    'Metal Rough',
                    TextureStage.M_selector,
                )

            # Normal map
            add_texture(
                gltf_mat.get('normalTexture'),
                'Normal',
                TextureStage.M_normal
            )

            # Emission map
            add_texture(
                gltf_mat.get('emissiveTexture'),
                'Emissive',
                TextureStage.M_emission,
                make_srgb=True
            )
            pmat.set_emission(LColor(*gltf_mat.get('emissiveFactor', [0.0, 0.0, 0.0]), w=0.0))

            # Index of refraction
            ior_ext = mat_extensions.get('KHR_materials_ior', {})
            pmat.set_refractive_index(ior_ext.get('ior', 1.5))

        double_sided = gltf_mat.get('doubleSided', False)
        pmat.set_twoside(double_sided)

        state = state.set_attrib(MaterialAttrib.make(pmat))

        if double_sided:
            state = state.set_attrib(CullFaceAttrib.make(CullFaceAttrib.MCullNone))

        state = state.set_attrib(tex_attrib)
        if tex_mat_attrib:
            state = state.set_attrib(tex_mat_attrib)

        # Setup Alpha mode
        alpha_mode = gltf_mat.get('alphaMode', 'OPAQUE')
        if alpha_mode == 'MASK':
            alpha_cutoff = gltf_mat.get('alphaCutoff', 0.5)
            alpha_attrib = AlphaTestAttrib.make(AlphaTestAttrib.M_greater_equal, alpha_cutoff)
            state = state.set_attrib(alpha_attrib)
        elif alpha_mode == 'BLEND':
            transp_attrib = TransparencyAttrib.make(TransparencyAttrib.M_alpha)
            state = state.set_attrib(transp_attrib)
        elif alpha_mode != 'OPAQUE':
            print(
                f"Warning: material {matid} has an unsupported alphaMode: {alpha_mode}"
            )

        # Remove stale meshes
        self.mat_mesh_map[matid] = [
            pair for pair in self.mat_mesh_map[matid] if pair[0] in self.meshes
        ]

        # Reload the material
        for meshid, geom_idx in self.mat_mesh_map[matid]:
            self.meshes[meshid].set_geom_state(geom_idx, state)

        self.mat_states[matid] = state

    def load_skin(self, skinid, gltf_skin, gltf_data):
        # Find a common root node.  First gather the parents of each node.
        # Note that we ignore the "skeleton" property of the glTF file, since it
        # is just a hint and not particularly necessary.
        parents = [None] * len(gltf_data['nodes'])
        for i, node in enumerate(gltf_data['nodes']):
            for child in node.get('children', ()):
                parents[child] = i

        # Now create a path for each joint node as well as each node that
        # is skinned with this skeleton, so that both are under the Character.
        paths = []
        for i, gltf_node in enumerate(gltf_data['nodes']):
            if i in gltf_skin['joints'] or gltf_node.get('skin') == skinid:
                path = [i]
                while parents[i] is not None:
                    i = parents[i]
                    path.insert(0, i)
                paths.append(tuple(path))

        # Find the longest prefix that is shared by all paths.
        common_path = paths[0]
        for path in paths[1:]:
            path = list(path[:len(common_path)])
            while path:
                if common_path[:len(path)] == tuple(path):
                    common_path = tuple(path)
                    break

                path.pop()

        root_nodeid = common_path[-1]

        self.skeletons[root_nodeid] = skinid

    def load_primitive(self, geom_node, gltf_primitive, gltf_mesh, gltf_data):
        # Build Vertex Format
        vformat = GeomVertexFormat()
        mesh_attribs = gltf_primitive['attributes']
        accessors = [
            {**gltf_data['accessors'][acc_idx], '_attrib': attrib_name}
            for attrib_name, acc_idx in mesh_attribs.items()
        ]

        # Check for morph target columns.
        targets = gltf_primitive.get('targets')
        if targets:
            target_names = get_extras(gltf_mesh).get('targetNames', [])

            for i, target in enumerate(targets):
                if i < len(target_names):
                    target_name = target_names[i]
                else:
                    target_name = str(i)

                accessors += [
                    {**gltf_data['accessors'][acc_idx], '_attrib': attrib_name, '_target': target_name}
                    for attrib_name, acc_idx in target.items()
                ]

        accessors = sorted(accessors, key=lambda x: x['bufferView'])
        data_copies = []
        is_skinned = 'JOINTS_0' in mesh_attribs
        calc_normals = not 'NORMAL' in mesh_attribs
        calc_tangents = not 'TANGENT' in mesh_attribs
        normalize_weights = False

        for buffview, accs in itertools.groupby(accessors, key=lambda x: x['bufferView']):
            buffview = gltf_data['bufferViews'][buffview]
            accs = sorted(accs, key=lambda x: x.get('byteOffset', 0))
            is_interleaved = len(accs) > 1 and accs[1].get('byteOffset', 0) < buffview['byteStride']

            varray = GeomVertexArrayFormat()
            for acc in accs:
                # Gather column information
                attrib_parts = acc['_attrib'].lower().split('_')
                attrib_name = ATTRIB_NAME_MAP.get(attrib_parts[0], attrib_parts[0])
                if attrib_name == 'texcoord' and len(attrib_parts) > 1:
                    internal_name = InternalName.make(attrib_name+'.', int(attrib_parts[1]))
                else:
                    internal_name = InternalName.make(attrib_name)
                num_components = COMPONENT_NUM_MAP[acc['type']]
                numeric_type = COMPONENT_TYPE_MAP[acc['componentType']]
                numeric_size = COMPONENT_SIZE_MAP[acc['componentType']]
                content = ATTRIB_CONTENT_MAP.get(attrib_name, GeomEnums.C_other)
                size = numeric_size * num_components

                if '_target' in acc:
                    internal_name = InternalName.get_morph(attrib_name, acc['_target'])
                    content = GeomEnums.C_morph_delta

                # Add this accessor as a column to the current vertex array format
                varray.add_column(internal_name, num_components, numeric_type, content)

                # Check if the weights table is using float or integer component
                # Weights normalization will only be performed on float weights.
                if attrib_parts[0] == 'weights':
                    normalize_weights = numeric_type == GeomEnums.NT_float32

                if not is_interleaved:
                    # Start a new vertex array format
                    vformat.add_array(varray)
                    varray = GeomVertexArrayFormat()
                    data_copies.append((
                        buffview['buffer'],
                        acc.get('byteOffset', 0) + buffview.get('byteOffset', 0),
                        acc['count'],
                        size,
                        buffview.get('byteStride', size)
                    ))

            if is_interleaved:
                vformat.add_array(varray)
                stride = buffview.get('byteStride', varray.get_stride())
                data_copies.append((
                    buffview['buffer'],
                    buffview.get('byteOffset', 0),
                    accs[0]['count'],
                    stride,
                    stride,
                ))

        # Copy data from buffers
        reg_format = GeomVertexFormat.register_format(vformat)
        vdata = GeomVertexData(geom_node.name, reg_format, GeomEnums.UH_stream)

        for array_idx, data_info in enumerate(data_copies):
            buffid, start, count, size, stride = data_info

            handle = vdata.modify_array(array_idx).modify_handle()
            handle.unclean_set_num_rows(count)

            buff = self.buffers[buffid]
            end = start + count * stride
            if stride == size:
                handle.copy_data_from(buff[start:end])
            else:
                src = start
                dest = 0
                while src < end:
                    handle.copy_subdata_from(dest, size, buff[src:src+size])
                    dest += size
                    src += stride
            handle = None

        # Flip UVs
        num_uvs = len({i for i in gltf_primitive['attributes'] if i.startswith('TEXCOORD')})
        for i in range(num_uvs):
            uv_data = GeomVertexRewriter(vdata, InternalName.get_texcoord_name(str(i)))

            while not uv_data.is_at_end():
                uvs = uv_data.get_data2f()
                uv_data.set_data2f(uvs[0], 1 - uvs[1])

        if self.compose_cs == CS_yup_right:
            # Flip morph deltas from Y-up to Z-up.  This is apparently not done by
            # transform_vertices(), below, so we do it ourselves.
            for morph_i in range(reg_format.get_num_morphs()):
                delta_data = GeomVertexRewriter(vdata, reg_format.get_morph_delta(morph_i))

                while not delta_data.is_at_end():
                    data = delta_data.get_data3f()
                    delta_data.set_data3f(data[0], -data[2], data[1])
            # Flip tangents from Y-up to Z-up.
            if 'TANGENT' in mesh_attribs:
                tangent = GeomVertexRewriter(vdata, InternalName.make('tangent'))
                while not tangent.is_at_end():
                    data = tangent.get_data4f()
                    tangent.set_data4f(data[0], -data[2], data[1], data[3])

        if normalize_weights:
            # The linear sum of all the joint weights must be as close as possible to 1, if the weights are
            # stored as float.
            # Some malformed assets do not respect this, hence we are normalizing them here.
            weights_data = GeomVertexRewriter(vdata, InternalName.get_transform_weight())
            while not weights_data.is_at_end():
                weights = weights_data.get_data4f()
                max_weight = abs(weights[0]) + abs(weights[1]) + abs(weights[2]) + abs(weights[3])
                if max_weight != 0.0:
                    weights = weights / max_weight
                weights_data.set_data4f(weights)

        # Repack mesh data
        vformat = GeomVertexFormat()
        varray_vert = GeomVertexArrayFormat()
        varray_skin = GeomVertexArrayFormat()
        varray_morph = GeomVertexArrayFormat()

        skip_columns = (
            InternalName.get_transform_index(),
            InternalName.get_transform_weight(),
            InternalName.get_transform_blend()
        )
        for arr in reg_format.get_arrays():
            for column in arr.get_columns():
                column_name = column.get_name()
                if column_name in skip_columns:
                    varray = varray_skin
                elif column_name.parent.basename == "morph":
                    varray = varray_morph
                else:
                    varray = varray_vert
                varray.add_column(
                    column_name,
                    column.get_num_components(),
                    column.get_numeric_type(),
                    column.get_contents()
                )
        vformat.add_array(varray_vert)

        if is_skinned or targets:
            aspec = GeomVertexAnimationSpec()
            aspec.set_panda()
            vformat.set_animation(aspec)

        if is_skinned:
            varray_blends = GeomVertexArrayFormat()
            varray_blends.add_column(InternalName.get_transform_blend(), 1, GeomEnums.NT_uint16, GeomEnums.C_index)

            vformat.add_array(varray_blends)
            vformat.add_array(varray_skin)

        if targets:
            vformat.add_array(varray_morph)

        reg_format = GeomVertexFormat.register_format(vformat)
        vdata = vdata.convert_to(reg_format)

        # Construct primitive
        primitiveid = geom_node.get_num_geoms()
        primitivemode = gltf_primitive.get('mode', 4)
        try:
            prim = PRIMITIVE_MODE_MAP[primitivemode](GeomEnums.UH_static)
        except KeyError:
            print(
                f"Warning: primitive {primitiveid} on mesh {geom_node.name} has an unsupported mode: {primitivemode}"
            )
            return

        if 'indices' in gltf_primitive:
            index_accid = gltf_primitive['indices']
            index_acc = gltf_data['accessors'][index_accid]
            num_elements = index_acc['count']
            prim.set_index_type(COMPONENT_TYPE_MAP[index_acc['componentType']])

            handle = prim.modify_vertices(num_elements).modify_handle()
            handle.unclean_set_num_rows(num_elements)
            handle.copy_data_from(self.get_buffer_from_accessor(gltf_data, index_accid, unpack=False))
            handle = None
        else:
            index_acc = gltf_data['accessors'][gltf_primitive['attributes']["POSITION"]]
            start = index_acc.get('byteOffset', 0)
            prim.setNonindexedVertices(start, index_acc['count'])

        # Assign a material
        matid = gltf_primitive.get('material', None)
        if matid is None:
            print(
                f"Warning: mesh {geom_node.name} has a primitive with no material, using an empty RenderState"
            )
            pmat = Material('fallback material')
            matattrib = MaterialAttrib.make(pmat)
            mat = RenderState.make(matattrib)
        elif matid not in self.mat_states:
            print(
                f"Warning: material with name {matid} has no associated mat state, using an empty RenderState"
            )
            pmat = Material('fallback material')
            matattrib = MaterialAttrib.make(pmat)
            mat = RenderState.make(matattrib)
        else:
            mat = self.mat_states[gltf_primitive['material']]
            self.mat_mesh_map[gltf_primitive['material']].append((geom_node.name, primitiveid))

        # Add this primitive back to the geom node
        #ss = StringStream()
        #vdata.write(ss)
        ###prim.write(ss, 2)
        #print(ss.data.decode('utf8'))
        geom = Geom(vdata)
        geom.add_primitive(prim)
        if calc_normals:
            self.calculate_normals(geom)
        if calc_tangents:
            self.calculate_tangents(geom)
        geom.transform_vertices(self.csxform)
        geom_node.add_geom(geom, mat)

    def calculate_normals(self, geom):
        # Generate flat normals, as required by the glTF spec.
        if geom.get_primitive_type() != GeomEnums.PT_polygons:
            return

        # We need to deindex the primitive since each occurrence of a vertex on
        # a triangle could have a different normal vector.
        geom.decompose_in_place()
        geom.make_nonindexed(False)

        gvd = geom.get_vertex_data()
        gvd = gvd.replace_column(InternalName.get_normal(), 3, GeomEnums.NT_float32, GeomEnums.C_normal)
        vertex_reader = GeomVertexReader(gvd, 'vertex')
        normal_writer = GeomVertexWriter(gvd, 'normal')

        read_vertex = vertex_reader.get_data3
        write_normal = normal_writer.set_data3

        while not vertex_reader.is_at_end():
            vtx1 = read_vertex()
            vtx2 = read_vertex()
            vtx3 = read_vertex()
            normal = (vtx2 - vtx1).cross(vtx3 - vtx1)
            normal.normalize()
            write_normal(normal)
            write_normal(normal)
            write_normal(normal)

        geom.set_vertex_data(gvd)

    def calculate_tangents(self, geom):
        # Adapted from https://www.marti.works/calculating-tangents-for-your-mesh/
        prim = geom.get_primitive(0)
        gvd = geom.get_vertex_data()
        gvd = gvd.replace_column(InternalName.get_tangent(), 4, GeomEnums.NT_float32, GeomEnums.C_other)
        tangent_writer = GeomVertexWriter(gvd, InternalName.get_tangent())

        primverts = prim.get_vertex_list()
        tris = [primverts[i:i+3] for i in range(0, len(primverts), 3)]
        posdata = self.read_vert_data(gvd, InternalName.get_vertex())
        normaldata = [LVector3(i[0], i[1], i[2]) for i in self.read_vert_data(gvd, InternalName.get_normal())]
        uvdata = [LVector2(i[0], i[1]) for i in self.read_vert_data(gvd, InternalName.get_texcoord_name('0'))]
        tana = [LVector3(0, 0, 0) for i in range(len(posdata))]
        tanb = [LVector3(0, 0, 0) for i in range(len(posdata))]

        if not uvdata:
            # No point generating tangents without UVs.
            return

        # Gather tangent data from triangles
        for tri in tris:
            idx0, idx1, idx2 = tri
            edge1 = posdata[idx1] - posdata[idx0]
            edge2 = posdata[idx2] - posdata[idx0]
            duv1 = uvdata[idx1] - uvdata[idx0]
            duv2 = uvdata[idx2] - uvdata[idx0]

            denom = duv1.x * duv2.y - duv2.x * duv1.y
            if denom != 0.0:
                fconst = 1.0 / denom
                tangent = (edge1.xyz * duv2.y - edge2.xyz * duv1.y) * fconst
                bitangent = (edge2.xyz * duv1.x - edge1.xyz * duv2.x) * fconst
            else:
                tangent = LVector3(0)
                bitangent = LVector3(0)

            for idx in tri:
                tana[idx] += tangent
                tanb[idx] += bitangent

        # Calculate per-vertex tangent values
        for normal, tan0, tan1 in zip(normaldata, tana, tanb):
            tangent = tan0 - (normal * normal.dot(tan0))
            tangent.normalize()

            tangent4 = LVector4(
                tangent.x,
                tangent.y,
                tangent.z,
                -1.0 if normal.cross(tan0).dot(tan1) < 0 else 1.0
            )
            if self.compose_cs == CS_yup_right:
                tangent_writer.set_data4(tangent4[0], -tangent4[2], tangent4[1], tangent4[3])
            else:
                tangent_writer.set_data4(tangent4)

        geom.set_vertex_data(gvd)

    def load_mesh(self, meshid, gltf_mesh, gltf_data):
        mesh_name = gltf_mesh.get('name', 'mesh'+str(meshid))
        node = self.meshes.get(meshid, GeomNode(mesh_name))

        # Clear any existing mesh data
        node.remove_all_geoms()

        # Load primitives
        for gltf_primitive in gltf_mesh['primitives']:
            self.load_primitive(node, gltf_primitive, gltf_mesh, gltf_data)

        # Save mesh
        self.meshes[meshid] = node

    def read_vert_data(self, gvd, column_name):
        gvr = GeomVertexReader(gvd, column_name)
        data = []
        while not gvr.is_at_end():
            data.append(LVecBase4(gvr.get_data4()))
        return data

    def build_character(self, charinfo: CharInfo, nodeid, gltf_data, recurse=True):
        char = charinfo.character
        affected_nodeids = set()
        jvtmap = charinfo.jvtmap
        cvsmap = charinfo.cvsmap

        for bundle in char.bundles:
            bundle.frame_blend_flag = True

        if nodeid in self.skeletons:
            skinid = self.skeletons[nodeid]
            gltf_skin = gltf_data['skins'][skinid]

            if 'skeleton' in gltf_skin:
                root_nodeids = [gltf_skin['skeleton']]
            else:
                # find a common root node
                joint_nodes = [gltf_data['nodes'][i] for i in gltf_skin['joints']]
                child_set = list(itertools.chain(*[node.get('children', []) for node in joint_nodes]))
                root_nodeids = [nodeid for nodeid in gltf_skin['joints'] if nodeid not in child_set]

            jvtmap.update(self.build_character_joints(char, root_nodeids,
                                                      affected_nodeids, skinid,
                                                      gltf_data))

        cvsmap.update(self.build_character_sliders(char, nodeid, affected_nodeids,
                                                   gltf_data, recurse=recurse))

        # Find animations that affect the collected nodes.
        #print("Looking for actions for", skinname, node_ids)
        if not self.settings.skip_animations:
            anims = [
                (animid, anim)
                for animid, anim in enumerate(gltf_data.get('animations', []))
                if affected_nodeids & {chan['target']['node'] for chan in anim['channels']}
            ]
        else:
            anims = []

        for animid, gltf_anim in anims:
            anim_name = gltf_anim.get('name', 'anim'+str(animid))
            #print("\t", anim_name)

            samplers = gltf_anim['samplers']
            channels = gltf_anim['channels']

            time_acc_ids = list({i['input'] for i in samplers})
            time_data = [
                list(self.get_buffer_from_accessor(
                    gltf_data,
                    accid
                ))
                for accid in time_acc_ids
            ]
            max_time = max([max(i) for i in time_data])
            fps = self.settings.animation_fps
            num_frames = max(math.ceil(max_time * fps), 1)

            bundle_name = anim_name
            bundle = AnimBundle(bundle_name, fps, num_frames)

            if nodeid in self.skeletons and any(chan['target']['path'] != 'weights' for chan in channels):
                skeleton = AnimGroup(bundle, '<skeleton>')
                for root_nodeid in root_nodeids:
                    self.build_animation_skeleton(char, skeleton, root_nodeid,
                                                  num_frames, gltf_anim, gltf_data)

            if cvsmap and any(chan['target']['path'] == 'weights' for chan in channels):
                morph = AnimGroup(bundle, 'morph')
                self.build_animation_morph(morph, nodeid, num_frames, gltf_anim,
                                           gltf_data, recurse=recurse)

            char.add_child(AnimBundleNode(char.name, bundle))

    def combine_mesh_skin(self, geom_node, charinfo):
        jvtmap = charinfo.jvtmap
        if not jvtmap:
            return
        jvtmap = collections.OrderedDict(sorted(jvtmap.items()))

        for geom in geom_node.modify_geoms():
            gvd = geom.modify_vertex_data()
            tbtable = TransformBlendTable()
            tdata = GeomVertexWriter(gvd, InternalName.get_transform_blend())

            if not tdata.has_column():
                continue

            jointdata = self.read_vert_data(gvd, InternalName.get_transform_index())
            weightdata = self.read_vert_data(gvd, InternalName.get_transform_weight())

            for joints, weights in zip(jointdata, weightdata):
                tblend = TransformBlend()
                for joint, weight in zip(joints, weights):
                    joint = int(joint)
                    try:
                        jvt = jvtmap[joint]
                    except KeyError:
                        print(
                            f"Could not find joint in jvtmap:\n\tjoint={joint}\n\tjvtmap={jvtmap}"
                        )
                        # Don't warn again for this joint.
                        jvt = None
                        jvtmap[joint] = None
                    if jvt is not None:
                        tblend.add_transform(jvt, weight)
                tdata.add_data1i(tbtable.add_blend(tblend))
            tbtable.set_rows(SparseArray.lower_on(gvd.get_num_rows()))
            gvd.set_transform_blend_table(tbtable)

            # Set the transform of the skinned node to the inverse of the parent's
            # transform.  This allows skinning to happen in global space.
            net_xform = NodePath(geom_node.get_parent(0)).get_net_transform()
            inverse = net_xform.get_inverse()
            gvd.transform_vertices(inverse.get_mat())

    def combine_mesh_morphs(self, geom_node, meshid, charinfo):
        cvsmap = charinfo.cvsmap
        if not cvsmap:
            return
        zero = LVecBase4.zero()

        for geom in geom_node.modify_geoms():
            gvd = geom.modify_vertex_data()
            vformat = gvd.get_format()

            stable = SliderTable()

            for (slider_meshid, name), slider in cvsmap.items():
                if slider_meshid != meshid:
                    continue

                # Iterate through the morph columns to figure out which rows are
                # affected by which slider.
                rows = SparseArray()

                for morph_i in range(vformat.get_num_morphs()):
                    column_name = vformat.get_morph_delta(morph_i)
                    if column_name.basename == name:
                        gvr = GeomVertexReader(gvd, vformat.get_morph_delta(morph_i))
                        row = 0
                        while not gvr.is_at_end():
                            if gvr.get_data4() != zero:
                                rows.set_bit(row)
                            row += 1

                if not rows.is_zero():
                    stable.add_slider(slider, rows)

            if stable.get_num_sliders() > 0:
                gvd.set_slider_table(SliderTable.register_table(stable))

    def build_character_joints(self, char, root_nodeids, affected_nodeids, skinid, gltf_data):
        gltf_skin = gltf_data['skins'][skinid]

        bundle = char.get_bundle(0)
        skeleton = PartGroup(bundle, "<skeleton>")
        jvtmap = {}

        bind_mats = {}
        if 'inverseBindMatrices' in gltf_skin:
            ibmdata = self.get_buffer_from_accessor(gltf_data, gltf_skin['inverseBindMatrices'])

            for i, matdata in enumerate(ibmdata):
                mat = self.load_matrix(matdata)
                mat.invert_in_place()
                bind_mats[i] = mat

        def create_joint(parent, nodeid, transform):
            node = gltf_data['nodes'][nodeid]
            node_name = node.get('name', 'bone'+str(nodeid))

            if 'mesh' in node:
                self.joint_parents[nodeid] = parent
                return

            inv_transform = LMatrix4(transform)
            inv_transform.invert_in_place()
            joint_index = None
            joint_mat = LMatrix4.ident_mat()
            if nodeid in gltf_skin['joints']:
                joint_index = gltf_skin['joints'].index(nodeid)
                joint_mat = bind_mats.get(joint_index, LMatrix4.ident_mat())
                self._joint_nodes.add(nodeid)

            # glTF uses an absolute bind pose, Panda wants it local
            bind_pose = joint_mat * inv_transform
            joint = CharacterJoint(char, bundle, parent, node_name, self.csxform_inv * bind_pose * self.csxform)

            # Non-deforming bones are not in the skin's jointNames, don't add them to the jvtmap
            if joint_index is not None:
                jvtmap[joint_index] = JointVertexTransform(joint)

            affected_nodeids.add(nodeid)

            for child in node.get('children', []):
                #print("Create joint for child", child)
                create_joint(joint, child, bind_pose * transform)

        root_mat = self.csxform * NodePath(char).get_net_transform().get_mat() * self.csxform_inv
        for root_nodeid in root_nodeids:
            # Construct a path to the root
            create_joint(skeleton, root_nodeid, root_mat)

        return jvtmap

    def build_character_sliders(self, char, root_nodeid, affected_nodeids, gltf_data, recurse=True):
        bundle = char.get_bundle(0)
        morph = PartGroup(bundle, "morph")
        cvsmap = {}

        def create_slider(nodeid):
            gltf_node = gltf_data['nodes'][nodeid]

            if 'mesh' in gltf_node:
                meshid = gltf_node['mesh']
                gltf_mesh = gltf_data['meshes'][meshid]
                weights = gltf_mesh.get('weights')

                num_targets = 0
                for gltf_primitive in gltf_mesh['primitives']:
                    targets = gltf_primitive.get('targets')
                    if targets:
                        num_targets = max(len(targets), num_targets)

                if num_targets > 0:
                    target_names = get_extras(gltf_mesh).get('targetNames', [])
                    num_targets = max(len(target_names), num_targets)

                    if not weights:
                        weights = [0] * num_targets

                    if len(target_names) < len(weights):
                        target_names += [str(i) for i in range(len(target_names), len(weights))]

                    assert len(target_names) == len(weights)
                    affected_nodeids.add(nodeid)

                    # If we do this recursively, create a group for every mesh.
                    if recurse:
                        group = PartGroup(morph, 'mesh'+str(meshid))
                    else:
                        group = morph

                    for i, name in enumerate(target_names):
                        try:
                            slider = CharacterSlider(group, name, weights[i])
                        except TypeError:
                            # Panda versions before 1.10.6.dev6 did not permit default values.
                            slider = CharacterSlider(group, name)

                        cvsmap[(meshid, name)] = CharacterVertexSlider(slider)

            if recurse:
                for child in gltf_node.get('children', []):
                    create_slider(child)

        create_slider(root_nodeid)
        return cvsmap

    def build_animation_skeleton(self, character, parent, boneid, num_frames, gltf_anim, gltf_data):
        bone = gltf_data['nodes'][boneid]
        bone_name = bone.get('name', 'bone'+str(boneid))
        channels = [chan for chan in gltf_anim['channels'] if chan['target']['node'] == boneid]
        joint_mat = character.find_joint(bone_name).get_transform()

        group = AnimChannelMatrixXfmTable(parent, bone_name)

        def extract_chan_data(path):
            samplers = [
                gltf_anim['samplers'][chan['sampler']]
                for chan in channels
                if chan['target']['path'] == path
            ]
            if not samplers:
                return None
            sampler = samplers[0]

            accid = sampler['output']
            output_buff = list(self.get_buffer_from_accessor(gltf_data, accid))

            if path == 'rotation':
                output_buff = [p3d.LQuaternion(x[3], x[0], x[1], x[2]) for x in output_buff]

            accid = sampler['input']
            input_buff = self.get_buffer_from_accessor(gltf_data, accid)

            interpolation_mode = sampler.get('interpolation', 'LINEAR')
            if interpolation_mode == 'CUBICSPLINE':
                print(
                    f'Warning: CUBICSPLINE interpolation mode for {bone_name}:{path} is not supported, '
                    'falling back to LINEAR'
                )
                interpolation_mode = 'LINEAR'
            return list(input_buff), list(output_buff), interpolation_mode

        # Create default animaton data
        translation = LVector3()
        rotation_vec = LVector3()
        scale = LVector3()
        decompose_matrix(self.csxform * joint_mat * self.csxform_inv, scale, rotation_vec, translation, CS_yup_right)
        rotation = LQuaternion()
        rotation.set_hpr(rotation_vec, CS_yup_right)

        # Override defaults with any found animation data
        default_anim_data = {
            'translation': translation,
            'rotation': rotation,
            'scale': scale,
        }
        anim_data = {
            path: extract_chan_data(path)
            for path in ['translation', 'rotation', 'scale']
        }

        loc_vals = [[], [], []]
        rot_vals = [[], [], []]
        scale_vals = [[], [], []]

        def calculate_frame_value(frame, path):
            currtime = frame / self.settings.animation_fps
            if not anim_data[path]:
                return default_anim_data[path]

            input_buff, output_buff, interpolation_mode = anim_data[path]

            if len(input_buff) == 1:
                # no need to interpolate, just return the value
                return output_buff[0]

            nextidx = get_next_time_index(currtime, input_buff)
            lastidx = nextidx - 1

            if interpolation_mode == 'STEP':
                return output_buff[lastidx]
            elif interpolation_mode == 'LINEAR':
                nexttime = input_buff[nextidx]
                lasttime = input_buff[lastidx]
                lerpfactor = get_lerp_factor(currtime, lasttime, nexttime)

                if path == 'rotation':
                    return slerp(output_buff[lastidx], output_buff[nextidx], lerpfactor)
                return vlerp(output_buff[lastidx], output_buff[nextidx], lerpfactor)
            else:
                return RuntimeError(
                    f'Unrecognized interpolation mode ({interpolation_mode}) found on {bone_name}:{path}'
                )

        for i in range(num_frames):
            frame_translation = calculate_frame_value(i, 'translation')
            frame_rotation = calculate_frame_value(i, 'rotation')
            frame_scale = calculate_frame_value(i, 'scale')

            mat = LMatrix4(LMatrix4.ident_mat())
            mat *= LMatrix4.scale_mat(frame_scale)
            mat = frame_rotation * mat
            mat *= LMatrix4.translate_mat(frame_translation)
            mat = self.csxform_inv * mat * self.csxform

            frame_translation = LVector3()
            frame_scale = LVector3()
            frame_rotation = LVector3()
            decompose_matrix(mat, frame_scale, frame_rotation, frame_translation)

            loc_vals[0].append(frame_translation[0])
            loc_vals[1].append(frame_translation[1])
            loc_vals[2].append(frame_translation[2])
            rot_vals[0].append(frame_rotation[0])
            rot_vals[1].append(frame_rotation[1])
            rot_vals[2].append(frame_rotation[2])
            scale_vals[0].append(frame_scale[0])
            scale_vals[1].append(frame_scale[1])
            scale_vals[2].append(frame_scale[2])

        # if all values for a given channel are close enough, we can use the first value for
        # all frames and save some space
        def almost_equal(val1, val2):
            return abs(val2 - val1) < 0.00001
        for val_arrays in [loc_vals, rot_vals, scale_vals]:
            for val_array in val_arrays:
                if almost_equal(min(val_array), max(val_array)):
                    val_array[:] = val_array[:1]

        # Write data to tables
        group.set_table(b'x', CPTA_stdfloat(PTA_stdfloat(loc_vals[0])))
        group.set_table(b'y', CPTA_stdfloat(PTA_stdfloat(loc_vals[1])))
        group.set_table(b'z', CPTA_stdfloat(PTA_stdfloat(loc_vals[2])))

        group.set_table(b'h', CPTA_stdfloat(PTA_stdfloat(rot_vals[0])))
        group.set_table(b'p', CPTA_stdfloat(PTA_stdfloat(rot_vals[1])))
        group.set_table(b'r', CPTA_stdfloat(PTA_stdfloat(rot_vals[2])))

        group.set_table(b'i', CPTA_stdfloat(PTA_stdfloat(scale_vals[0])))
        group.set_table(b'j', CPTA_stdfloat(PTA_stdfloat(scale_vals[1])))
        group.set_table(b'k', CPTA_stdfloat(PTA_stdfloat(scale_vals[2])))

        for childid in bone.get('children', []):
            gltf_node = gltf_data['nodes'][childid]
            if 'mesh' in gltf_node:
                continue
            self.build_animation_skeleton(character, group, childid, num_frames, gltf_anim, gltf_data)

    def build_animation_morph(self, parent, nodeid, num_frames, gltf_anim, gltf_data, recurse=True):
        def create_channels(parent, nodeid, target_names, default_weights):
            channels = [
                chan for chan in gltf_anim['channels']
                if chan['target']['node'] == nodeid and chan['target']['path'] == 'weights'
            ]

            samplers = [
                gltf_anim['samplers'][chan['sampler']]
                for chan in channels
                if chan['target']['path'] == 'weights'
            ]

            if samplers:
                sampler = samplers[0]
                buff_data = self.get_buffer_from_accessor(gltf_data, sampler['output'])
                weights = list(CPTAFloat(buff_data))
                time_data = self.get_buffer_from_accessor(gltf_data, sampler['input'])
                interpolation_mode = sampler.get('interpolation', 'LINEAR')
                if interpolation_mode == 'CUBICSPLINE':
                    print(
                        'Warning: CUBICSPLINE interpolation mode for morph targets is not supported, '
                        'falling back to LINEAR'
                    )
                    interpolation_mode = 'LINEAR'
            else:
                weights = default_weights
                time_data = [0]
                interpolation_mode = 'STEP'

            num_targets = len(default_weights)

            for i, target_name in enumerate(target_names):
                group = AnimChannelScalarTable(parent, target_name)

                target_weights = weights[i::num_targets]
                interpolated_weights = []

                if len(time_data) == 1 or min(target_weights) == max(target_weights):
                    # If all frames are the same, we only need to store one frame.
                    interpolated_weights = target_weights[:1]
                else:
                    for frame in range(num_frames):
                        currtime = frame / self.settings.animation_fps
                        nextidx = get_next_time_index(currtime, time_data)
                        lastidx = nextidx - 1
                        if interpolation_mode == 'STEP':
                            interpolated_weights.append(target_weights[lastidx])
                        elif interpolation_mode == 'LINEAR':
                            lasttime = time_data[lastidx]
                            nexttime = time_data[nextidx]
                            lerpfactor = get_lerp_factor(currtime, lasttime, nexttime)
                            interpolated_weights.append(
                                target_weights[lastidx] * (1 - lerpfactor)
                                + target_weights[nextidx] * lerpfactor
                            )
                        else:
                            return RuntimeError(
                                f'Unrecognized interpolation mode ({interpolation_mode}) found on {target_name}'
                            )

                group.set_table(CPTA_stdfloat(interpolated_weights))

        gltf_node = gltf_data['nodes'][nodeid]

        if 'mesh' in gltf_node:
            meshid = gltf_node['mesh']
            gltf_mesh = gltf_data['meshes'][meshid]
            weights = gltf_mesh.get('weights')
            if weights:
                target_names = get_extras(gltf_mesh).get('targetNames', [])
                if len(target_names) < len(weights):
                    target_names += [str(i) for i in range(len(target_names), len(weights))]

                assert len(target_names) == len(weights)

                # If we do this recursively, group the sliders for each mesh
                # under a group for their respective mesh, so that the names will
                # not conflict.
                if recurse:
                    group = AnimGroup(parent, 'mesh'+str(meshid))
                else:
                    group = parent

                create_channels(group, nodeid, target_names, weights)

        if recurse:
            for child in gltf_node.get('children', []):
                self.build_animation_morph(parent, child, num_frames, gltf_anim, gltf_data)

    def load_camera(self, camid, gltf_camera):
        camname = gltf_camera.get('name', 'cam'+str(camid))
        node = self.cameras.get(camid, Camera(camname))

        if gltf_camera['type'] == 'perspective':
            gltf_lens = gltf_camera['perspective']
            lens = PerspectiveLens()
            aspect_ratio = gltf_lens.get(
                'aspectRatio',
                lens.get_aspect_ratio()
            )
            lens.set_fov(math.degrees(gltf_lens['yfov'] * aspect_ratio), math.degrees(gltf_lens['yfov']))
            lens.set_near_far(gltf_lens['znear'], gltf_lens['zfar'])
            lens.set_view_vector((0, 0, -1), (0, 1, 0))
            node.set_lens(lens)

        self.cameras[camid] = node

    def load_light(self, lightid, gltf_light, punctual=False):
        node = self.lights.get(lightid, None)
        lightname = gltf_light.get('name', 'light'+str(lightid))

        ltype = gltf_light['type']
        # Construct a new light if needed
        if node is None:
            if ltype == 'point':
                node = PointLight(lightname)
            elif ltype == 'directional':
                node = DirectionalLight(lightname)
            elif ltype == 'spot':
                node = Spotlight(lightname)
            else:
                print(f"Unsupported light type for light with name {lightname}: {gltf_light['type']}")
                node = PandaNode(lightname)

        # Update the light
        if punctual:
            # For PBR, attention should always be (1, 0, 1)
            if hasattr(node, 'attenuation'):
                node.attenuation = LVector3(1, 0, 1)

            intensity = gltf_light.get('intensity', 1) / PBR_WATTS_TO_LUMENS
            if ltype != 'directional':
                intensity *= 4 * math.pi

            # intensity = gltf_light.get('intensity', 1)

            if 'color' in gltf_light:
                node.set_color(LColor(*gltf_light['color'], w=1) * intensity)
            if 'range' in gltf_light:
                node.max_distance = gltf_light['range']
            if ltype == 'spot':
                spot = gltf_light.get('spot', {})
                inner = spot.get('innerConeAngle', 0)
                outer = spot.get('outerConeAngle', math.pi / 4)
                fov = math.degrees(outer) * 2
                node.get_lens().set_fov(fov, fov)

                if inner >= outer:
                    node.exponent = 0
                else:
                    # The value of exp was chosen empirically to give a smooth
                    # cutoff without straying too far from the spec; higher
                    # exponents will have a smoother cutoff but sharper falloff.
                    exp = 8 / 3
                    node.exponent = 2 * (math.pi * 0.5 / outer) ** exp
        else:
            if ltype == 'unsupported':
                lightprops = {}
            else:
                lightprops = gltf_light[ltype]

            if ltype in ('point', 'directional', 'spot'):
                node.set_color(LColor(*lightprops['color'], w=1))

            if ltype in ('point', 'spot'):
                att = LPoint3(
                    lightprops['constantAttenuation'],
                    lightprops['linearAttenuation'],
                    lightprops['quadraticAttenuation']
                )
                node.set_attenuation(att)

        self.lights[lightid] = node

    def load_physics_bullet(self, node_name, geomnode, shape_type, bounding_box, radius, height, intangible, gltf_rigidbody): # pylint: disable=line-too-long
        shape = None
        static = gltf_rigidbody is not None and 'static' in gltf_rigidbody and gltf_rigidbody['static']

        if shape_type == 'BOX':
            shape = bullet.BulletBoxShape(LVector3(*bounding_box) / 2.0)
        elif shape_type == 'SPHERE':
            shape = bullet.BulletSphereShape(max(bounding_box) / 2.0)
        elif shape_type == 'CAPSULE':
            shape = bullet.BulletCapsuleShape(radius, height - 2.0 * radius, bullet.ZUp)
        elif shape_type == 'CYLINDER':
            shape = bullet.BulletCylinderShape(radius, height, bullet.ZUp)
        elif shape_type == 'CONE':
            shape = bullet.BulletConeShape(radius, height, bullet.ZUp)
        elif shape_type == 'CONVEX_HULL':
            if geomnode:
                shape = bullet.BulletConvexHullShape()

                for geom in geomnode.get_geoms():
                    shape.add_geom(geom)
        elif shape_type == 'MESH':
            if geomnode:
                mesh = bullet.BulletTriangleMesh()
                for geom in geomnode.get_geoms():
                    mesh.add_geom(geom)
                shape = bullet.BulletTriangleMeshShape(mesh, dynamic=not static)
        else:
            print(f"Unknown collision shape ({shape_type}) for object ({node_name})")

        if shape is not None:
            if intangible:
                phynode = bullet.BulletGhostNode(node_name)
            else:
                phynode = bullet.BulletRigidBodyNode(node_name)
            phynode.add_shape(shape)
            if not static:
                mass = 1.0 if gltf_rigidbody is None else gltf_rigidbody.get('mass', 1.0)
                phynode.set_mass(mass)
            return phynode
        else:
            print(f"Could not create collision shape for object ({node_name})")

    def load_physics_builtin(self, node_name, geomnode, shape_type, bounding_box, radius, height, intangible):
        phynode = CollisionNode(node_name)

        solids = []

        if shape_type == 'BOX':
            solids.append(CollisionBox(Point3(0, 0, 0), *LVector3(*bounding_box) / 2.0))
        elif shape_type == 'SPHERE':
            solids.append(CollisionSphere(0, 0, 0, radius))
        elif shape_type in ('CAPSULE', 'CYLINDER', 'CONE'):
            if shape_type != 'CAPSULE':
                print(
                    f'Warning: builtin collisions do not support shape type {shape_type} for object {node_name}, '
                    'falling back to CAPSULE'
                )
            half_height = height / 2.0 - radius
            start = LPoint3(0, 0, -half_height)
            end = LPoint3(0, 0, half_height)
            solids.append(CollisionCapsule(start, end, radius))
        elif shape_type in ('MESH', 'CONVEX_HULL'):
            if shape_type != 'MESH':
                print(
                    f'Warning: builtin collisions do not support shape type {shape_type} for object {node_name}, '
                    'falling back to MESH'
                )
            if geomnode:
                for geom in geomnode.get_geoms():
                    vdata = self.read_vert_data(geom.get_vertex_data(), InternalName.get_vertex())
                    polygons = []
                    triangle_map = {}

                    for prim in geom.primitives:
                        prim_tmp = prim.decompose()

                        vertices = prim_tmp.get_vertex_list()
                        for i in range(0, len(vertices), 3):
                            pos0 = vdata[vertices[i]].xyz
                            pos1 = vdata[vertices[i + 1]].xyz
                            pos2 = vdata[vertices[i + 2]].xyz

                            # Find adjacent triangles lying on the same plane.
                            normal = (pos2 - pos0).cross(pos1 - pos0)
                            if not normal.normalize():
                                # Zero-area triangle.
                                continue

                            # Quantize the normal.
                            normal = (int(normal[0] * 0x1000 + 0.5),
                                      int(normal[1] * 0x1000 + 0.5),
                                      int(normal[2] * 0x1000 + 0.5))

                            key0 = (normal, pos1, pos0)
                            key1 = (normal, pos2, pos1)
                            key2 = (normal, pos0, pos2)
                            if key0 in triangle_map:
                                poly, pos3 = triangle_map[key0]
                                quad = (pos0, pos3, pos1, pos2)
                                if CollisionPolygon.verify_points(*quad) and \
                                   not CollisionPolygon(*quad).is_concave():
                                    poly[:] = quad
                                    del triangle_map[key0]
                                    del triangle_map[(normal, pos0, pos3)]
                                    del triangle_map[(normal, pos3, pos1)]
                                    continue

                            if key1 in triangle_map:
                                poly, pos3 = triangle_map[key1]
                                quad = (pos1, pos3, pos2, pos0)
                                if CollisionPolygon.verify_points(*quad) and \
                                   not CollisionPolygon(*quad).is_concave():
                                    poly[:] = quad
                                    del triangle_map[key1]
                                    del triangle_map[(normal, pos1, pos3)]
                                    del triangle_map[(normal, pos3, pos2)]
                                    continue

                            if key2 in triangle_map:
                                poly, pos3 = triangle_map[key2]
                                quad = (pos2, pos3, pos0, pos1)
                                if CollisionPolygon.verify_points(*quad) and \
                                   not CollisionPolygon(*quad).is_concave():
                                    poly[:] = quad
                                    del triangle_map[key2]
                                    del triangle_map[(normal, pos2, pos3)]
                                    del triangle_map[(normal, pos3, pos0)]
                                    continue

                            if triangle_map.get((normal, pos0, pos1), (None, None))[1] != pos2:
                                poly = [pos0, pos1, pos2]
                                triangle_map[(normal, pos0, pos1)] = (poly, pos2)
                                triangle_map[(normal, pos1, pos2)] = (poly, pos0)
                                triangle_map[(normal, pos2, pos0)] = (poly, pos1)
                                polygons.append(poly)

                    solids.extend(CollisionPolygon(*poly) for poly in polygons)

        else:
            print(f"Unknown collision shape ({shape_type}) for object ({node_name})")

        for solid in solids:
            if intangible:
                solid.set_tangible(False)
            phynode.add_solid(solid)

        if phynode.solids:
            return phynode
        else:
            print(f"Could not create collision shape for object ({node_name})")
