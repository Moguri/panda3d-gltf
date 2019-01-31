import os
import sys

from direct.actor.Actor import Actor
from direct.showbase.ShowBase import ShowBase
import panda3d.core as p3d

import gltf

p3d.load_prc_file_data(
    __file__,
    'window-size 1024 768\n'
)


class App(ShowBase):
    def __init__(self):
        if len(sys.argv) < 2:
            print("Missing input file")
            sys.exit(1)

        super().__init__()

        gltf.patch_loader(self.loader)

        infile = p3d.Filename.from_os_specific(os.path.abspath(sys.argv[1]))
        p3d.get_model_path().prepend_directory(infile.get_dirname())

        self.model_root = self.loader.load_model(infile)

        self.accept('escape', sys.exit)
        self.accept('q', sys.exit)

        viewer_dir = os.path.dirname(__file__)
        vertfname = os.path.join(viewer_dir, 'simplepbr.vert')
        fragfname = os.path.join(viewer_dir, 'simplepbr.frag')
        pbrshader = p3d.Shader.load(p3d.Shader.SL_GLSL, vertex=vertfname, fragment=fragfname)
        self.render.set_shader(pbrshader)

        self.light = self.render.attach_new_node(p3d.PointLight('light'))
        self.light.set_pos(-5, 5, 5)
        self.render.set_light(self.light)

        self.cam.set_pos(-10, 10, 10)

        if self.model_root.find('**/+Character'):
            self.actor = Actor(self.model_root)
            self.actor.reparent_to(self.render)
            self.actor.loop('anim0')
            self.cam.look_at(self.actor)
        else:
            self.model_root.reparent_to(self.render)
            self.cam.look_at(self.model_root)

def main():
    App().run()

if __name__ == '__main__':
    main()
