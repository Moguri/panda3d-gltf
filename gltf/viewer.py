import os
import sys

from direct.actor.Actor import Actor
from direct.showbase.ShowBase import ShowBase
import panda3d.core as p3d

import gltf

import simplepbr

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

        simplepbr.init()

        gltf.patch_loader(self.loader)

        infile = p3d.Filename.from_os_specific(os.path.abspath(sys.argv[1]))
        p3d.get_model_path().prepend_directory(infile.get_dirname())

        self.model_root = self.loader.load_model(infile, noCache=True)

        self.accept('escape', sys.exit)
        self.accept('q', sys.exit)

        if not self.model_root.find('**/+Light'):
            self.light = self.render.attach_new_node(p3d.PointLight('light'))
            self.light.set_pos(-5, 5, 5)
            self.render.set_light(self.light)

        self.cam.set_pos(-6, 6, 6)
        self.cam.look_at(self.model_root)

        if self.model_root.find('**/+Character'):
            self.actor = Actor(self.model_root)
            self.actor.reparent_to(self.render)
            anims = self.actor.get_anim_names()
            if anims:
                self.actor.loop(anims[0])
        else:
            self.model_root.reparent_to(self.render)


def main():
    App().run()

if __name__ == '__main__':
    main()
