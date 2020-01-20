import os
import sys

from direct.showbase.ShowBase import ShowBase
import panda3d.core as p3d

import simplepbr

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

        simplepbr.init()

        gltf.patch_loader(self.loader)

        infile = p3d.Filename.from_os_specific(os.path.abspath(sys.argv[1]))
        p3d.get_model_path().prepend_directory(infile.get_dirname())

        self.model_root = self.loader.load_model(infile, noCache=True)

        self.accept('escape', sys.exit)
        self.accept('q', sys.exit)
        self.accept('w', self.toggle_wireframe)
        self.accept('t', self.toggle_texture)
        self.accept('shift-l', self.model_root.ls)
        self.accept('shift-a', self.model_root.analyze)

        if not self.model_root.find('**/+Light'):
            self.light = self.render.attach_new_node(p3d.PointLight('light'))
            self.light.set_pos(-5, 5, 5)
            self.render.set_light(self.light)

        self.cam.set_pos(-6, 6, 6)
        self.cam.look_at(self.model_root)

        self.model_root.reparent_to(self.render)

        if self.model_root.find('**/+Character'):
            self.anims = p3d.AnimControlCollection()
            p3d.autoBind(self.model_root.node(), self.anims, ~0)
            if self.anims.get_num_anims() > 0:
                self.anims.get_anim(0).loop(True)

def main():
    App().run()

if __name__ == '__main__':
    main()
