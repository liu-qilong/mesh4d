import copy
import numpy as np
import open3d as o3d
from probreg import cpd, bcpd

import obj3d


class obj4d(object):
    def __init__(self, enable_rigid=True, enable_nonrigid=True):
        self.obj_ls = []

        self.enable_rigid = enable_rigid
        self.enable_nonrigid = enable_nonrigid

        if enable_rigid:
            self.rot_4d = []

        if enable_nonrigid:
            self.disp_4d = []

    def add_obj(self, *obj):
        """add object(s) and parse its 4d movement between adjacent frames"""
        pass

    def remove_obj(self, idx):
        """remove object(s) according to index(es)"""
        pass

    def rigid_regist(self):
        pass

    def rigid_fix(self):
        pass

    def parse_rot(self):
        pass

    def nonrigid_regist(self):
        pass

    def nonrigid_fix(self):
        pass

    def parse_disp(self):
        pass

    def show(self):
        o3d.visualization.draw_geometries([

        ])

    def get_o3d(self):
        pass

if __name__ == '__main__':
    o4 = obj4d()