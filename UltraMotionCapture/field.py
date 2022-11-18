import copy
import numpy as np
import open3d as o3d
from probreg import cpd

import sys
sys.path.insert(0, '../')

from UltraMotionCapture import obj3d

class Trans(object):
    def __init__(self, source_obj, target_obj, **kwargs):
        self.source = source_obj.pcd
        self.target = target_obj.pcd


class Trans_Rigid(Trans):
    def regist(self, method=cpd.registration_cpd, **kwargs):
        tf_param, _, _ = method(
            self.source, self.target, 'rigid', **kwargs
        )
        self.__parse(tf_param)
        self.__fix()
        print("registered 1 rigid transformation")

    def __parse(self, tf_param):
        self.rot = tf_param.rot
        self.scale = tf_param.scale
        self.t = tf_param.t

    def __fix(self):
        if np.abs(self.scale - 1) > 0.05:
            print("warnning: large rigid scale {}".format(self.scale))

    def show(self):
        o3d.visualization.draw_geometries([
            self.source,
            copy.deepcopy(self.deform).translate((10, 0, 0)),
            copy.deepcopy(self.target).translate((30, 0, 0))
        ])


class Trans_Nonrigid(Trans):
    def regist(self, method=cpd.registration_cpd, **kwargs):
        tf_param, _, _ = method(
            self.source, self.target, 'nonrigid', **kwargs
        )
        self.__parse(tf_param)
        self.__fix()
        print("registered 1 nonrigid transformation")

    def __parse(self, tf_param):
        deform = copy.deepcopy(self.source)
        deform.points = tf_param.transform(deform.points)
        
        self.deform_points = obj3d.pcd2np(deform)
        self.source_points = obj3d.pcd2np(self.source)
        self.disp = self.deform_points - self.source_points

    def __fix(self):
        deform_fix_points = []

        for n in range(len(self.deform_points)):
            deform_fix_points.append(
                obj3d.search_nearest_point(self.deform_points[n], self.target_points)
            )

        self.deform_points = deform_fix_points
        self.disp = self.deform_points - self.source_points

    def shift_points(self, points):
        idxs = []
        for point in points:
            idx = obj3d.search_nearest_point_idx(point, self.source_points)
            idxs.append(idx)
        return self.deform_points[idx]
