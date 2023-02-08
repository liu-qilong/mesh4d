from __future__ import annotations
from typing import Type, Union, Iterable

import os
import copy
import numpy as np
import open3d as o3d
import pyvista as pv
from scipy.spatial import KDTree

import UltraMotionCapture
import UltraMotionCapture.config.param
from UltraMotionCapture import obj3d, obj4d

class Obj3d_Closest(obj3d.Obj3d_Deform):
    """tbf"""
    def __init__(
        self,
        filedir: str,
        scale_rate: float = 1e-3,
        sample_num: None = None,
    ):
        # revise Obj3d __init__()
        self.mesh = obj3d.pvmesh_fix_disconnect(pv.read(filedir))
        self.texture = pv.read_texture(filedir.replace('.obj', '.jpg'))
        self.scale_rate = scale_rate

        self.mesh.scale(self.scale_rate, inplace=True)
        self.pcd = obj3d.np2pcd(self.mesh.points)

        # follows Obj3d_Kps, Obj3d_Deform __init__()
        self.kps_group = {}
        self.trans_rigid = None
        self.trans_nonrigid = None

class Obj4d_Closest(obj4d.Obj4d_Deform):
    """tbf"""
    def process_nonrigid_dynamic(self, idx_source, idx_target, **kwargs):
        pass