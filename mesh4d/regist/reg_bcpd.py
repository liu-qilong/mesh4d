from __future__ import annotations
from typing import Type, Union, Iterable

import copy
import open3d as o3d
import pyvista as pv
from probreg import bcpd
from scipy.interpolate import RBFInterpolator

import mesh4d
import mesh4d.config.param
from mesh4d import obj3d, obj4d, field

class Trans_Nonrigid_BCPD(field.Trans_Nonrigid):
    def regist(self, sample_num: int = 1000, field_nbr: int = 100, scale_rate: float = 1, **kwargs):
        # registration
        source_pcd = obj3d.np2pcd(self.source.get_sample_points(sample_num) / scale_rate)
        target_pcd = obj3d.np2pcd(self.target.get_sample_points(sample_num) / scale_rate)

        tf_param = bcpd.registration_bcpd(
            source=source_pcd, 
            target=target_pcd,
            **kwargs
            )
        
       # parse
        deform = copy.deepcopy(source_pcd)
        deform.points = tf_param.transform(deform.points)
        self.deform_points = obj3d.pcd2np(deform) * scale_rate

        self.source_points = self.source.get_sample_points(sample_num)
        self.field = RBFInterpolator(self.source_points, self.deform_points, neighbors=field_nbr)


class Obj4d_BCPD(obj4d.Obj4d_Deform):
    def __init__(self, regist_points_num: int = 1000, **kwargs):
        obj4d.Obj4d_Deform.__init__(self, **kwargs)
        self.regist_points_num = regist_points_num

    def process_nonrigid_dynamic(self, idx_source: int, idx_target: int, **kwargs):
        trans = Trans_Nonrigid_BCPD(
            source_obj=self.obj_ls[idx_source],
            target_obj=self.obj_ls[idx_target],
        )
        trans.regist(sample_num=self.regist_points_num, **kwargs)
        self.obj_ls[idx_source].set_trans_nonrigid(trans)