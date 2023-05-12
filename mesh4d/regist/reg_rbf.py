"""Replace the simple nearest point alignment displacement field estimation workflow with Radial Basis Function (RBF) based approach."""
from __future__ import annotations
from typing import Type, Union, Iterable

import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import RBFInterpolator

import mesh4d
import mesh4d.config.param
from mesh4d import obj3d, obj4d, field, utils
from mesh4d.analyse import measure

class Trans_Nonrigid_RBF(field.Trans_Nonrigid):
    def regist(self, landmark_name: str, field_nbr: int = 100, **kwargs):
        self.field_nbr = field_nbr

        landmarks_source = self.source.kps_group[landmark_name].get_points_coord()
        landmarks_target = self.target.kps_group[landmark_name].get_points_coord()

        landmarks_field = RBFInterpolator(landmarks_source, landmarks_target, **kwargs)
        self.parse(landmarks_field)

    def parse(self, landmarks_field):
        self.source_points = self.source.get_vertices()
        shift_points = landmarks_field(self.source_points)

        self.deform_points = measure.search_nearest_points_plane(self.target.mesh, shift_points)

        self.disp = self.deform_points - self.source_points
        self.field = RBFInterpolator(self.source_points, self.deform_points, neighbors=self.field_nbr)

    def shift_points(self, points: np.array, **kwargs) -> np.array:
        return self.field(points)
    
    def invert(self):
        trans_invert = type(self)(source_obj=self.target, target_obj=self.source)
        trans_invert.field_nbr = self.field_nbr
        trans_invert.source_points = self.deform_points
        trans_invert.deform_points = self.source_points
        trans_invert.disp = -self.disp
        trans_invert.field = RBFInterpolator(trans_invert.source_points, trans_invert.deform_points, neighbors=trans_invert.field_nbr)

        return trans_invert


class Obj4d_RBF(obj4d.Obj4d_Deform):
    def regist(self, landmark_name: str, **kwargs):
        """Implement registration among 3D objects in :attr:`self.obj_ls`.

        Parameters
        ---
        landmark_name
            the keyword of the controlling landmarks in :attr:`self.kps_group`.

            Warning
            -----
            The landmarks marker set object must have been attach to the object (:meth:`~mesh4d.obj4d.load_markerset`) before registration.
        **kwargs
            configuration parameters for the registration and the configuration parameters of the base classes (:class:`Obj3d` and :class:`Obj3d_Kps`)'s :meth:`add_obj` method can be passed in via :code:`**kwargs`.
        """
        reg_num = len(self.obj_ls)
        
        for idx in range(reg_num):
            if idx == 0:
                self.process_first_obj()
                continue

            if self.enable_rigid:
                self.process_rigid_dynamic(idx - 1, idx, **kwargs)  # aligned to the previous one

            if self.enable_nonrigid:
                self.process_nonrigid_dynamic(idx - 1, idx, landmark_name, **kwargs)  # aligned to the later one
            
            if mesh4d.output_msg:
                percent = (idx + 1) / reg_num
                utils.progress_bar(percent, back_str=" registered the {}-th frame".format(idx))
            

    def process_nonrigid_dynamic(self, idx_source: int, idx_target: int, landmark_name: str, **kwargs):
        trans = Trans_Nonrigid_RBF(
            source_obj=self.obj_ls[idx_source],
            target_obj=self.obj_ls[idx_target],
        )
        trans.regist(landmark_name, **kwargs)
        self.obj_ls[idx_source].set_trans_nonrigid(trans)