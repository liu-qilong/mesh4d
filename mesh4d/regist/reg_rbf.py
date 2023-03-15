"""Replace the simple nearest point alignment displacement field estimation workflow with Radial Basis Function (RBF) based approach."""
from __future__ import annotations
from typing import Type, Union, Iterable

import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import RBFInterpolator

import mesh4d
import mesh4d.config.param
from mesh4d import obj4d, field, utils

class Trans_Nonrigid_RBF(field.Trans_Nonrigid):
    def regist(self, landmark_name: str, k_nbr: int = 1, **kwargs):
        landmarks_source = self.source.kps_group[landmark_name].get_points_coord()
        landmarks_target = self.target.kps_group[landmark_name].get_points_coord()

        field = RBFInterpolator(landmarks_source, landmarks_target)
        self.parse(field, k_nbr)

    def parse(self, field, k_nbr: int = 1):
        self.source_points = self.source.get_vertices()
        shift_points = field(self.source_points)

        target_points = self.target.get_vertices()
        tree = KDTree(target_points)
        _, idx = tree.query(shift_points, k=k_nbr)

        if k_nbr == 1:
            self.deform_points = target_points[idx]

        else:
            deform_points = np.take(target_points, idx, axis=0)
            self.deform_points = np.mean(deform_points, axis=1)

        self.disp = self.deform_points - self.source_points
        self.search_tree = KDTree(self.source_points)
        

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