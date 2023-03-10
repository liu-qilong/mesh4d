"""Replace the simple nearest point alignment displacement field estimation workflow with Radial Basis Function (RBF) based approach."""
from __future__ import annotations
from typing import Type, Union, Iterable

import copy
from scipy.spatial import KDTree
from scipy.interpolate import RBFInterpolator

import mesh4d
import mesh4d.config.param
from mesh4d import kps, obj3d, obj4d, field

class Obj3d_RBF(obj3d.Obj3d_Deform):
    def attach_control_landmarks(self, kps: Type[kps.Kps]):
        self.control_landmarks = kps

class Trans_Nonrigid_RBF(field.Trans_Nonrigid):
    def regist(self, **kwargs):
        landmarks_source = self.source.control_landmarks.get_points_coord()
        landmarks_target = self.target.control_landmarks.get_points_coord()
        field = RBFInterpolator(landmarks_source, landmarks_target)
        self.parse(field)
        
        if mesh4d.output_msg:
            print("registered 1 nonrigid transformation")

    def parse(self, field):
        self.source_points = obj3d.pcd2np(self.source.pcd)
        shift_points = field(self.source_points)
        
        target_points = obj3d.pcd2np(self.target.pcd)
        tree = KDTree(target_points)
        _, idx = tree.query(shift_points)
        self.deform_points = target_points[idx]
        
        self.disp = self.deform_points - self.source_points
        self.search_tree = KDTree(self.source_points)
        

class Obj4d_RBF(obj4d.Obj4d_Deform):
    def add_obj(self, *objs: Iterable[Type[obj3d.Obj3d_Deform]], landmarks: kps.MarkerSet, **kwargs):
        # follows Obj3d_Kps, Obj4d_Deform add_obj()
        reg_start_index = len(self.obj_ls)
        obj4d.Obj4d_Kps.add_obj(self, *objs, **kwargs)
        reg_end_index = len(self.obj_ls) - 1
        
        for idx in range(reg_start_index, reg_end_index + 1):
            # attach control landmarks
            time = self.start_time + idx / self.fps
            kps = landmarks.get_time_coord(time)
            self.obj_ls[idx].attach_control_landmarks(kps)

            # follows Obj3d_Kps, Obj4d_Deform add_obj()
            if idx == 0:
                self.process_first_obj()
                continue

            if self.enable_rigid:
                self.process_rigid_dynamic(idx - 1, idx, **kwargs)  # aligned to the previous one

            if self.enable_nonrigid:
                self.process_nonrigid_dynamic(idx - 1, idx, **kwargs)  # aligned to the later one
            

    def process_nonrigid_dynamic(self, idx_source: int, idx_target: int, **kwargs):
        trans = Trans_Nonrigid_RBF(
            source_obj=self.obj_ls[idx_source],
            target_obj=self.obj_ls[idx_target],
        )
        trans.regist(**kwargs)
        self.obj_ls[idx_source].set_trans_nonrigid(trans)