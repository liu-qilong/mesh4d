"""Replace the Coherent Point Drift (CPD) based displacement field estimation workflow with simple nearest point alignment approach."""
from __future__ import annotations
from typing import Type, Union, Iterable

import copy
import numpy as np
import pyvista as pv
from scipy.spatial import KDTree

import UltraMotionCapture
import UltraMotionCapture.config.param
from UltraMotionCapture import obj3d, obj4d, field

class Obj3d_Closest(obj3d.Obj3d_Deform):
    """Derived from :class:`UltraMotionCapture.obj3d.Obj3d_Deform` and replace the displacement field estimation as simple nearest point alignment approach.
    
    Attention
    ---
    Since the nearest point search realised with :class:`scipy.spatial.KDTree` is very efficient, all vertices form 3dMD mesh can be used as the sampling points. Therefore, the :attr:`sample_num` argument is removed from the initialisation method.
    """
    def __init__(
        self,
        filedir: str,
        scale_rate: float = 1e-3,
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

class Trans_Nonrigid_Closest(field.Trans_Nonrigid):
    """Derived from :class:`UltraMotionCapture.field.Trans_Nonrigid` and replace the displacement field estimation as simple nearest point alignment approach.
    """
    def regist(self, **kwargs):
        """Align every point from the source object to the nearest point in the target object and use it a this point's displacement.
        """
        self.source_points = obj3d.pcd2np(self.source)
        target_points = obj3d.pcd2np(self.target)

        tree = KDTree(target_points)
        _, idx = tree.query(self.source_points)
        self.deform_points = target_points[idx]
        
        _, idx = tree.query(self.source_points)
        self.deform_points = target_points[idx]

        self.disp = self.deform_points - self.source_points
        self.search_tree = KDTree(self.source_points)


class Obj4d_Closest(obj4d.Obj4d_Deform):
    """Derived from :class:`UltraMotionCapture.obj4d.Obj4d_Deform` and replace the displacement field estimation as simple nearest point alignment approach.
    """
    def process_nonrigid_dynamic(self, idx_source: int, idx_target: int, **kwargs):
        """Estimate the non-rigid transformation of the added 3D object. The lastly added 3D object is used as source object and the newly added 3D object as the target object.

        Parameters
        ---
        idx_source
            the index of the source 3D object in :code:`self.obj_ls`.
        idx_target
            the index of the target 3D object in :code:`self.obj_ls`.

        Attention
        ---
        The estimated transformation is load to the source object, via its :meth:`UltraMotionCapture.obj3d.Obj3d_Deform.set_trans_nonrigid` method.
        
        Attention
        ---
        Called by :meth:`add_obj`.
        """
        trans = Trans_Nonrigid_Closest(
            source_obj=self.obj_ls[idx_source],
            target_obj=self.obj_ls[idx_target],
        )
        trans.regist(**kwargs)
        self.obj_ls[idx_source].set_trans_nonrigid(trans)