"""Replace the simple nearest point alignment displacement field estimation workflow with Coherent Point Drift (CPD) based approach."""
from __future__ import annotations
from typing import Type, Union, Iterable

import copy
import pyvista as pv
from probreg import cpd
from scipy.spatial import KDTree

import mesh4d
import mesh4d.config.param
from mesh4d import obj3d, obj4d, field

class Obj3d_CPD(obj3d.Obj3d_Deform):
    """Derived from :class:`mesh4d.obj3d.Obj3d_Deform` and replace the displacement field estimation as Coherent Point Drift (CPD) based approach.
    
    Parameters
    ---
    filedir
        the direction of the 3D object.
    sample_num
        the number of the points sampled from the mesh to construct the point cloud.
        
        Attention
        ---
        Since the Coherent Point Drift (CPD) is not very efficient, the number of the sampling points used to estimate the displacement field should relatively small. The default value is :code:`3000`.
    mode
        
        - :code:`load` the default mode is load from a file.
        - :code:`empty` create a 3D object without any 3D data.
    """
    def __init__(
        self,
        filedir: str = '',
        sample_num: int = 3000,
        mode: str = "load"
    ):
        if mode == "load":
            # revise Obj3d __init__()
            self.mesh = obj3d.pvmesh_fix_disconnect(pv.read(filedir))
            self.texture = pv.read_texture(filedir.replace('.obj', '.jpg'))
            self.pcd = obj3d.pvmesh2pcd_pro(self.mesh, sample_num)
            # self.pcd = pvmesh2pcd(self.mesh, sample_num)

            # follows Obj3d_Kps, Obj3d_Deform __init__()
            self.kps_group = {}
            self.trans_rigid = None
            self.trans_nonrigid = None
        
        elif mode == "empty":
            self.mesh = None
            self.texture = None
            self.pcd = None

            self.kps_group = {}
            self.trans_rigid = None
            self.trans_nonrigid = None

class Trans_Nonrigid_CPD(field.Trans_Nonrigid):
    """Derived from :class:`mesh4d.field.Trans_Nonrigid` and replace the displacement field estimation as Coherent Point Drift (CPD) based approach.
    """
    def regist(self, **kwargs):
        """The registration method.

        Parameters
        ---
        **kwargs
            Configurations parameters of the registration.
            
            See Also
            --------
            `probreg.cpd.registration_cpd <https://probreg.readthedocs.io/en/latest/probreg.html?highlight=registration_cpd#probreg.cpd.registration_cpd>`_
        """
        tf_param, _, _ = cpd.registration_cpd(
            self.source.pcd, self.target.pcd, 'nonrigid', **kwargs
        )
        self.parse(tf_param)
        
        if mesh4d.output_msg:
            print("registered 1 nonrigid transformation")

    def parse(self, tf_param):
        """Parse the registration result to provide :attr:`self.source_points`, :attr:`self.deform_points`, and :attr:`self.disp`. Called by :meth:`regist`.
        
        Parameters
        ---
        tf_param
            Attention
            ---
            At current stage, the default registration method is Coherent Point Drift (CPD) method realised by :mod:`probreg` package. Therefore the accepted transformation object to be parse is derived from :class:`cpd.CoherentPointDrift`. Transformation object provided by other registration method shall be tested in future development.
        """
        deform = copy.deepcopy(self.source.pcd)
        deform.points = tf_param.transform(deform.points)
        
        self.deform_points = obj3d.pcd2np(deform)
        self.source_points = obj3d.pcd2np(self.source.pcd)
        self.disp = self.deform_points - self.source_points
        self.search_tree = KDTree(self.source_points)


class Obj4d_CPD(obj4d.Obj4d_Deform):
    """Derived from :class:`mesh4d.obj4d.Obj4d_Deform` and replace the displacement field estimation as Coherent Point Drift (CPD) based approach
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
        The estimated transformation is load to the source object, via its :meth:`mesh4d.obj3d.Obj3d_Deform.set_trans_nonrigid` method.
        
        Attention
        ---
        Called by :meth:`add_obj`."""
        trans = Trans_Nonrigid_CPD(
            source_obj=self.obj_ls[idx_source],
            target_obj=self.obj_ls[idx_target],
        )
        trans.regist(**kwargs)
        self.obj_ls[idx_source].set_trans_nonrigid(trans)