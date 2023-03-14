"""Replace the simple nearest point alignment displacement field estimation workflow with Extend Coherent Point Drift (ECPD) based approach. Comparing with basic CPD, ECPD considers prior correspondence to improve the registration quality."""
from __future__ import annotations
from typing import Type, Union, Iterable

import copy
import open3d as o3d
import pyvista as pv
from probreg import cpd
from scipy.spatial import KDTree

import mesh4d
import mesh4d.config.param
from mesh4d import obj3d, obj4d, field, kps, utils

class Obj3d_ECPD(obj3d.Obj3d_Deform):
    """Derived from :class:`mesh4d.obj3d.Obj3d_Deform` and replace the displacement field estimation as Extended Coherent Point Drift (ECPD) based approach.
    
    Parameters
    ---
    filedir
        the direction of the 3D object.
    mode
        
        - :code:`load` the default mode is load from a file.
        - :code:`empty` create a 3D object without any 3D data.
    """
    def attach_control_landmarks(self, kps: Type[kps.Kps]):
        """Attach controlling landmarks to the 3D object.

        Attention
        ---
        This step must be completed before it's added to a 4D object, since the controlling landmarks will be used to construct the RBF motion model in the adding procedure.

        Parameters
        ---
        kps
            controlling landmarks of this frame.
        """
        self.control_landmarks = kps

class Trans_Nonrigid_ECPD(field.Trans_Nonrigid):
    """Derived from :class:`mesh4d.field.Trans_Nonrigid` and replace the displacement field estimation as Coherent Point Drift (CPD) based approach.
    """
    def regist(self, sample_num: int = 3000, **kwargs):
        """The registration method.

        Parameters
        ---
        sample_num
            the number of the points sampled from the mesh to construct the point cloud.
            
            Attention
            ---
            Since the Coherent Point Drift (CPD) is not very efficient, the number of the sampling points used to estimate the displacement field should relatively small. The default value is :code:`3000`.
        **kwargs
            Configurations parameters of the registration.
            
        See Also
        --------
        `probreg.cpd.registration_cpd <https://probreg.readthedocs.io/en/latest/probreg.html?highlight=registration_cpd#probreg.cpd.registration_cpd>`_
        """
        # sample source & target mesh
        source_points = self.source.get_sample_points(sample_num)
        target_points = self.target.get_sample_points(sample_num)

        # get source & target point clouds
        def get_landmarks_idx(mesh_points, landmarks_kps):
            tree = KDTree(mesh_points)
            landmarks_points = landmarks_kps.get_points_coord()
            _, idx = tree.query(landmarks_points)
            return idx

        idx_source = get_landmarks_idx(source_points, self.source.control_landmarks)
        idx_target = get_landmarks_idx(source_points, self.target.control_landmarks)

        # get source & target point clouds
        source_pcd = obj3d.np2pcd(source_points)
        target_pcd = obj3d.np2pcd(target_points)

        tf_param, _, _ = cpd.registration_cpd(
            source=source_pcd, 
            target=target_pcd, 
            tf_type_name='nonrigid_constrained',
            idx_source=idx_source,
            idx_target=idx_target,
            **kwargs
            )
        
        self.parse(tf_param, source_pcd)

    def parse(self, tf_param, source_pcd: o3d.cpu.pybind.geometry.PointCloud):
        """Parse the registration result to provide :attr:`self.source_points`, :attr:`self.deform_points`, and :attr:`self.disp`. Called by :meth:`regist`.
        
        Parameters
        ---
        tf_param
            Attention
            ---
            At current stage, the default registration method is Coherent Point Drift (CPD) method realised by :mod:`probreg` package. Therefore the accepted transformation object to be parse is derived from :class:`cpd.CoherentPointDrift`. Transformation object provided by other registration method shall be tested in future development.
        source_pcd
            :mod:`open3d` point cloud object sampled from the source mesh.
        """
        self.source_points = obj3d.pcd2np(source_pcd)

        deform = copy.deepcopy(source_pcd)
        deform.points = tf_param.transform(deform.points)
        self.deform_points = obj3d.pcd2np(deform)

        self.disp = self.deform_points - self.source_points
        self.search_tree = KDTree(self.source_points)


class Obj4d_ECPD(obj4d.Obj4d_Deform):
    """Derived from :class:`mesh4d.obj4d.Obj4d_Deform` and replace the displacement field estimation as Coherent Point Drift (CPD) based approach.

    Parameters
    ---
    regist_points_num
        the number of the points sampled from the mesh for registration.
            
            Attention
            ---
            Since the Coherent Point Drift (CPD) is not very efficient, the number of the sampling points used to estimate the displacement field should relatively small. The default value is :code:`3000`.
    """
    def __init__(self, regist_points_num: int = 3000, **kwargs):
        obj4d.Obj4d_Deform.__init__(self, **kwargs)
        self.regist_points_num = regist_points_num

    def add_obj(self, *objs: Iterable[Type[obj3d.Obj3d_Deform]], landmarks: kps.MarkerSet, **kwargs):
        """Add object(s) and attach key points (:class:`mesh4d.kps.Kps`) to each of the 3D object via Vicon motion capture data (:attr:`markerset`). And then implement the activated transformation estimation.

        Parameters
        ---
        *objs
            unspecified number of 3D objects.

            .. warning::
            
                The 3D objects' class must derived from :class:`mesh4d.obj3d.Obj3d_Deform`.

            .. seealso::

                About the :code:`*` symbol and its effect, please refer to `*args and **kwargs - Python Tips <https://book.pythontips.com/en/latest/args_and_kwargs.html>`_
        
        **kwargs
            configuration parameters for the registration and the configuration parameters of the base classes (:class:`Obj3d` and :class:`Obj3d_Kps`)'s :meth:`add_obj` method can be passed in via :code:`**kwargs`.
        """
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
            
            if mesh4d.output_msg:
                percent = (idx - reg_start_index + 1) / (reg_end_index - reg_start_index + 1)
                utils.progress_bar(percent, back_str=" adding the {}-th 3d object".format(idx))

    def process_nonrigid_dynamic(self, idx_source: int, idx_target: int, **kwargs):
        trans = Trans_Nonrigid_ECPD(
            source_obj=self.obj_ls[idx_source],
            target_obj=self.obj_ls[idx_target],
        )
        trans.regist(sample_num=self.regist_points_num, **kwargs)
        self.obj_ls[idx_source].set_trans_nonrigid(trans)