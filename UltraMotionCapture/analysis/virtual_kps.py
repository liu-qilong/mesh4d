"""tbf"""
from __future__ import annotations
from typing import Type, Union, Iterable

import numpy as np

import UltraMotionCapture
from UltraMotionCapture import obj3d
from UltraMotionCapture import obj4d
from UltraMotionCapture import kps
from UltraMotionCapture import field

class Obj4d_VKps(obj4d.Obj4d_Deform):
    

    def regist_verify(self, markerset_gt: kps.MarkerSet) -> dict:
        """Estimate the accuracy of the registration, with frame-wise & whole trail mean error and standard deviation.

        Parameters
        ---
        markerset
            tbf
        
        Note
        ---
        As discussed in :mod:`UltraMotionCapture.field`, The purpose of point cloud registration is revealing the displacement field between different frames of 3D objects.
        
        Once the displacement field is revealed, it can be used to predict an arbitrary point's movement. Note the point's location in current frame as :math:`\\boldsymbol x`, its predicted location in the next frame as :math:`\\boldsymbol x_{pd}`, and its actual location in the next frame as :math:`\\boldsymbol x_{gt}`. Then the registration error can be defined as:
        
        .. math::
            E = \lVert\\boldsymbol x_{pd} - \\boldsymbol x_{gt} \\rVert_2

        where :math:`\lVert \cdot \\rVert_2` is the L2 norm that calculates the Euclidean distance.

        Returns
        ---
        :class:`dict`
            A dictionary that contains the comparison result:

            - :code:`'diff_ls'`: a list of the original frame-wise estimation result from :meth:`UltraMotionCapture.kps.Kps_Deform.compare_with_groundtruth`.
            - :code:`'dist_mean'`: the mean error of the whole trail error.
            - :code:`'dist_std'`: the standard deviation of the whole trail error.
            - :code:`'diff_str'`: a string in form of :code:`'dist_mean ± dist_std (mm)`.
        """
        # estimate frame-wise registration error
        # and collect all dist data in dist_ls
        diff_ls = []
        for idx in range(len(self.obj_ls) - 1):
            obj = self.obj_ls[idx]
            obj_next = self.obj_ls[idx+1]
            diff = obj.kps.compare_with_groundtruth(obj_next.kps)
            diff_ls.append(diff)

            if UltraMotionCapture.output_msg:
                print("estimated error of frame {}: {}".format(idx, diff['diff_str']))

        # estimate whole period registration error
        dist_ls = []
        for diff in diff_ls:
            dist_ls.append(diff['dist'])

        dist_array = np.concatenate(dist_ls)
        dist_mean = np.mean(dist_array)
        dist_std = np.std(dist_array)

        # combine the estimation result and print whole period error
        diff_dict = {
            'diff_ls': diff_ls,
            'dist_mean': dist_mean,
            'dist_std': dist_std,
            'diff_str': "diff = {:.3} ± {:.3} (mm)".format(dist_mean, dist_std),
        }

        if UltraMotionCapture.output_msg:
            print("whole duration error: {}".format(diff_dict['diff_str']))

        return diff_dict