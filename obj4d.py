import copy
import numpy as np
import open3d as o3d
from probreg import cpd

import obj3d

class Trans_hl(object):
    def __init__(self, source_obj, target_obj, *args, **kwargs):
        self.source = source_obj.pcd_hd
        self.target = target_obj.pcd_ld

        self.source_points = obj3d.pcd2np(self.source)
        self.target_points = obj3d.pcd2np(self.target)

        self.regist(*args, **kwargs)
        self.fix()

    def parse(self):
        pass

    def fix(self):
        pass

    def show(self):
        o3d.visualization.draw_geometries([
            self.source,
            copy.deepcopy(self.deform).translate((10, 0, 0)),
            copy.deepcopy(self.deform_fix).translate((20, 0, 0)),
            copy.deepcopy(self.target).translate((30, 0, 0))
        ])

    def get_o3ds(self):
        objs = [
            copy.deepcopy(self.source),
            copy.deepcopy(self.deform).translate((10, 0, 0)),
            copy.deepcopy(self.deform_fix).translate((20, 0, 0)),
            copy.deepcopy(self.target).translate((30, 0, 0))
        ]
        return objs


class Trans_hl_Rigid(Trans_hl):
    def regist(self, method=cpd.registration_cpd, *args, **kwargs):
        tf_param, _, _ = method(
            self.source, self.target, 'rigid', *args, **kwargs
        )
        self.parse(tf_param)
        print("registered 1 rigid transformation")

    def parse(self, tf_param):
        self.rot = tf_param.rot
        self.scale = tf_param.scale
        self.t = tf_param.t

    def fix(self):
        if np.abs(self.scale - 1) > 0.05:
            print("warnning: large rigid scale {}".format(self.scale))


class Trans_hl_Nonrigid(Trans_hl):
    def regist(self, method=cpd.registration_cpd, *args, **kwargs):
        tf_param, _, _ = method(
            self.source, self.target, 'nonrigid', *args, **kwargs
        )
        self.parse(tf_param)
        print("registered 1 nonrigid transformation")

    def parse(self, tf_param):
        self.deform = copy.deepcopy(self.source)
        self.deform.points = tf_param.transform(self.deform.points)
        self.deform_points = obj3d.pcd2np(self.deform)
        self.disp = self.deform_points - self.source_points

    def fix(self):
        self.deform_fix_points = []

        for n in range(len(self.deform_points)):
            self.deform_fix_points.append(
                obj3d.search_nearest_point(self.deform_points[n], self.target_points)
            )

        self.disp_fix = self.deform_fix_points - self.source_points
        self.deform_fix = obj3d.np2pcd(self.deform_fix_points)
        print("fixed 1 nonrigid transformation")

    def points_disp(self, points):
        idxs = []
        for point in points:
            idx = obj3d.search_nearest_point_idx(point, self.source_points)
            idxs.append(idx)
        return self.deform_fix_points[idx]


class Obj4d(object):
    def __init__(self, enable_rigid=False, enable_nonrigid=False, start_time=0, fps=120):
        self.obj_ls = []
        self.enable_rigid = enable_rigid
        self.enable_nonrigid = enable_nonrigid

        self.start_time = start_time
        self.fps = fps

    def add_obj(self, *objs, **kwargs):
        """ add object(s) and parse its 4d movement between adjacent frames """
        for obj in objs:
            self.obj_ls.append(obj)

            if len(self.obj_ls) == 1:
                self.process_first_obj()
                continue

            if self.enable_rigid:
                self.process_rigid_dynamic(idx_source=-1, idx_target=-2, **kwargs)  # aligned to the previous one

            if self.enable_nonrigid:
                self.process_nonrigid_dynamic(idx_source=-2, idx_target=-1, **kwargs)  # aligned to the later one

    def process_first_obj(self):
        pass

    def process_rigid_dynamic(self, idx_source, idx_target, *args, **kwargs):
        trans = Trans_hl_Rigid(self.obj_ls[idx_source], self.obj_ls[idx_target], *args, **kwargs)
        self.obj_ls[idx_source].set_trans_rigid(trans)

    def process_nonrigid_dynamic(self, idx_source, idx_target, *args, **kwargs):
        trans = Trans_hl_Nonrigid(self.obj_ls[idx_source], self.obj_ls[idx_target], *args, **kwargs)
        self.obj_ls[idx_source].set_trans_nonrigid(trans)

    def offset_rotate(self):
        for obj in self.obj_ls[1:]:  # the first 3d object doesn't need reorientation
            obj.offset_rotate()
        print("4d object reorientated")

    def show(self):
        o3d.visualization.draw_geometries([

        ])

    def get_o3d(self):
        pass


class Obj4d_Kps(Obj4d):
    def __init__(self, mode='data', markerset=None, *args, **kwargs):
        Obj4d.__init__(self, *args, **kwargs)
        self.mode = mode  # 'data' mode or 'manual' mode
        self.markerset = markerset

    def process_first_obj(self):
        init_obj = self.obj_ls[0]
        if self.mode == 'manual':
            front_pcd = obj3d.pcd_crop_front(init_obj.pcd_hd, 0.6)
            init_obj.kps.select_kps_points(front_pcd)
        elif self.mode == 'data':
            init_obj.kps.load_from_markerset_time(self.markerset, self.start_time)
        else:
            print("mode string invalid")

    def process_nonrigid_dynamic(self, idx_source, idx_target, *args, **kwargs):
        # trans = Trans_hl_Nonrigid(self.obj_ls[idx_source], self.obj_ls[idx_target], *args, **kwargs)
        # self.obj_ls[idx_source].set_trans_nonrigid(trans)
        Obj4d.process_nonrigid_dynamic(self, idx_source, idx_target, *args, **kwargs)

        source_kps = self.obj_ls[idx_source].kps
        target_kps = self.obj_ls[idx_target].kps
        source_kps.setup_kps_deform()
        target_kps.set_kps_source_points(source_kps.get_kps_deform_points())

    def error_estimate(self):
        print("\nerror analysis:")
        for obj_id in range(1, len(self.obj_ls)):
            obj = self.obj_ls[obj_id]

            if self.mode == 'manual':
                front_pcd = obj3d.pcd_crop_front(obj.pcd_hd, 0.6)
                obj.kps_gt.select_kps_points(front_pcd)
            elif self.mode == 'data':
                obj.kps_gt.load_from_markerset_time(
                    self.markerset,
                    self.start_time + obj_id / self.fps
                )
            else:
                print("mode string invalid")

            diff = obj.diff_kps_pred_gt()
            print("distance between tracked key points and ground truth:\n{:.4f}".format(diff))

            max_bound = obj3d.pcd_get_max_bound(obj.pcd_hd)
            min_bound = obj3d.pcd_get_min_bound(obj.pcd_hd)
            width = max_bound[0] - min_bound[0]
            print("ratio to lateral distance：\n{:.4%}".format(diff / width))


def offset_rotate_Obj4d(o4d):
    o4d_offset = copy.deepcopy(o4d)
    o4d_offset.offset_rotate()
    return o4d_offset


if __name__ == '__main__':
    '''
    # nonrigid - key points tracking
    o3_ls = obj3d.load_obj_series('dataset/45kmh_26markers_12fps/', 0, 1, obj_type=Obj3d_Kps, sample_hd=1000)
    vicon = kps.MarkerSet()
    vicon.load_from_vicon('dataset/6kmh_softbra_8markers_1.csv')
    vicon.interp_field()

    o4 = Obj4d_Kps(
        enable_nonrigid=True,
        mode='data',
        markerset=vicon,
        fps=120
    )
    o4.add_obj(*o3_ls, lmd=1e3)
    o4.obj_ls[0].trans_nonrigid.show()
    o4.error_estimate()
    '''

    # rigid - reorientation
    o3_ls = obj3d.load_obj_series('dataset/45kmh_26markers_12fps/', 0, 1, obj_type=obj3d.Obj3d_Deform, sample_hd=1000)

    o4 = Obj4d(
        enable_rigid=True,
        fps=120
    )
    o4.add_obj(*o3_ls)
    o4_offset = offset_rotate_Obj4d(o4)
    # o4.offset_rotate()