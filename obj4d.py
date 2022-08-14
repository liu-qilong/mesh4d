import copy
import open3d as o3d
from probreg import cpd

import obj3d


class Trans_hl(object):
    def __init__(self, source_obj, target_obj, **param):
        self.source = source_obj.pcd_hd
        self.target = target_obj.pcd_ld

        self.source_points = obj3d.pcd2np(self.source)
        self.target_points = obj3d.pcd2np(self.target)

        tf_param = self.regist(**param)
        self.parse(tf_param)
        self.fix()

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
    def regist(self, **param):
        tf_param, _, _ = cpd.registration_cpd(
            self.source, self.target, 'rigid', **param
        )
        print("registered 1 rigid transformation")
        return tf_param

    def parse(self, tf_param):
        pass

    def fix(self):
        pass


class Trans_hl_Nonrigid(Trans_hl):
    def regist(self, **param):
        tf_param, _, _ = cpd.registration_cpd(
            self.source, self.target, 'nonrigid', **param
        )
        print("registered 1 nonrigid transformation")
        return tf_param

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

    def deform(self, points):
        pass


class Obj4d(object):
    def __init__(self, enable_rigid=True, enable_nonrigid=True):
        self.obj_ls = []

        self.enable_rigid = enable_rigid
        self.enable_nonrigid = enable_nonrigid

        if self.enable_rigid:
            self.rigid_trans_ls = []

        if self.enable_nonrigid:
            self.nonrigid_trans_ls = []

    def add_obj(self, *objs, **param):
        """ add object(s) and parse its 4d movement between adjacent frames """
        for obj in objs:
            self.obj_ls.append(obj)

            if len(self.obj_ls) == 1:
                self.first_obj()
                continue

            if self.enable_rigid:
                self.process_rigid_dynamic(-2, -1, **param)

            if self.enable_nonrigid:
                self.process_nonrigid_dynamic(-2, -1, **param)

    def process_first_obj(self):
        pass

    def process_rigid_dynamic(self, idx_source, idx_target, **param):
        trans = Trans_hl_Rigid(self.obj_ls[idx_source], self.obj_ls[idx_target], **param)
        self.obj_ls[idx_source].set_trans_rigid(trans)

    def process_nonrigid_dynamic(self, idx_source, idx_target, **param):
        trans = Trans_hl_Nonrigid(self.obj_ls[idx_source], self.obj_ls[idx_target], **param)
        self.obj_ls[idx_source].set_trans_nonrigid(trans)

    def show(self):
        o3d.visualization.draw_geometries([

        ])

    def get_o3d(self):
        pass


class Obj4d_Kps(Obj4d):
    def process_rigid_dynamic(self, idx_source, idx_target, **param):
        trans = Trans_hl_Rigid(self.obj_ls[idx_source], self.obj_ls[idx_target], **param)
        self.obj_ls[idx_source].set_trans_rigid(trans)

    def process_nonrigid_dynamic(self, idx_source, idx_target, **param):
        trans = Trans_hl_Nonrigid(self.obj_ls[idx_source], self.obj_ls[idx_target], **param)
        self.obj_ls[idx_source].set_trans_nonrigid(trans)


if __name__ == '__main__':
    o3_ls = obj3d.load_obj_series('dataset/45kmh_26markers_12fps/', 0, 1)
    o4 = Obj4d(enable_rigid=False)
    o4.add_obj(*o3_ls, beta=1e3)
    o4.nonrigid_trans_ls[0].show()
