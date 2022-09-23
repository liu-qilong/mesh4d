import open3d as o3d

import obj3d


class Kps(object):
    def __init__(self):
        self.kps_source_points = None
        self.kps_deform_points = None
        self.trans = None

    def select_kps_points(self, source, save=False):
        print("please select key points")

        def pick_points(pcd):
            vis = o3d.visualization.VisualizerWithEditing()
            vis.create_window()
            vis.add_geometry(pcd)
            vis.run()
            vis.destroy_window()
            return vis.get_picked_points()

        pick = pick_points(source)
        points = obj3d.pcd2np(source)
        self.kps_source_points = points[pick, :]
        print("selected key points:\n{}".format(self.kps_source_points))

        if save:
            pass

    def set_kps_source_points(self, points):
        self.kps_source_points = points

    def set_trans(self, trans):
        self.trans = trans

    def get_kps_source_points(self):
        if self.kps_source_points is None:
            print("source key points haven't been set")
        else:
            return self.kps_source_points

    def setup_kps_deform(self):
        if self.kps_source_points is None:
            print("source key points haven't been set")
        elif self.trans is None:
            print("transformation of the key points haven't been set")
        else:
            self.kps_deform_points = self.trans.points_disp(self.kps_source_points)

    def get_kps_deform_points(self):
        if self.kps_source_points is None:
            print("source key points haven't been set")
        elif self.trans is None:
            print("transformation of the key points haven't been set")
        else:
            return self.kps_deform_points
