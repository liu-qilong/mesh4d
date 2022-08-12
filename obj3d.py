import copy
import numpy as np
import open3d as o3d
from probreg import cpd, bcpd


class obj3d(object):
    def __init__(
            self,
            filedir,
            scale_rate=0.01,
            scale_center=(0, 0, 0),
            sample_ld=1000,
            sample_hd=10000):
        self.mesh = o3d.io.read_triangle_mesh(filedir).scale(scale_rate, center=scale_center)
        self.mesh.compute_vertex_normals()
        self.sampling(sample_ld, sample_hd)

    def sampling(self, sample_ld, sample_hd):
        self.pcd_ld = self.mesh.sample_points_poisson_disk(number_of_points=sample_ld, init_factor=5)
        self.pcd_hd = self.mesh.sample_points_poisson_disk(number_of_points=sample_hd, init_factor=5)

    def show(self):
        o3d.visualization.draw_geometries([
            self.mesh,
            copy.deepcopy(self.pcd_ld).translate((10, 0, 0)),
            copy.deepcopy(self.pcd_hd).translate((20, 0, 0)),
        ])

    def get_o3d(self):
        return copy.deepcopy(self.mesh)


def mesh2pcd(mesh, sample_d):
    return mesh.sample_points_poisson_disk(number_of_points=sample_d, init_factor=5)


def mesh_crop(mesh, min_bound=[-1000, -1000, -1000], max_bound=[1000, 1000, 1000]):
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    return mesh.crop(bbox)


def pcd2np(pcd):
    return np.asarray(pcd.points)


def np2pcd(points):
    """ Transform obj numpy cloud point to Open3D cloud point. """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def pcd_crop(pcd, min_bound=[-1000, -1000, -1000], max_bound=[1000, 1000, 1000]):
    points = pcd2np(pcd)
    points_crop = []

    for point in points:
        min_to_point = point - min_bound
        point_to_max = max_bound - point
        less_than_zero = np.sum(min_to_point < 0) + np.sum(point_to_max < 0)
        if less_than_zero == 0:
            points_crop.append(point)

    return np2pcd(np.array(points_crop))


def pcd_get_center(pcd):
    points = pcd2np(pcd)
    return np.mean(points, 0)


def pcd_get_max_bound(pcd):
    points = pcd2np(pcd)
    return np.ndarray.max(points, 0)


def pcd_get_min_bound(pcd):
    points = pcd2np(pcd)
    return np.ndarray.min(points, 0)


if __name__ == '__main__':
    o3 = obj3d('dataset/45kmh_26markers_12fps/speed_45km_h_26_marker_set_1.000001.obj')

    o3_center = pcd_get_center(o3.pcd_ld)
    print("center: {}".format(o3_center))
    print("max bound: {}".format(pcd_get_max_bound(o3.pcd_ld)))
    print("min bound: {}".format(pcd_get_min_bound(o3.pcd_ld)))

    o3.show()
    o3d.visualization.draw_geometries([pcd_crop(o3.pcd_hd, min_bound=o3_center)])
    o3d.visualization.draw_geometries([mesh_crop(o3.mesh, min_bound=o3_center)])