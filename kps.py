import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
from scipy import interpolate

import obj3d
import utils


class Kps(object):
    def __init__(self):
        self.kps_source_points = None  # key points are stored in Nx3 numpy array

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
        self.set_kps_source_points(
            points[pick, :]
        )
        print("selected key points:\n{}".format(self.kps_source_points))

        if save:
            pass

    def load_from_markerset_frame(self, markerset, calibration, frame_id=0):
        points = markerset.get_frame_coord(frame_id)
        points_cal = points
        self.set_kps_source_points(points_cal)

    def load_from_markerset_time(self, markerset, calibration, time=0):
        points = markerset.get_time_coord(time)
        points_cal = points
        self.set_kps_source_points(points_cal)

    def set_kps_source_points(self, points):
        self.kps_source_points = points

    def get_kps_source_points(self):
        if self.kps_source_points is None:
            print("source key points haven't been set")
        else:
            return self.kps_source_points


class Kps_Deform(Kps):
    def __init__(self):
        Kps.__init__(self)
        self.kps_deform_points = None
        self.trans = None

    def set_trans(self, trans):
        self.trans = trans

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


class Marker(object):
    def __init__(self, name, start_time=0, fps=100):
        self.name = name
        self.start_time = start_time
        self.fps = fps

        # x, y,z data are stored in 3xN numpy array
        self.coord = None  # coordinates
        self.speed = None  # speed
        self.accel = None  # acceleration

        self.frame_num = None

        self.x_field = None
        self.y_field = None
        self.z_field = None

    def fill_data(self, data_input):
        if self.coord is None:
            self.coord = data_input
            self.frame_num = data_input.shape[1]
        elif self.speed is None:
            self.speed = data_input
        elif self.accel is None:
            self.accel = data_input

    def interp_field(self):
        if self.coord is None:
            print("coordinates information not found")
            return

        frame_range = range(len(self.coord[0]))

        self.x_field = interpolate.interp1d(frame_range, self.coord[0], 'slinear')
        self.y_field = interpolate.interp1d(frame_range, self.coord[1], 'slinear')
        self.z_field = interpolate.interp1d(frame_range, self.coord[2], 'slinear')

    def get_frame_coord(self, frame_id):
        return self.coord[:, frame_id]

    def get_time_coord(self, time):
        if self.x_field is None:
            print("coordinates field need to be interped first")
            return

        frame_id = (time - self.start_time) * self.fps
        coord_interp = np.array(
            [self.x_field(frame_id),
             self.y_field(frame_id),
             self.z_field(frame_id)]
        )
        return coord_interp

    def plot_track(
            self,
            line_start_frame=0,
            line_end_frame=None,
            dot_start_frame=0,
            dot_end_frame=None,
            line_alpha=0.5,
            line_width=1,
            dot_s=10,
            dot_alpha=0.5,
            dpi=300,
            is_show=True,
            is_save=False,
            *args,
            **kwargs
    ):
        fig = plt.figure(dpi=dpi)
        ax = fig.add_subplot(projection='3d')

        self.plot_add_line(
            ax,
            line_start_frame,
            line_end_frame,
            line_alpha,
            line_width)

        self.plot_add_dot(
            ax,
            dot_start_frame,
            dot_end_frame,
            dot_s,
            dot_alpha,
        )

        font = {'family': 'Times New Roman'}
        plt.title('Marker Point Trajectory\n' + self.name, fontdict=font)
        ax.set_xlabel('X-axis', fontdict=font)
        ax.set_ylabel('Y-axis', fontdict=font)
        ax.set_zlabel('Z-axis', fontdict=font)
        ax.tick_params(labelsize=7)

        if is_show:
            plt.show()

        if is_save:
            plt.savefig('figures/save-line'
                        + str(line_start_frame) + ' ' + str(line_end_frame)
                        + '-dot' + str(dot_start_frame) + ' ' + str(dot_end_frame))

    def plot_add_line(
            self,
            ax,
            line_start_frame=0,
            line_end_frame=None,
            line_alpha=0.5,
            line_width=1,
            *args,
            **kwargs
    ):
        ax.plot3D(
            self.coord[0, line_start_frame:line_end_frame],
            self.coord[1, line_start_frame:line_end_frame],
            self.coord[2, line_start_frame:line_end_frame],
            'gray', alpha=line_alpha, linewidth=line_width, *args, **kwargs
        )

    def plot_add_dot(
            self,
            ax,
            dot_start_frame=0,
            dot_end_frame=None,
            dot_s=10,
            dot_alpha=0.5,
            *args,
            **kwargs
    ):
        ax.scatter3D(
            self.coord[0, dot_start_frame:dot_end_frame],
            self.coord[1, dot_start_frame:dot_end_frame],
            self.coord[2, dot_start_frame:dot_end_frame],
            s=dot_s, alpha=dot_alpha, *args, **kwargs
        )


class MarkerSet(object):
    def __init__(self):
        pass

    def load_from_vicon(self, filedir):
        def parse(df, df_head):
            self.fps = df_head.values.tolist()[0][0]  # parse the fps
            self.points = {}
            col_names = df.columns.values.tolist()

            for col_id in range(len(col_names)):
                col_name = col_names[col_id]
                point_name = col_name.split('.')[0]

                # skip columns that contain NaN
                # (checking start from row 4, because for speed and acceleration the first few rows are empty)
                # or that follows the 'X' columns
                if df.loc[4:, col_name].isnull().values.any():
                    continue

                if 'Unnamed' in col_name:
                    continue

                # the first occurrence of a point
                if point_name not in self.points.keys():
                    self.points[point_name] = Marker(point_name, fps=self.fps)

                # fill the following 3 columns' X, Y, Z values into the point's object
                data_input = df.loc[2:, col_name:col_names[col_id+2]].to_numpy(dtype=float).transpose()
                self.points[point_name].fill_data(data_input)

        df = pd.read_csv(filedir, skiprows=2)  # skip the first two rows
        df_head = pd.read_csv(filedir, nrows=1)  # only read the first two rows
        parse(df, df_head)

    def load_from_obj4d(self):
        pass

    def interp_field(self):
        for point in self.points.values():
            point.interp_field()

    def get_frame_coord(self, frame_id):
        points = []
        for point in self.points.values():
            points.append(
                point.get_frame_coord(frame_id)
            )
        return np.array(points)

    def get_time_coord(self, time):
        points = []
        for point in self.points.values():
            points.append(
                point.get_time_coord(time)
            )
        return np.array(points)

    def plot_track(
            self,
            start_frame=0,
            end_frame=None,
            step=1,
            remove=True,
            *args,
            **kwargs
    ):
        if end_frame is None:
            first_point = list(self.points.values())[0]
            end_frame = first_point.frame_num

        for frame_id in range(start_frame, end_frame, step):
            self.plot_frame(frame_id, is_show=False, is_save=True, *args, **kwargs)

        utils.images_to_gif('figures/', remove=remove)

    def plot_frame(
            self,
            frame_id,
            dpi=300,
            is_add_line=True,
            is_show=True,
            is_save=False,
            *args, **kwargs
    ):
        fig = plt.figure(dpi=dpi)
        ax = fig.add_subplot(projection='3d')

        for point in self.points.values():
            point.plot_add_dot(ax, frame_id, frame_id+1)
            if is_add_line:
                point.plot_add_line(ax)

        font = {'family': 'Times New Roman'}
        plt.title('Marker Point Trajectory', fontdict=font)
        ax.set_xlabel('X-axis', fontdict=font)
        ax.set_ylabel('Y-axis', fontdict=font)
        ax.set_zlabel('Z-axis', fontdict=font)
        ax.tick_params(labelsize=7)

        plt.savefig('figures/gif-{:0>4d}'.format(frame_id))
        if is_show:
            plt.show()

        if is_save:
            filedir = 'figures/gif-' + str(frame_id)
            plt.savefig(filedir)
            print('saved ' + filedir)


if __name__ == '__main__':

    vicon = MarkerSet()
    vicon.load_from_vicon('dataset/6kmh_softbra_8markers_1.csv')
    vicon.interp_field()

    # data loading verification
    print(vicon.points.keys())
    print(vicon.points['Bra_Miss Sun:CLAV'].coord)
    '''
    print(vicon.points['Bra_Miss Sun:CLAV'].speed)
    print(vicon.points['Bra_Miss Sun:CLAV'].accel)

    # trajectory plot verification
    vicon.points['Bra_Miss Sun:CLAV'].plot_track(
        dot_alpha=0.9, line_alpha=0.5, dot_start_frame=0, dot_end_frame=1, is_show=False, is_save=True
    )
    vicon.plot_frame(frame_id=10, is_add_line=True)
    vicon.plot_track(step=3, end_frame=100)

    # coordinates field interpolation verification
    print(vicon.points['Bra_Miss Sun:CLAV'].get_time_coord(1.1001))
    print(vicon.get_time_coord(1.1001))

    # kps test
    kps = Kps()
    kps.load_from_markerset_time(vicon, 1.1001)
    print(kps.get_kps_source_points())
    '''