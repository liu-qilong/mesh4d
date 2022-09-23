import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate


class Vicon_Point(object):
    def __init__(self, name, start_time=0, fps=100):
        self.name = name
        self.start_time = start_time
        self.fps = fps

        self.coord = None  # coordinates
        self.speed = None  # speed
        self.accel = None  # acceleration

        self.x_field = None
        self.y_field = None
        self.z_field = None

    def fill_data(self, data_input):
        if self.coord is None:
            self.coord = data_input
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
            start_frame=0,
            end_frame=None,
            s=10,
            alpha=0.5,
            *args,
            **kwargs
    ):
        fig = plt.figure(dpi=500)
        ax = fig.add_subplot(projection='3d')

        ax.plot3D(
            self.coord[0, start_frame:end_frame],
            self.coord[1, start_frame:end_frame],
            self.coord[2, start_frame:end_frame],
            'gray'
        )

        ax.scatter3D(
            self.coord[0, start_frame:end_frame],
            self.coord[1, start_frame:end_frame],
            self.coord[2, start_frame:end_frame],
            s=s, alpha=alpha, *args, **kwargs
        )

        font = {'family': 'Times New Roman'}
        plt.title('Marker Point Trajectory\n' + self.name, fontdict=font)
        ax.set_xlabel('X-axis', fontdict=font)
        ax.set_ylabel('Y-axis', fontdict=font)
        ax.set_zlabel('Z-axis', fontdict=font)
        ax.tick_params(labelsize=7)

        plt.show()


class Vicon(object):
    def __init__(self, filedir):
        self.df = pd.read_csv(filedir, skiprows=2)  # skip the first two rows
        self.df_head = pd.read_csv(filedir, nrows=1)  # only read the first two rows
        self.parse()

    def parse(self):
        self.fps = self.df_head.values.tolist()[0][0]  # parse the fps
        self.points = {}
        col_names = self.df.columns.values.tolist()

        for col_id in range(len(col_names)):
            col_name = col_names[col_id]
            point_name = col_name.split('.')[0]

            # skip columns that contain NaN
            # (checking start from row 4, because for speed and acceleration the first few rows are empty)
            # or that follows the 'X' columns
            if self.df.loc[4:, col_name].isnull().values.any():
                continue

            if 'Unnamed' in col_name:
                continue

            # the first occurrence of a point
            if point_name not in self.points.keys():
                self.points[point_name] = Vicon_Point(point_name, fps=self.fps)

            # fill the following 3 columns' X, Y, Z values into the point's object
            data_input = self.df.loc[2:, col_name:col_names[col_id+2]].to_numpy(dtype=float).transpose()
            self.points[point_name].fill_data(data_input)

    def interp_field(self):
        for point in self.points.values():
            point.interp_field()


if __name__ == '__main__':
    vi = Vicon('dataset/6kmh_softbra_8markers_1.csv')

    vi.interp_field()

    # data loading verification
    print(vi.points.keys())
    print(vi.points['Bra_Miss Sun:CLAV'].coord)
    print(vi.points['Bra_Miss Sun:CLAV'].speed)
    print(vi.points['Bra_Miss Sun:CLAV'].accel)

    # trajectory plot verification
    print(vi.points['Bra_Miss Sun:CLAV'].get_time_coord(1.1001))

    # coordinates field interpolation verification
    vi.points['Bra_Miss Sun:CLAV'].plot_track()
