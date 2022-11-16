import copy
import open3d as o3d

import obj3d
import kps
import field

class Obj4d(object):
    """This is a conceptual class representation of a simple BLE device
    (GATT Server). It is essentially an extended combination of the
    :class:`bluepy.btle.Peripheral` and :class:`bluepy.btle.ScanEntry` classes

    Args:
        path (str): The path of the file to wrap
        field_storage (FileStorage): The :class:`FileStorage` instance to wrap
        temporary (bool): Whether or not to delete the file when the File instance is destructed

    Returns:
        BufferedFileStorage: A buffered writable file descriptor
    """
    def __init__(self, start_time=0, fps=120, **kwargs):
        self.obj_ls = []
        self.start_time = start_time
        self.fps = fps

    def add_obj(self, *objs, **kwargs):
        """ add object(s) and parse its 4d movement between adjacent frames """
        for obj in objs:
            self.obj_ls.append(obj)


class Obj4d_Kps(Obj4d):
    def __init__(self, markerset=None, **kwargs):
        Obj4d.__init__(self, **kwargs)
        self.markerset = markerset

    def add_obj(self, *objs, **kwargs):
        """ add object(s) and parse its 4d movement between adjacent frames """
        Obj4d.add_obj(self, *objs, **kwargs)

        for idx in range(len(self.obj_ls)):
            obj = self.obj_ls[idx]
            obj.kps.load_from_markerset_time(self.markerset, self.start_time + idx / self.fps)


class Obj4d_Deform(Obj4d_Kps):
    def __init__(self, enable_rigid=False, enable_nonrigid=False,  **kwargs):
        Obj4d_Kps.__init__(self, **kwargs)
        self.enable_rigid = enable_rigid
        self.enable_nonrigid = enable_nonrigid

    def add_obj(self, *objs, **kwargs):
        """ add object(s) and parse its 4d movement between adjacent frames """
        Obj4d_Kps.add_obj(self, *objs, **kwargs)

        for obj in self.obj_ls:
            if len(self.obj_ls) == 1:
                self.__process_first_obj()
                continue

            if self.enable_rigid:
                self.__process_rigid_dynamic(idx_source=-1, idx_target=-2, **kwargs)  # aligned to the previous one

            if self.enable_nonrigid:
                self.__process_nonrigid_dynamic(idx_source=-2, idx_target=-1, **kwargs)  # aligned to the later one

    def __process_first_obj(self):
        pass

    def __process_rigid_dynamic(self, idx_source, idx_target, **kwargs):
        trans = field.Trans_Rigid(
            source_obj=self.obj_ls[idx_source],
            target_obj=self.obj_ls[idx_target],
        )
        trans.regist(**kwargs)
        self.obj_ls[idx_source].set_trans_rigid(trans)

    def __process_nonrigid_dynamic(self, idx_source, idx_target, **kwargs):
        trans = field.Trans_Nonrigid(
            source_obj=self.obj_ls[idx_source],
            target_obj=self.obj_ls[idx_target],
        )
        trans.regist(**kwargs)
        self.obj_ls[idx_source].set_trans_nonrigid(trans)

    def offset_rotate(self):
        for obj in self.obj_ls[1:]:  # the first 3d object doesn't need reorientation
            obj.offset_rotate()
        print("4d object reorientated")


def offset_rotate_Obj4d(o4d):
    o4d_offset = copy.deepcopy(o4d)
    o4d_offset.offset_rotate()
    return o4d_offset


if __name__ == '__main__':
    # non-rigid - key points tracking
    o3_ls = obj3d.load_obj_series(
        folder='data/6kmh_softbra_8markers_1/',
        start=0,
        end=1,
        obj_type=obj3d.Obj3d_Deform, sample_num=1000
    )

    vicon = kps.MarkerSet()
    vicon.load_from_vicon('data/6kmh_softbra_8markers_1.csv')
    vicon.interp_field()

    o4 = Obj4d_Deform(
        enable_nonrigid=True,
        markerset=vicon,
        fps=120,
    )
    o4.add_obj(*o3_ls, lmd=1e3)
    o4.obj_ls[0].trans_nonrigid.show()
    '''
    # rigid - reorientation
    o3_ls = obj3d.load_obj_series('data/6kmh_softbra_8markers_1/', 0, 1, obj_type=obj3d.Obj3d_Deform, sample_hd=1000)

    o4 = Obj4d(
        enable_rigid=True,
        fps=120
    )
    o4.add_obj(*o3_ls)
    o4_offset = offset_rotate_Obj4d(o4)
    # o4.offset_rotate()
    '''