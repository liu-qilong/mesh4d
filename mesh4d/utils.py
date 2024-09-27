from __future__ import annotations
from typing import Type, Union, Iterable

import os
import sys
import pickle
import imageio
import numpy as np
import pyvista as pv

import mesh4d.config.param
from mesh4d import kps

def images_to_gif(folder: Union[str, None] = None, remove: bool = False):
    files = os.listdir(folder)
    files.sort()
    images = []

    for file in files:
        if ('png' in file or 'jpg' in file) and ('gif-' in file):
            images.append(imageio.imread(folder + file))
            if remove:
                os.remove(folder + file)

    if len(images) == 0:
        print("No images in folder")
    else:
        imageio.mimsave(folder + 'output.gif', images)


def progress_bar(percent: float, bar_len: int = 20, front_str: str = '', back_str: str = ''):
    sys.stdout.write("\r")
    sys.stdout.write("{}[{:<{}}] {:.1%}{}".format(front_str, "=" * int(bar_len * percent), bar_len, percent, back_str))
    sys.stdout.flush()
    # avoiding '%' appears when progress completed
    if percent == 1:
        print()


def save_pkl_object(obj, export_folder: str = 'output/', export_name: str = 'pickle'):
    filepath = os.path.join(export_folder, "{}.pkl".format(export_name))
    
    with open(filepath, 'wb') as outp:  # overwrites any existing file
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_pkl_object(filepath: str):
    with open(filepath, 'rb') as inp:
        return pickle.load(inp)