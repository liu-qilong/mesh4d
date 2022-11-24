# Ultra Motion Capture

_Fig. Overall structure of the core modules._

![overall structure](https://github.com/TOB-KNPOB/Ultra-Motion-Capture/blob/main/docs/source/figures/overall_structure.png)

_The solid arrow pointing from `class A` to `class B` indicates that `class B` is derived from `class A`, while the dotted arrow indicates that a `class A` object contains a `class B` object as an attribute._

## Introduction

This package is developed for the data processing of the 3dMD 4D scanning system. Compared with traditional motion capture systems, such as Vicon:

- Vicon motion capture system can provide robust & accurate key points tracking based on physical marker points attached to the human body. But it suffers from the lack of continuous surface deformation information.

- 3dMD 4D scanning system can record continuous surface deformation information. But it doesn't provide key point tracking functionality and it's challenging to track the key points via the Computer Vision approach, even with the state-of-the-art methods in academia[^Min_Z_2021].

[^Min_Z_2021]: Min, Z., Liu, J., Liu, L., & Meng, M. Q.-H. (2021). Generalized coherent point drift with multi-variate gaussian distribution and Watson distribution. IEEE Robotics and Automation Letters, 6(4), 6749–6756. https://doi.org/10.1109/lra.2021.3093011

To facilitate human factor research, we deem it an important task to construct a hybrid system that can integrate the advantages and potentials of both systems. The motivation and the core value of this project can be described as *adding continuous spatial dynamic information to Vicon* or *adding discrete key points information to 3dMD*, leading to an advancing platform for human factor research in the domain of dynamic human activity.

## Development Notes

### Documentation

The documentation web pages can be found in `docs/build/html/`. Please open `index.html` to enter the documentation which provides comprehensive descriptions and working examples for each class and function we provided.

The documentation is generated with [Sphinx](https://www.sphinx-doc.org/en/master/index.html). If you are not familiar with it, I would recommend two tutorials for quick-start:

- [A “How to” Guide for Sphinx + ReadTheDocs - sglvladi](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/) provides an easy-to-follow learning curve but omitted some details.
- [Getting Started - Sphinx](https://www.sphinx-doc.org/en/master/usage/quickstart.html) is harder to understand, but provides some essential information to understand the background, which is vital even for very basic configuration.

### Project Code

The project code is stored in `UltraMotionCapture/` folder. Under it is a `data/` folder, a default output folder `output/`, a `config/` folder storing configuration variables.

Except for these folders, you must have noticed there are also some `.py` files, including `utils.py`, `field.py`, `kps.py`, `obj3d.py`, and `obj4d.py`. These are **core modules** for this package. They provide a skeleton for building any downstream analysis task and shall not be modified unless there are very strong reasons to do so.

Other than these, there is an `analysis/` folder haven't been discussed. It's the `UltraMotionCapture.analysis/` sub-package storing all downstream analysis modules. At current stage, it's not completed and is still under active development.

### Version Control

We use `git` as the version control tool, which is very popular among developers and works seamlessly with GitHub. If you are not familiar with it, I would recommend this tutorial for quick-start: [Git 教程 - 廖雪峰](https://www.liaoxuefeng.com/wiki/896043488029600)

Following is a series of notes that summarise major commands:

- [001-新建仓库与配置](https://dynalist.io/d/98jG0ek7Inu6QtMoBTjP4vj6)
- [002-本地仓库操作](https://dynalist.io/d/4L3UM0yhrYAriHjoQTptEMBk)
- [003-远程仓库操作](https://dynalist.io/d/0NozPTssxkVC8aVebCbNmBkR)

### Dependencies

This project is built upon various open-source packages. All dependencies are listed in `requirements.txt`, please install them properly under a Python 3 environment.
