# Ultra Motion Capture

This package is developed for the data processing of the 3dMD 4D scanning system. Comparing with traditional motion capture system, such as Vicon:

- Vicon motion capture system can provided robust & accurate key points tracking based on physical marker points attached to human body. But it suffer from the lack of continuous surface deformation information.

- 3dMD 4D scanning system can record continuous surface deformation information. But it doesn't provide key point tracking feature and it's challenging to track the key points via Computer Vision approach, even with the state of the art methods in academia [^Min_Z_2021].

[^Min_Z_2021]: Min, Z., Liu, J., Liu, L., & Meng, M. Q.-H. (2021). Generalized coherent point drift with multi-variate gaussian distribution and watson distribution. IEEE Robotics and Automation Letters, 6(4), 6749–6756. https://doi.org/10.1109/lra.2021.3093011

To facilitate human factor research, we deem it an important task to construct a hybrid system that can integrate the advantages and potentials of both systems. The motivation and the core value of this project can be described as: *adding continuous spatial dynamic information to Vicon* or *adding discrete key points information to 3dMD*, leading to advancing platform for human factor research in the domain of dynamic human activity.

## Development Note

### Documentation

The documentation web pages can be found in `docs/build/html/`. Please open `index.html` to enter the documentation which provides comprehensive descriptions and working examples for each classes and functions we provided.

The documentation is generated with [Sphinx](https://www.sphinx-doc.org/en/master/index.html). If you aren't familiar with it, I would recommend two tutorials for quick-start:

- [A “How to” Guide for Sphinx + ReadTheDocs - sglvladi](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/) provides an easy-to-follow learning curve but omitted some details.
- [Getting Started - Sphinx](https://www.sphinx-doc.org/en/master/usage/quickstart.html) is harder to understand, but provides some essential information to understand the background, which is vital even for very basic configuration.

### Project Code

The project code is stored in `UltraMotionCapture/` folder. Under it is a `data/` folder, a default output folder `/output`, a `config/` folder storing configuration variables.

Except for these folders, you must have noticed there are also some `.py` files, including `utils.py`, `field.py`, `kps.py`, `obj3d.py`, and `obj4d.py`. These are **core modules** for this package. They provides a skeleton for building any downstream analysis task and shall not been modified, unless there are very strong reasons to do so.

Other than these, there is a `analysis` folder haven't been discussed. It's the `UltraMotionCapture.analysis` sub-package storing all downstream analysis modules. At current stage, it's not completed and is still under active development.

### Version Control

We use `git` as the version control tool, which is very popular among developers and works seamlessly with GitHub. If you aren't familiar with it, I would recommend this tutorial for quick-start: [Git 教程 - 廖雪峰](https://www.liaoxuefeng.com/wiki/896043488029600)

Following is a series of notes summarise major commands:

- [001-新建仓库与配置](https://dynalist.io/d/98jG0ek7Inu6QtMoBTjP4vj6)
- [002-本地仓库操作](https://dynalist.io/d/4L3UM0yhrYAriHjoQTptEMBk)
- [003-远程仓库操作](https://dynalist.io/d/0NozPTssxkVC8aVebCbNmBkR)
