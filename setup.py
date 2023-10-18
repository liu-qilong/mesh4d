from setuptools import setup

setup(
	name='mesh4d',
	version='0.0',
	description='Toolkit for 4D (3D + T) data visualisation, operation, and dynamic dense-registration. Extents the existing concepts of 3D mesh toolkit to a 4D mesh toolkit.',
    url='',
    license='',
	author='Qilong Liu',
	author_email='qilong-kirov.liu@outlook.com',
	packages=['mesh4d'],
	install_requires=[
		'open3d',
        'pyvista[all]',
        'probreg',
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
	],
)