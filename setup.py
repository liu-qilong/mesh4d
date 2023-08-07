from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
	name='mesh4d',
	version='0.0',
	description='Toolkit for 4D (3D + T) data visualisation, operation, and dynamic dense-registration. Extents the existing concepts of 3D mesh toolkit to a 4D mesh toolkit.',
    long_description=long_description,
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