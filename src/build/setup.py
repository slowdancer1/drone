from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension


setup(name='quadsim',
      version='0.0.1',
      author="Yuang Zhang",
      author_email='1328410180@qq.com',
      description="QuadSim",
      long_description='',
      packages=['quadsim'],
      package_dir={'quadsim': './'},
      package_data={'quadsim': ['*.so']},
      zip_fase=True,
      url=None)
