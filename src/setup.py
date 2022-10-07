from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "quadsim",
        sorted(glob("src/*.cpp")),  # Sort source files for reproducibility
    ),
]

setup(name='quadsim',
      version='0.0.1',
      author="Yuang Zhang",
      author_email='1328410180@qq.com',
      description="QuadSim",
      long_description='',
      ext_modules=ext_modules)
