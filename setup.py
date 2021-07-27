from setuptools import setup 
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules = cythonize("ChessGame.pyx", language_level = "3", annotate = True),
    include_dirs = [np.get_include()],
)