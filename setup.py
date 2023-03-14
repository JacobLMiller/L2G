from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy 

"""
Run the following command for dynamic lib 
python setup.py build_ext --inplace
"""
setup(ext_modules= cythonize(Extension(
    'cython_l2g',
    sources=['modules/cython_l2g.pyx'],
    language='c',
    include_dirs=[numpy.get_include()],
    library_dirs=[],
    libraries=[],
    extra_compile_args=[],
    extra_link_args=[]
)))

import shutil

s = "cython_l2g.cpython-39-x86_64-linux-gnu.so"
shutil.move(s, f"modules/{s}")