from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy 

"""
Run the following command for dynamic lib 
python setup.py build_ext --inplace
"""
setup(ext_modules= cythonize(Extension(
    'cython_mds',
    sources=['cython_mds.pyx'],
    language='c',
    include_dirs=[numpy.get_include()],
    library_dirs=[],
    libraries=[],
    extra_compile_args=[],
    extra_link_args=[]
)))