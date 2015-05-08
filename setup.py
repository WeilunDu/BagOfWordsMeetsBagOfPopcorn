"""
this code is used for compiling doc2vec_inner.pyx into c code
"""
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("doc2vec_inner.pyx")
)