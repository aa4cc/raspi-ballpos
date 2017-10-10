from distutils.core import setup, Extension

# the c++ extension module
extension_mod = Extension("hooppos", ["hooppos_wrapper.c"])

setup(name = "hooppos", ext_modules=[extension_mod])
