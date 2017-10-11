from distutils.core import setup, Extension

# the c++ extension module
extension_mod = Extension("sharemem", ["sharemem.c"])

setup(name = "sharemem", ext_modules=[extension_mod])
