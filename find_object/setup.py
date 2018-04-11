from distutils.core import setup, Extension

# the c++ extension module
extension_mod = Extension("find_object", ["find_object.c"])

setup(name = "find_object", ext_modules=[extension_mod])
