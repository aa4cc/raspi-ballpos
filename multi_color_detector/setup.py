from distutils.core import setup, Extension
import numpy
# the c++ extension module
extension_mod = Extension("multi_color_detector", ["multi_color_detector.c"],include_dirs=[numpy.get_include()])

setup(name = "multi_color_detector", ext_modules=[extension_mod])
