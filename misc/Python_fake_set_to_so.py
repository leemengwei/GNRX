# just :python -m compileall ./   #to generate pyc
# python Python_fake_set_to_so.py build_ext  --inplace    # to generate so


from distutils.core import setup
from Cython.Build import cythonize
from glob import glob
from IPython import embed
import sys
import os


pys = glob("*.py")
pys = list(set(pys) - set([sys.argv[0]]))
#pys = ['dependency_train.py']

#embed()
for py in pys:
    setup(ext_modules=cythonize([py]))
    pass

print("Done")
