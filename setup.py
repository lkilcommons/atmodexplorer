# This code is covered under the GPL v3, which is included with this software
#
# This application is QT front end that allows a user to interactively play with output from
# various emperical atmosphere models
# 
# A project of the University of Colorado Space Environment Data Analysis Group (CU-SEDA)
# Liam Kilcommons
# December, 2014

import os
import glob

os.environ['DISTUTILS_DEBUG'] = "1"

from setuptools import setup, Extension
from setuptools.command import install as _install
from numpy.distutils.core import setup, Extension

#This is a stripped down version of the setup file for davitpy. I stripped out everything
#except MSIS so that it could be used independantly
# This code is covered under the DavitPy license (GPL v3), which is included with this software

#blah.extensionBlah is the way to specify that you want extensionBlah to be a member of package blah
msisFort = Extension("msispy.msisFort",sources=["msispy/nrlmsise00_sub.for",'msispy/nrlmsis.pyf'])

setup(name='atmodexplorer',
      version = "0.1.0",
      description = "QT front end that allows a user to interactively play with output from various emperical atmosphere models",
      author = "University of Colorado Space Environent Data Analysis Group (CU-SEDA)",
      author_email = 'liam.kilcommons@colorado.edu',
      url = "http://github.com/lkilcommons/atmodexplorer",
      download_url = "https://github.com/lkilcommons/atmodexplorer/",
      long_description = "None yet"
      install_requires=['msispy','numpy','matplotlib','BaseMap'],
      packages=['atmodexplorer'],
      package_data={}, #data names must be list
      license='LICENSE.txt',
      zip_safe = False,
      classifiers = [
            "Development Status :: 4 - Beta",
            "Topic :: Scientific/Engineering",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: GNU General Public License (GPL)",
            "Natural Language :: English",
            "Programming Language :: Python"
            ],
      )

# setup(
#     name='TowelStuff',
#     version='0.1.0',
#     author='J. Random Hacker',
#     author_email='jrh@example.com',
#     packages=['towelstuff', 'towelstuff.test'],
#     scripts=['bin/stowe-towels.py','bin/wash-towels.py'],
#     url='http://pypi.python.org/pypi/TowelStuff/',
#     license='LICENSE.txt',
#     description='Useful towel-related stuff.',
#     long_description=open('README.md').read(),
#     install_requires=[
#         "Django >= 1.1.1",
#         "caldav == 0.1.4",
#     ],
# )
