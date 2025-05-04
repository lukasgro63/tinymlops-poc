# tinylcm/setup.py
#!/usr/bin/env python
"""Setup script for TinyLCM."""

import os

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

if __name__ == "__main__":
    setup(
        packages=find_packages(where=here, exclude=['tests*']),
        package_dir={'': '.'},
        py_modules=['tinylcm'],
    )