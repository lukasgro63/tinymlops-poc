# tinylcm/setup.py
#!/usr/bin/env python
"""Setup script for TinyLCM.

TinyLCM (TinyML Lifecycle Management) is a framework for autonomous 
and adaptive on-device machine learning lifecycle management.
"""

import os
from setuptools import find_packages, setup

# Allow setup.py to be run from any path
here = os.path.abspath(os.path.dirname(__file__))

if __name__ == "__main__":
    # The actual setup function - we rely on pyproject.toml for most metadata
    setup(
        packages=find_packages(exclude=['tests*']),
        package_dir={'': '.'},
    )