"""Setup script for backward compatibility with older pip versions."""
from setuptools import setup, find_packages

setup(
    name="tinymlops-poc",
    version="0.1.0",
    packages=["tinylcm", "tinysphere"],
    package_dir={"": "."},
)