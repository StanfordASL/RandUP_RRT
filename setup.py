from setuptools import setup, find_packages
import sys

setup(
    name="randUP_RRT",
    version="0.0.1",
    install_requires=[
        "numpy",
        "matplotlib",
        "rtree",
        "pybullet",
        "scipy"
    ],
    include_package_data=True,
    packages=find_packages(),
)
