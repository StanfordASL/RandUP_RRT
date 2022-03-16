from setuptools import setup, find_packages
import sys

REQUIRES=[
    "numpy",
    "matplotlib",
    "rtree",
    "pybullet",
    "scipy"
]
if sys.version_info < (3, 7):
    REQUIRES.append("dataclasses==0.8")

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
