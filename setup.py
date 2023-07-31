#!/usr/bin/env python

from setuptools import find_packages, setup

VERSION = "0.1dev"

with open("README.md", encoding="UTF-8") as f:
    readme = f.read()

with open("requirements.txt", encoding="UTF-8") as f:
    required = f.read().splitlines()

setup(
    name="velvet-dynamics",
    version=VERSION,
    description="Deep generative modelling of developmental trajectories from temporal transcriptomics",
    long_description=readme,
    long_description_content_type="text/markdown",
    keywords=['single-cell','transcriptomics','velocity','dynamics','generative'],
    author="Rory J. Maizels",
    author_email="rory.maizels@crick.ac.uk",
    url="https://github.com/rorymaizels/velvet",
    license="MIT",
    python_requires=">=3.9",
    install_requires=required,
    packages=find_packages(exclude="docs"),
    include_package_data=True,
    zip_safe=False,
)
