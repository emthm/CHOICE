from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# This call to setup() does all the work
setup(
    name="choice",
    version="1.0",
    description="Aircraft noise prediction calculation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="",
    author="Marily Thoma",
    author_email="marily@chalmers.se",
    license="gpl-3.0",
    classifiers=[
        "Intended Audience :: Researchers/Scientists/Developers/Engineers",
        "License :: OSI Approved :: GNU General Public License v3, GPLv3",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent"
    ],
    packages=["choice"],
    include_package_data=True,
    install_requires=["numpy~=1.21.6", "scipy~=1.7.3"]
)
