from setuptools import setup, find_packages
import numpy as np

# Package metadata
NAME = 'sliding_nucleosome'
VERSION = '0.1.0'
AUTHOR = 'Joseph Wakim and Andrew Spakowitz'
AUTHOR_EMAIL = 'ajspakow@stanford.edu'
DESCRIPTION = 'Linear polymer model with sliding binding sites'
URL = 'https://github.com/JosephWakim/sliding_nucleosome.git'
KEYWORDS = 'thermodynamics, epigenetics, nucleosomes, euchromatin, polymer'
CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science Research',
    'Programming Language :: Python :: 3.9',
]

# Package dependencies
INSTALL_REQUIRES = [
    'numpy~=1.21.6',
    'pandas~=1.3.5',
    'matplotlib~=3.5.2',
    'notebook~=6.0.0',
    'jupyter~=1.0.0',
    'scipy~=1.11.1'
]

# Read the long description from README.md
with open('README.md', 'r', encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

# Setup configuration
setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url=URL,
    keywords=KEYWORDS,
    classifiers=CLASSIFIERS,
    packages=find_packages(include=["sliding_nucleosome", "binding_model"]),
    include_package_data=True,
    include_dirs=[np.get_include(), "sliding_nucleosome", "binding_model"],
    install_requires=INSTALL_REQUIRES,
)
