#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='MultiCYP',
    description='An end-to-end metabolite prediction model',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version='1.0.0',
    packages=find_packages(),
    project_urls={
        "Documentation": "https://github.com/YilingZhou/MultiCYP/",
        "Forum": "https://github.com/YilingZhou/MultiCYP/",
        "Gitter": "https://github.com/YilingZhou/MultiCYP/",
        "Source": "https://github.com/YilingZhou/MultiCYP/"
    },
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.8.0",
        "rdkit",
        "numpy>=1.19.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0"
    ]
)
