from setuptools import setup
import os
import sys

if sys.version_info[0] < 3:
    with open('README.rst') as f:
        long_description = f.read()
else:
    with open('README.rst', encoding='utf-8') as f:
        long_description = f.read()


setup(
    name='MetaHeuristicsFS',
    version='0.0.2',
    description='Implementation of metaheuristic algorithms for machine learning feature selection. Companion library for the book `Feature Engineering & Selection for Explainable Models A Second Course for Data Scientists`',
    long_description=long_description,
    long_description_content_type='text/markdown',  # This is important!
    author='StatguyUser',
    url='https://github.com/StatguyUser/MetaHeuristicsFS',
    install_requires=['numpy','scikit-learn'],
    download_url='https://github.com/MetaHeuristicsFS/MetaHeuristicsFS.git',
    py_modules=["MetaHeuristicsFS"],
    package_dir={'':'src'},
)
