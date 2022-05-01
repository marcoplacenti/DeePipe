from setuptools import find_packages, setup
import os

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='DeePipe',
    packages=find_packages(),
    install_requires=required,
    version='0.0.2',
    description='DL Pipeline',
    author='MOPC',
    license='MIT',
    project_urls={
        "Source Code": 'https://github.com/marcoplacenti/DeePipe'
    }
)
