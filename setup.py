from setuptools import find_packages, setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='DeePipe',
    packages=find_packages(),
    install_requires=required,
    version='0.3.3',
    description='The AWS based Deep Learning Pipeline Framework',
    author='MOPC',
    author_email="s202798@student.dtu.dk",
    license='MIT',
    project_urls={
        "Source Code": 'https://github.com/marcoplacenti/DeePipe'
    }
)
