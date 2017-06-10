#!/usr/bin/python
from setuptools import find_packages
from setuptools import setup

setup(
    name='mlengine-boilerplate',
    version='0.1',
    author='Matthias Feys',
    author_email='matthiasfeys@gmail.com',
    install_requires=['tensorflow==1.1.0','tensorflow-transform==0.1.10'],
    packages=find_packages(
        exclude=['data', 'predictions']),
    scripts=[],
    package_data={
        'trainer': ['*'],  # include any none python files in trainer
    },
    description='ML Engine boilerplate code',
    url='https://github.com/Fematich/mlengine-boilerplate'
)
