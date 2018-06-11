#!/usr/bin/python
from setuptools import find_packages
from setuptools import setup

setup(
    name='mlengine-flowers',
    version='0.1',
    author='Matthias Feys',
    author_email='matthiasfeys@gmail.com',
    install_requires=['tensorflow==1.8.0','tensorflow-transform==0.6.0', 'Pillow==5.0.0', 'numpy==1.14.0'],
    packages=find_packages(),
    scripts=[],
    package_data={
        'trainer': ['*'],  # include any none python files in trainer
    },
    description='ML Engine boilerplate code',
    url='https://github.com/Fematich/mlengine-boilerplate'
)
