#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='akpy',
      version='0.0.1',
      description='Python code for DSP and telecom',
      author='LASSE',
      author_email='aldebaro',
      packages=find_packages(exclude=['tests']),
      url='https://gitlab.lasse.ufpa.br/software/ak-py',
      install_requires=['numpy(>=1.14)'])
