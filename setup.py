# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

from glob import glob
import os

from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name="sagemaker",
      version="1.2.5",
      description="Open source library for training and deploying models on Amazon SageMaker.",
      packages=find_packages('src'),
      package_dir={'': 'src'},
      py_modules=[os.splitext(os.basename(path))[0] for path in glob('src/*.py')],
      long_description=read('README.rst'),
      author="Amazon Web Services",
      url='https://github.com/aws/sagemaker-python-sdk/',
      license="Apache License 2.0",
      keywords="ML Amazon AWS AI Tensorflow MXNet",
      classifiers=[
          "Development Status :: 5 - Production/Stable",
          "Intended Audience :: Developers",
          "Natural Language :: English",
          "License :: OSI Approved :: Apache Software License",
          "Programming Language :: Python",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3.5",
      ],

      # Declare minimal set for installation
      install_requires=['boto3>=1.4.8', 'numpy>=1.9.0', 'protobuf>=3.1', 'scipy>=1.0.0', 'urllib3>=1.2',
                        'PyYAML>=3.2'],

      extras_require={
          'test': ['tox', 'flake8', 'pytest', 'pytest-cov', 'pytest-xdist',
                   'mock', 'tensorflow>=1.3.0', 'contextlib2', 'awslogs']},

      entry_points={
          'console_scripts': ['sagemaker=sagemaker.cli.main:main'],
      }
      )
