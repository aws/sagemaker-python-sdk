###############
Getting started
###############

************************************
Installing the SageMaker Python SDK
************************************

The SageMaker Python SDK is built to PyPI and can be installed with pip as follows:

.. code:: bash

    pip install sagemaker

You can install from the source by cloning this repository and running a pip install command in the root directory of the repository:

.. code:: bash

    git clone https://github.com/aws/sagemaker-python-sdk.git
    cd sagemaker-python-sdk
    pip install .

Supported Operating Systems
============================

SageMaker Python SDK supports Unix/Linux and Mac.

Supported Python Versions
============================

SageMaker Python SDK is tested on: ``Python 3.8``, ``Python 3.9`` and ``Python 3.10``.

If you are executing this pip install command in a notebook, make sure to restart your kernel.

*********************************************
Installing the SDK with minimum dependencies
*********************************************

By default, the SDK installs all dependencies including ``numpy``, ``pandas``,... For resource-constrained environments like Amazon Lambda, users can opt for a smaller installation footprint by installing the dependencies manually as follows:

Install the SDK without dependencies:

.. code:: bash

    pip install sagemaker --no-deps

Create a ``requirements.txt`` file with the following content:

.. include:: ../requirements/core_requirements.txt
   :literal:

Then install the dependencies:

.. code:: bash

    pip install -r requirements.txt
