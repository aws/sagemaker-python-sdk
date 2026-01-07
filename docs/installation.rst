Installation
============

This guide covers how to install SageMaker Python SDK V3 and set up your development environment.

Quick Installation
------------------

Install the latest version of SageMaker Python SDK V3:

.. code-block:: bash

   pip install sagemaker

Prerequisites
---------------

**Python Version**
  SageMaker Python SDK V3 supports Python 3.9, 3.10, 3.11, and 3.12

**Operating Systems**
  - Linux
  - macOS

**AWS Credentials**
  Configure AWS credentials using one of these methods:
  
  - AWS CLI: ``aws configure``
  - Environment variables: ``AWS_ACCESS_KEY_ID`` and ``AWS_SECRET_ACCESS_KEY``
  - IAM roles

Installation Methods
----------------------

Standard Installation
~~~~~~~~~~~~~~~~~~~~~

Install the complete SageMaker Python SDK V3:

.. code-block:: bash

   pip install sagemaker

Modular Installation
~~~~~~~~~~~~~~~~~~~

Install specific components based on your needs:

.. code-block:: bash

   # Core functionality only
   pip install sagemaker-core
   
   # Training capabilities
   pip install sagemaker-train
   
   # Inference capabilities  
   pip install sagemaker-serve
   
   # ML Operations
   pip install sagemaker-mlops

Virtual Environment (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create an isolated environment for your SageMaker projects:

.. code-block:: bash

   # Using venv
   python -m venv sagemaker-v3-env
   source sagemaker-v3-env/bin/activate
   pip install sagemaker
   
   # Using conda
   conda create -n sagemaker-v3 python=3.10
   conda activate sagemaker-v3
   pip install sagemaker

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~

Install the SageMaker Python SDK directly from source when you want to:

- develop locally,
- test your own modifications, or
- work with the latest unreleased features.

.. code-block:: bash

   git clone https://github.com/aws/sagemaker-python-sdk.git
   cd sagemaker-python-sdk
   pip install -e .

This installs the top-level sagemaker package in editable mode, so any changes you make to its code are picked up immediately without needing to reinstall.

**Working with SDK Submodules**

The repository contains additional installable components such as:

- sagemaker-core
- sagemaker-train
- sagemaker-serve
- sagemaker-mlops

If you plan to modify code inside one of these packages, install that submodule in editable mode as well:

.. code-block:: bash

   cd sagemaker-core
   pip install -e .

Repeat for any other submodule you are actively developing.

Editable installations operate at the Python package level, not the Git repository level. Installing the relevant submodule ensures your local changes are reflected during development.

Verification
-----------

Verify your installation:

.. code-block:: python

   import sagemaker
   print(f"SageMaker SDK version: {sagemaker.__version__}")
   
   # Check if you can create a session
   import sagemaker
   session = sagemaker.Session()
   print(f"Default bucket: {session.default_bucket()}")
   print(f"Region: {session.boto_region_name}")

Troubleshooting
--------------

**Common Issues:**

*ImportError: No module named 'sagemaker'*
  - Ensure you're using the correct Python environment
  - Verify installation with ``pip list | grep sagemaker``

*Permission denied errors*
  - Use ``pip install --user sagemaker`` for user-level installation
  - Or use a virtual environment

*AWS credential errors*
  - Configure AWS credentials: ``aws configure``
  - Verify with ``aws sts get-caller-identity``

*Version conflicts*
  - Uninstall old versions: ``pip uninstall sagemaker``
  - Install fresh: ``pip install sagemaker``

Upgrading from V2
-----------------

If you have SageMaker Python SDK V2 installed:

.. code-block:: bash

   # Upgrade to V3
   pip install --upgrade sagemaker
   
   # Or install V3 in a new environment (recommended)
   python -m venv sagemaker-v3-env
   source sagemaker-v3-env/bin/activate
   pip install sagemaker

**Note:** V3 introduces breaking changes. See the :doc:`overview` page for migration guidance.

Next Steps
----------

After installation:

1. **Configure AWS credentials** if you haven't already
2. **Read the** :doc:`overview` **to understand V3 changes**
3. **Try the** :doc:`quickstart` **guide**
4. **Explore** :doc:`training/index`, :doc:`inference/index`, and other capabilities
