Version Lifecycle
=================

The SageMaker Python SDK follows the `AWS SDKs and Tools maintenance policy
<https://docs.aws.amazon.com/sdkref/latest/guide/maint-policy.html>`_. This page
describes the maintenance lifecycle for the major versions of the SDK and the
timelines associated with each phase.

Version 3 became generally available on 2025-11-19 and is the actively developed
major version. Version 2 is in maintenance mode: it receives critical bug fixes
and security updates only, and will not receive new features, API updates, or new
region support.

.. note::
   We recommend all users migrate to V3. See the `migration guide
   <https://github.com/aws/sagemaker-python-sdk/blob/master/migration.md>`_ for
   detailed guidance on moving from V2 to V3.

Version Support Matrix
----------------------

The following table shows the available major versions of the SageMaker Python
SDK and where each is in the maintenance lifecycle.

.. list-table::
   :header-rows: 1
   :widths: 20 25 35 20

   * - Major Version
     - Current Phase
     - Install
     - General Availability
   * - Version 3
     - General Availability
     - ``pip install sagemaker``
     - 2025-11-19
   * - Version 2
     - Maintenance mode
     - ``pip install "sagemaker<3"``
     - 2020-08-04

Lifecycle Phases
----------------

Each major version moves through the following phases, as defined by the
`AWS SDKs and Tools maintenance policy
<https://docs.aws.amazon.com/sdkref/latest/guide/maint-policy.html>`_:

- **General Availability** -- the SDK is fully supported, with regular releases
  for new services, API updates, and bug and security fixes.
- **Maintenance mode** -- AWS limits releases to critical bug fixes and security
  issues only. No new APIs, features, or region support are added.
- **End-of-Support** -- the SDK no longer receives updates or releases.
  Previously published versions remain available on PyPI and the source remains
  on GitHub.

Version 3
---------

Version 3 is the current, actively developed major version and is in the
**General Availability** phase.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Phase
     - Start Date
     - End Date
   * - General Availability
     - 2025-11-19
     - N/A

Version 2
---------

Version 2 is in **Maintenance mode**. The following table shows its lifecycle
phases and timelines.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Phase
     - Start Date
     - End Date
   * - General Availability
     - 2020-08-04
     - N/A
   * - Maintenance mode
     - 2026-07-06
     - 2027-07-05
   * - End-of-Support
     - 2027-07-06
     - N/A

Managing Your Version
---------------------

To upgrade to the latest V3 release:

.. code-block:: bash

   pip install --upgrade sagemaker

To pin to V2 during migration:

.. code-block:: bash

   pip install "sagemaker<3"

Next Steps
----------

- Read the :doc:`overview` to understand what changed in V3
- Follow the :doc:`installation` guide to set up V3
- Review the `migration guide
  <https://github.com/aws/sagemaker-python-sdk/blob/master/migration.md>`_ to move
  from V2 to V3
