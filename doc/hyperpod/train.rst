.. _training-with-hyperpod:

Training with SageMaker HyperPod
================================

This section explains how to launch and manage distributed training jobs on SageMaker HyperPod clusters using both the CLI and Python SDK. Whether you're quickly iterating on a model or scaling production workloads, HyperPod offers flexible ways to define and manage training jobs.

.. note::

   This guide applies to HyperPod CLI v0.5+ and SDK v0.3+.  
   Run ``hp version`` or ``pip show sagemaker`` to check your versions.

Using CLI
---------

You can create training jobs via CLI using one of four approaches:

- Minimal CLI parameters (quick start)
- Config file (for reproducibility)
- Kubernetes YAML (Kubernetes-native)
- Built-in recipes (preconfigured workflows)

Create a Training Job with CLI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tabs::

   .. tab:: Quick Start

      Ideal for prototyping and quick iteration.

      .. code-block:: bash

         hp hp-pytorch-job create \
           --job-name my-job \
           --image <docker-image> \
           --node-count 4

   .. tab:: Using Config File

      Recommended for reproducible and team-shared jobs.

      .. code-block:: bash

         hp hp-pytorch-job create --config-file config.yaml

   .. tab:: Using Kubernetes YAML

      For full control using standard Kubernetes specs.

      .. code-block:: bash

         hp hp-pytorch-job create --k8s-yaml k8s-spec.yaml

   .. tab:: Using Built-in Recipe

      Launch common workloads using predefined templates.

      .. code-block:: bash

         hp hp-pytorch-job create --recipe resnet50

Dry Runs and Interactive Editing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use these options to preview or modify your job before launching:

.. tabs::

   .. tab:: Generate Config

      .. code-block:: bash

         hp hp-pytorch-job create \
           --job-name my-job \
           --image <docker-image> \
           --node-count 4 \
           --generate-config

   .. tab:: Generate Kubernetes YAML

      .. code-block:: bash

         hp hp-pytorch-job create \
           --job-name my-job \
           --image <docker-image> \
           --node-count 4 \
           --generate-k8s-yaml

   .. tab:: Generate Recipe

      .. code-block:: bash

         hp hp-pytorch-job create \
           --recipe <recipe-name> \
           --generate-recipe

   .. tab:: Interactive Editing

      Customize config or recipe using interactive editor:

      .. code-block:: bash

         hp hp-pytorch-job create --editable

.. note::

   Use ``--editable`` to tweak job parameters before launching — ideal for debugging or quick experimentation.

Manage Training Jobs with CLI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # List all jobs
   hp hp-pytorch-job list

   # Describe a specific job
   hp hp-pytorch-job get --job-name <job-name>

   # Delete a job and its resources
   hp hp-pytorch-job delete --job-name <job-name>

   # Suspend and resume jobs
   hp hp-pytorch-job patch --job-name <job-name> --suspend
   hp hp-pytorch-job patch --job-name <job-name> --resume

   # List pods or view logs
   hp hp-pytorch-job list-pods --job-name <job-name>
   hp hp-pytorch-job get-logs --job-name <job-name> --pod <pod>

   # Execute commands inside a pod
   hp hp-pytorch-job exec --job-name <job-name> --pod <pod> -- <command>

CLI Configuration Reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Job Identification**

- ``--job-name`` *(Required)*: Unique job name  
- ``--namespace`` *(Optional)*: Kubernetes namespace

**Container Configuration**

- ``--image`` *(Required)*: Docker image  
- ``--entry-script`` *(Optional)*  
- ``--script-args`` *(Optional)*  
- ``--environment`` *(Optional)*: KEY=VALUE  
- ``--pull-policy`` *(Optional)*: Always \| IfNotPresent \| Never

**Resources and Scheduling**

- ``--node-count`` *(Required)*  
- ``--instance-type`` *(Optional)*  
- ``--tasks-per-node`` *(Optional)*  
- ``--label-selector`` *(Optional)*  
- ``--deep-health-check-passed-nodes-only`` *(Optional)*  
- ``--scheduler-type`` *(Optional)*: Kueue \| SageMaker \| None  
- ``--queue-name`` *(Optional)*  
- ``--priority`` *(Optional)*

**Storage and Lifecycle**

- ``--volumes`` *(Optional)*  
- ``--persistent-volume-claims`` *(Optional)*  
- ``--results-dir`` *(Optional)*  
- ``--service-account-name`` *(Optional)*  
- ``--pre-script`` / ``--post-script`` *(Optional)*

Using Python SDK
----------------

The Python SDK is ideal when integrating HyperPod jobs into notebooks, pipelines, or custom automation scripts.

Choose the right method depending on your level of customization:

.. list-table::
   :header-rows: 1

   * - Method
     - Use When
     - Example
   * - ``create()``
     - Standard jobs using basic parameters
     - Entry script, env vars, container image
   * - ``create_from_spec()``
     - Advanced jobs needing custom pod specs or multi-container configs
     - Multiple replicas, fine-grained control

Create a Training Job with Python SDK
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tabs::

   .. tab:: Basic Job

      .. code-block:: python

         from sagemaker.hyperpod.training import HyperPodPytorchJob

         job = HyperPodPytorchJob.create(
             job_name="my-training-job",
             image="python:tag",
             node_count=4,
             entry_script="train.py",
             script_args="--epochs 10",
             environment={"LEARNING_RATE": "0.001"},
             namespace="kubeflow"
         )

   .. tab:: Advanced Job (Spec)

      .. code-block:: python

         from sagemaker.hyperpod.training import (
             HyperPodPytorchJob, HyperPodPytorchJobSpec,
             ReplicaSpec, Template, Spec, Container
         )

         spec = HyperPodPytorchJobSpec(
             nproc_per_node=2,
             replica_specs=[
                 ReplicaSpec(
                     name="trainer",
                     template=Template(
                         spec=Spec(
                             containers=[
                                 Container(name="trainer", image="python:tag")
                             ]
                         )
                     )
                 )
             ]
         )

         job = HyperPodPytorchJob.create_from_spec(
             job_name="advanced-training-job",
             namespace="kubeflow",
             spec=spec
         )

Manage Training Jobs with Python SDK
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sagemaker.hyperpod.training import HyperPodPytorchJob

   # List all jobs
   HyperPodPytorchJob.list_jobs(namespace="default")

   # Describe a job
   HyperPodPytorchJob.describe_job(name="my-job", namespace="default")

   # Delete a job
   HyperPodPytorchJob.delete_job(name="my-job", namespace="default")

.. note::
   Coming soon: SDK support for streaming logs, retrying failed jobs, and waiting for job completion.

Examples
--------

Sample Config File (`config.yaml`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   jobName: my-training-job
   image: python:tag
   nodeCount: 4
   entryScript: train.py
   scriptArgs: "--epochs 10"
   environment:
     LEARNING_RATE: 0.001

Sample Kubernetes YAML (`k8s-spec.yaml`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   apiVersion: batch/v1
   kind: Job
   metadata:
     name: my-training-job
   spec:
     template:
       spec:
         containers:
           - name: trainer
             image: python:tag
             command: ["python", "train.py"]
         restartPolicy: Never
