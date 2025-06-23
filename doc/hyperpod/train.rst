Training with SageMaker Hyperpod
==================================

This section covers the tools and methods for creating, configuring, and managing distributed training jobs on HyperPod clusters.

Using CLI
------------

.. tabs::

   .. tab:: Create Training Job

      **Minimal parameters for quick job creation:**

      .. code-block:: bash

         hp hp-pytorch-job create \
           --job-name my-job \
           --image <docker-image> \
           --node-count 4

      **Using Config File for more complex configurations:**

      .. code-block:: bash

         hp hp-pytorch-job create --config-file <path-to-config.yaml>

      **Using Kubernetes YAML for advanced Kubernetes-native configurations:**

      .. code-block:: bash

         hp hp-pytorch-job create --k8s-yaml <path-to-k8s-spec.yaml>

      **Using Built-in Recipe for standardized training workflows:**

      .. code-block:: bash

         hp hp-pytorch-job create --recipe <recipe-name>

   .. tab:: Dry Runs and Interactive Modes

      **Generate Config File**

      .. code-block:: bash

         hp hp-pytorch-job create \
         --job-name my-job \
         --image <docker-image> \
         --node-count 4 \
         --generate-config

      **Generate Kubernetes YAML**

      .. code-block:: bash

         hp hp-pytorch-job create \
         --job-name my-job \
         --image <docker-image> \
         --node-count 4 \
         --generate-k8s-yaml

      **Generate Recipe with Customizations**

      .. code-block:: bash

         hp hp-pytorch-job create \
         --recipe <recipe-name> \
         --generate-recipe

      **Interactive Config Editing**

      .. code-block:: bash

         hp hp-pytorch-job create \
         --job-name my-job \
         --image <docker-image> \
         --node-count 4 \
         --editable

      **Interactive Recipe Editing**

      .. code-block:: bash

         hp hp-pytorch-job create \
           --recipe <recipe-name> \
           --editable

   .. tab:: Manage Training Jobs

      .. code-block:: bash

         # List all training jobs in the current namespace
         hp hp-pytorch-job list

         # Get detailed information about a specific job
         hp hp-pytorch-job get --job-name <job-name>

         # Remove a job and its associated resources
         hp hp-pytorch-job delete --job-name <job-name>

         # Temporarily pause a running job
         hp hp-pytorch-job patch --job-name <job-name> --suspend

         # Continue execution of a suspended job
         hp hp-pytorch-job patch --job-name <job-name> --resume

         # View all pods associated with a specific job
         hp hp-pytorch-job list-pods --job-name <job-name>

         # Execute commands inside a running pod
         hp hp-pytorch-job exec --job-name <job-name> --pod <pod> -- <command>

         # Retrieve and display logs from a specific pod
         hp hp-pytorch-job get-logs --job-name <job-name> --pod <pod>

   .. tab:: CLI Configuration Options

      **Job Identification**

      - --job-name (Required): Unique name for the training job

      - --namespace (Optional): Kubernetes namespace

      **Container Configuration**

      - --image (Required): Docker image for training

      - --entry-script (Optional): Script to execute

      - --script-args (Optional): Arguments for entry script

      - --environment (Optional): Key-value env variables

      - --pull-policy (Optional): Always | IfNotPresent | Never

      **Resource Allocation**

      - --node-count (Required): Number of nodes

      - --instance-type (Optional): AWS instance type

      - --tasks-per-node (Optional): Number of tasks per node

      **Node Selection**

      - --label-selector (Optional): Node label filter

      - --deep-health-check-passed-nodes-only (Optional)

      **Scheduling**

      - --scheduler-type (Optional): Kueue | SageMaker | None

      - --queue-name (Optional): Name of the queue

      - --priority (Optional): Priority level

      **Resilience**

      - --max-retry (Optional): Retry count on failure

      **Storage**

      - --volumes (Optional): List of volumes

      - --persistent-volume-claims (Optional): PVC mounts

      - --results-dir (Optional): Job results path

      - --service-account-name (Optional): K8s service account

      **Lifecycle Hooks**

      - --pre-script (Optional): Pre-job shell commands

      - --post-script (Optional): Post-job shell commands


Using SDK
---------

.. tabs::

   .. tab:: Create Training Job

      **Basic Job**

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

      **Advanced Job via Spec**

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

   .. tab:: Manage Training Job

      .. code-block:: python

         # Retrieve a list of all jobs in the specified namespace
         job.list_jobs(namespace="default")

         # Get detailed information about a specific job
         job.describe_job(name="my-job", namespace="default")

         # Remove a job and its associated resources
         job.delete_job(name="my-job", namespace="default")