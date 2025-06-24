---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# SageMaker HyperPod PySDK Training Example

This example demonstrates how to create and manage distributed training jobs using the SageMaker HyperPod Python SDK.

**Prerequisites:**
- A working EKS HyperPod cluster
- Installed `sagemaker-hyperpod` SDK

## Steps
1. Cluster Selection and Context
2. Quick Create Training Job
3. Advanced Spec-based Training Job
4. Job Management

```{code-cell}
import sys
import warnings
import logging

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

# Adjust the path as needed
sys.path.insert(0, '/Users/jzhaoqwa/Documents/GitHub/private-sagemaker-hyperpod-cli-staging/sagemaker-hyperpod/src/sagemaker')
```

## 1. Cluster Selection and Context

```{code-cell}
from hyperpod.hyperpod_manager import HyperPodManager
hyperpod_manager = HyperPodManager()
hyperpod_manager.list_clusters()
```

```{code-cell}
# Set the cluster context
hyperpod_manager.set_current_cluster("ml-cluster")
hyperpod_manager.current_context()
```

## 2. Quick Create Training Job

```{code-cell}
from sagemaker.hyperpod.training import HyperPodPytorchJob

job = HyperPodPytorchJob.create(
    job_name="my-quick-training-job",
    image="python:3.8-slim",
    node_count=2,
    entry_script="train.py",
    script_args="--epochs 10",
    environment={"LEARNING_RATE": "0.001"},
    namespace="default"
)
```

## 3. Advanced Spec-Based Training Job

```{code-cell}
from sagemaker.hyperpod.training import (
    HyperPodPytorchJob, HyperPodPytorchJobSpec,
    ReplicaSpec, Template, Spec, Container
)

advanced_spec = HyperPodPytorchJobSpec(
    nproc_per_node=2,
    replica_specs=[
        ReplicaSpec(
            name="trainer",
            template=Template(
                spec=Spec(
                    containers=[
                        Container(
                            name="trainer",
                            image="python:3.8-slim",
                            command=["python"],
                            args=["train.py", "--epochs", "5"]
                        )
                    ]
                )
            )
        )
    ]
)

advanced_job = HyperPodPytorchJob.create_from_spec(
    job_name="my-advanced-job",
    namespace="default",
    spec=advanced_spec
)
```

## 4. Manage Training Jobs

```{code-cell}
# List jobs
HyperPodPytorchJob.list_jobs(namespace="default")
```

```{code-cell}
# Describe job
HyperPodPytorchJob.describe_job(name="my-quick-training-job", namespace="default")
```

```{code-cell}
# Delete job
HyperPodPytorchJob.delete_job(name="my-quick-training-job", namespace="default")
```
