# Setup

## Prerequisites

1. AWS CLI configured with the appropriate region 
2. Python 3.8–3.11 
3. MacOS or Linux (Windows not currently supported)

## Installation
Let’s get set up with the SageMaker Hyperpod command line interface (CLI) and SDK. Install them via pip into your python3 environment:

```bash
pip install sagemaker-hyperpod
```
To verify CLI installation:
```bash
hp --version
```

## CLI

```bash
# List available clusters
hp cluster list

# Set the cluster context
hp cluster set-context <cluster-name>

# View current context
hp cluster get-context
```

## Python SDK
```python
from hyperpod.hyperpod_manager import HyperPodManager

hyperpod_manager = HyperPodManager()
hyperpod_manager.list_clusters()
hyperpod_manager.set_cluster(cluster_name="<cluster-name>")
hyperpod_manager.get_context()
```