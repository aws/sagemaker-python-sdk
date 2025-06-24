# CLI Reference
## Cluster Commands
```bash
hp cluster list
hp cluster set-context <name>
hp cluster get-context
```

## Training Commands
```bash
hp hp-pytorch-job create ...
hp hp-pytorch-job list
hp hp-pytorch-job get --job-name <name>
hp hp-pytorch-job delete --job-name <name>
```
## Inference Commands
```bash
hp hp-jumpstart-endpoint create ...
hp hp-endpoint create ...
hp hp-jumpstart-endpoint invoke ...
```
For a full list of supported options for each command, run:
```bash
hp <command> --help
```

## Configurable Flags
- `--job-name`, `--image`, `--node-count`, etc.

# API Reference
The Python SDK modules are structured as follows:

## Cluster Management
```python
from hyperpod.hyperpod_manager import HyperPodManager
```

## Training
```python
from sagemaker.hyperpod.training import HyperPodPytorchJob
```
## Inference
```python
from hyperpod.inference.hp_endpoint import HPEndpoint
from hyperpod.inference.hp_jumpstart_endpoint import HPJumpStartEndpoint
```

Each module supports:

.create() methods for launching jobs or endpoints

.create_from_spec() methods for advanced use

.describe(), .list(), .delete() methods for lifecycle management