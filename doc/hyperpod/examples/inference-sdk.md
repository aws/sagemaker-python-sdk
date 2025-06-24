---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Inference Operator PySDK E2E Experience

```{code-cell} ipython3
import sys
import warnings
import logging

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

sys.path.insert(0, '/Users/jzhaoqwa/Documents/GitHub/private-sagemaker-hyperpod-cli-staging/sagemaker-hyperpod/src/sagemaker')
```

<b>Prerequisite:</b> Data scientists should list clusters and set cluster context

```{code-cell} ipython3
from hyperpod.hyperpod_manager import HyperPodManager
hyperpod_manager = HyperPodManager()
```

```{code-cell} ipython3
hyperpod_manager.list_clusters()
```

```{code-cell} ipython3
# choose the HP cluster user works on
hyperpod_manager.set_current_cluster('ml-cluster')
```

```{code-cell} ipython3
# verify current kube context
hyperpod_manager.current_context()
```

## Create JumpStart model endpoint

+++

### Create from spec object (for experienced users)

```{code-cell} ipython3
from hyperpod.inference.config.jumpstart_model_endpoint_config import Model, Server, SageMakerEndpoint, JumpStartModelSpec
from hyperpod.inference.jumpstart_model_endpoint import JumpStartModelEndpoint
```

```{code-cell} ipython3
# create configs
model=Model(model_id='sklearn-regression-linear')
server=Server(instance_type='ml.t3.medium')
endpoint_name=SageMakerEndpoint(name='sklearn-regression-linear-endpoint')

# create spec
spec=JumpStartModelSpec(model=model, server=server, sage_maker_endpoint=endpoint_name)
```

```{code-cell} ipython3
# use spec to deploy
jumpstart_endpoint = JumpStartModelEndpoint()
jumpstart_endpoint.create(namespace='default', spec=spec)
```

<b>Note:</b> We auto-generate config class definitions above using script, such as `Model`, `Server`, `SageMakerEndpoint` and `JumpStartModelSpec`. This is based on [Inference CRD file](https://code.amazon.com/packages/AWSCrescendoInferenceOperator/blobs/mainline/--/dist/config/crd/inference.sagemaker.aws.amazon.com_jumpstartmodels.yaml).

+++

### Quick create with required inputs only

+++

This method overloads `create` function with required inputs. There is validation inside to make sure user cannot enter `spec` and other inputs at the same time.

```{code-cell} ipython3
quick_create_endpoint = JumpStartModelEndpoint()

# create with required inputs
quick_create_endpoint.create(
    namespace='default',
    model_id='sklearn-regression-linear',
    instance_type='ml.t3.medium',
    spec=None
)
```

### Alternative implementation (using different method names)

```{code-cell} ipython3
# create from full spec
'''
jumpstart_endpoint = JumpStartModelEndpoint()
jumpstart_endpoint.create_from_spec(namespace='default', spec=spec)
'''

# quick create from required inputs
'''
quick_create_endpoint = JumpStartModelEndpoint()
jumpstart_endpoint.create(
    namespace='default',
    model_id='sklearn-regression-linear',
    instance_type='ml.t3.medium',
    sagemaker_endpoint='sklearn-regression-linear-endpoint',
    spec=None
)
'''
```

### Other operations

```{code-cell} ipython3
# list all deployed endpoints
jumpstart_endpoint.list_endpoints(namespace='default')

# kubectl get jumpstartmodels
```

```{code-cell} ipython3
# describe deployed endpoints
jumpstart_endpoint.describe_endpoint(name='sklearn-regression-linear', namespace='default')

# kubectl describe jumpstartmodel sklearn-regression-linear
```

```{code-cell} ipython3
jumpstart_endpoint.delete_endpoint(name='sklearn-regression-linear', namespace='default')
```

```{code-cell} ipython3
# output should be empty after deletion
jumpstart_endpoint.list_endpoints(namespace='default')
```
