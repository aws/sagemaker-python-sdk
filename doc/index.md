# What is SageMaker Python SDK

```{note} Version Info
You're viewing latest documentation for **SageMaker Python SDK v3.0.1** ([view v2.x docs](https://sagemaker.readthedocs.io/en/v2.0.0/)).
```

```{admonition} What's New
:class: important

- ModelTrainer for simplified training workflows  
- ModelBuilder for easy model packaging  
- Support for HyperPod for distributed training  
```

The Amazon SageMaker Python SDK is an open source library for training and deploying machine learning models on Amazon SageMaker.

::::{container}
::::{grid}
:gutter: 3

:::{grid-item-card} Installation
:link: installation
:link-type: ref

Get the SDK setup
:::

:::{grid-item-card} Quickstart
:link: quickstart
:link-type: ref

Train and deploy your first model
:::

:::{grid-item-card} Hyperpod
:link: hyperpod-overview
:link-type: ref

Train and deploy using SageMaker Hyperpod.
:::

:::{grid-item-card} Examples
:link: examples-overview
:link-type: ref

Notebooks and code samples
:::

:::{grid-item-card} Migration
:link: installation
:link-type: ref

Migrate to latest SDK offerings
:::

::::
::::

## ML Lifecycle Features

::::{container}
::::{grid}
:gutter: 3

:::{grid-item-card} Build
:link: installation
:link-type: ref

Preprocessing 
:::

:::{grid-item-card} Train
:link: training-overview
:link-type: ref

Model trainer, estimator and advanced training
:::

:::{grid-item-card} Deploy
:link: hyperpod-overview
:link-type: ref

Model hosting, Batch transform, and inference
:::

:::{grid-item-card} ML Ops
:link: examples-overview
:link-type: ref

Pipelines and Monitoring
:::
::::
::::

```{toctree}
:caption: Getting Started
:hidden:
:maxdepth: 1

installation
quickstart
hyperpod/index
```

```{toctree}
:caption: Training Models
:hidden:
:maxdepth: 1

train
hyperpod/train
```

```{toctree}
:caption: Deploying Models
:hidden:
:maxdepth: 1

deploy
hyperpod/inference
```

```{toctree}
:caption: Data Processing
:hidden:
:maxdepth: 1

deploy
```

```{toctree}
:caption: ML Workflows
:hidden:
:maxdepth: 1

workflows/index

amazon_sagemaker_model_building_pipeline
amazon_sagemaker_model_monitoring
```

```{toctree}
:caption: API Reference
:hidden:
:maxdepth: 1

api/index
```

```{toctree}
:caption: Resources
:hidden:
:maxdepth: 1

examples/index
release-notes
```

```{toctree}
:caption: Temporary
:hidden:
:maxdepth: 1

overview
hyperpod/troubleshooting
hyperpod/examples/index
hyperpod/reference
hyperpod/installation
```