Deploy Models for Inference
============================

SageMaker Python SDK V3 transforms model deployment and inference with the unified **ModelBuilder** class, replacing the complex framework-specific model classes from V2. This modern approach provides a consistent interface for all inference scenarios while maintaining the flexibility and performance you need.

Key Benefits of V3 Inference
----------------------------

* **Unified Interface**: Single ``ModelBuilder`` class replaces multiple framework-specific model classes
* **Simplified Deployment**: Object-oriented API with intelligent defaults for endpoint configuration
* **Enhanced Performance**: Optimized inference pipelines with automatic scaling and load balancing
* **Multi-Modal Support**: Deploy models for real-time, batch, and serverless inference scenarios

Quick Start Example
-------------------

Here's how inference has evolved from V2 to V3:

**SageMaker Python SDK V2:**

.. code-block:: python

   from sagemaker.model import Model
   from sagemaker.predictor import Predictor
   
   model = Model(
       image_uri="my-inference-image",
       model_data="s3://my-bucket/model.tar.gz",
       role="arn:aws:iam::123456789012:role/SageMakerRole"
   )
   predictor = model.deploy(
       initial_instance_count=1,
       instance_type="ml.m5.xlarge"
   )
   result = predictor.predict(data)

**SageMaker Python SDK V3:**

.. code-block:: python

   from sagemaker.serve import ModelBuilder

   model_builder = ModelBuilder(
       model="my-model",
       model_path="s3://my-bucket/model.tar.gz"
   )

   model = model_builder.build(model_name="my-deployed-model")

   endpoint = model_builder.deploy(
       endpoint_name="my-endpoint",
       instance_type="ml.m5.xlarge",
       initial_instance_count=1
   )

   result = endpoint.invoke(
       body=data,
       content_type="application/json"
   )



Custom InferenceSpec
--------------------


Define custom model loading and inference logic by extending ``InferenceSpec``. Implement ``load()`` to deserialize your model and ``invoke()`` to run predictions.

.. code-block:: python

   from sagemaker.serve.model_builder import ModelBuilder
   from sagemaker.serve.spec.inference_spec import InferenceSpec
   from sagemaker.serve.builder.schema_builder import SchemaBuilder
   from sagemaker.serve.utils.types import ModelServer

   class MyModelSpec(InferenceSpec):
       def load(self, model_dir: str):
           import torch
           return torch.jit.load(f"{model_dir}/model.pth", map_location="cpu")

       def invoke(self, input_object, model):
           import torch
           tensor = torch.tensor(input_object, dtype=torch.float32)
           with torch.no_grad():
               return model(tensor).tolist()

   schema_builder = SchemaBuilder(
       [[0.1, 0.2, 0.3, 0.4]],   # sample input
       [[0.9, 0.1]]               # sample output
   )

   model_builder = ModelBuilder(
       inference_spec=MyModelSpec(),
       model_path="./model_artifacts",
       model_server=ModelServer.TORCHSERVE,
       schema_builder=schema_builder,
   )

   core_model = model_builder.build(model_name="my-custom-model")
   endpoint = model_builder.deploy(endpoint_name="my-endpoint")

   result = endpoint.invoke(
       body=json.dumps([[0.1, 0.2, 0.3, 0.4]]),
       content_type="application/json"
   )

:doc:`Full example notebook <../v3-examples/inference-examples/inference-spec-example>`



JumpStart Model Deployment
--------------------------


Deploy pre-trained models from the JumpStart hub using ``ModelBuilder.from_jumpstart_config()``.

.. code-block:: python

   from sagemaker.serve.model_builder import ModelBuilder
   from sagemaker.core.jumpstart.configs import JumpStartConfig
   from sagemaker.train.configs import Compute

   compute = Compute(instance_type="ml.g5.2xlarge")
   jumpstart_config = JumpStartConfig(model_id="huggingface-llm-falcon-7b-bf16")

   model_builder = ModelBuilder.from_jumpstart_config(
       jumpstart_config=jumpstart_config,
       compute=compute,
   )

   core_model = model_builder.build(model_name="falcon-model")
   endpoint = model_builder.deploy(endpoint_name="falcon-endpoint")

   result = endpoint.invoke(
       body=json.dumps({"inputs": "What are falcons?", "parameters": {"max_new_tokens": 32}}),
       content_type="application/json"
   )

:doc:`Full example notebook <../v3-examples/inference-examples/jumpstart-example>`



Model Optimization (Quantization)
----------------------------------


Optimize models with quantization (e.g., AWQ) using ``model_builder.optimize()`` before deployment.

.. code-block:: python

   from sagemaker.serve.model_builder import ModelBuilder
   from sagemaker.serve.builder.schema_builder import SchemaBuilder

   schema_builder = SchemaBuilder(
       {"inputs": "What are falcons?", "parameters": {"max_new_tokens": 32}},
       [{"generated_text": "Falcons are birds of prey."}]
   )

   model_builder = ModelBuilder(
       model="meta-textgeneration-llama-3-8b-instruct",
       schema_builder=schema_builder,
   )

   optimized_model = model_builder.optimize(
       instance_type="ml.g5.2xlarge",
       quantization_config={"OverrideEnvironment": {"OPTION_QUANTIZE": "awq"}},
       accept_eula=True,
       job_name="optimize-llama",
       model_name="llama-optimized",
   )

   endpoint = model_builder.deploy(endpoint_name="llama-endpoint", initial_instance_count=1)

:doc:`Full example notebook <../v3-examples/inference-examples/optimize-example>`



Train-to-Inference End-to-End
------------------------------


Pass a ``ModelTrainer`` directly to ``ModelBuilder`` to go from training to deployment in one flow.

.. code-block:: python

   from sagemaker.train.model_trainer import ModelTrainer
   from sagemaker.train.configs import SourceCode
   from sagemaker.serve.model_builder import ModelBuilder
   from sagemaker.serve.spec.inference_spec import InferenceSpec
   from sagemaker.serve.builder.schema_builder import SchemaBuilder
   from sagemaker.serve.utils.types import ModelServer

   # Train
   trainer = ModelTrainer(
       training_image="pytorch-training:1.13.1-cpu-py39",
       source_code=SourceCode(source_dir="./src", entry_script="train.py"),
       base_job_name="my-training",
   )
   trainer.train()

   # Deploy from trainer
   model_builder = ModelBuilder(
       model=trainer,
       schema_builder=SchemaBuilder([[0.1, 0.2, 0.3, 0.4]], [[0.8, 0.2]]),
       model_server=ModelServer.TORCHSERVE,
       inference_spec=MyInferenceSpec(),
   )

   core_model = model_builder.build(model_name="trained-model")
   endpoint = model_builder.deploy(endpoint_name="trained-endpoint", initial_instance_count=1)

:doc:`Full example notebook <../v3-examples/inference-examples/train-inference-e2e-example>`



JumpStart Train-to-Inference
-----------------------------


Train a JumpStart model with ``ModelTrainer.from_jumpstart_config()`` then deploy via ``ModelBuilder``.

.. code-block:: python

   from sagemaker.train.model_trainer import ModelTrainer
   from sagemaker.serve.model_builder import ModelBuilder
   from sagemaker.core.jumpstart.configs import JumpStartConfig

   jumpstart_config = JumpStartConfig(model_id="huggingface-spc-bert-base-cased")

   trainer = ModelTrainer.from_jumpstart_config(
       jumpstart_config=jumpstart_config,
       base_job_name="js-training",
       hyperparameters={"epochs": 1},
   )
   trainer.train()

   model_builder = ModelBuilder(model=trainer, dependencies={"auto": False})
   core_model = model_builder.build(model_name="bert-trained")
   endpoint = model_builder.deploy(endpoint_name="bert-endpoint")

:doc:`Full example notebook <../v3-examples/inference-examples/jumpstart-e2e-training-example>`



HuggingFace Model Deployment
------------------------------


Deploy HuggingFace models with a custom ``InferenceSpec`` using Multi Model Server (MMS).

.. code-block:: python

   from sagemaker.serve.model_builder import ModelBuilder
   from sagemaker.serve.spec.inference_spec import InferenceSpec
   from sagemaker.serve.builder.schema_builder import SchemaBuilder
   from sagemaker.serve.utils.types import ModelServer

   class HFSpec(InferenceSpec):
       def load(self, model_dir):
           from transformers import AutoTokenizer, AutoModelForCausalLM
           tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
           model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
           return {"model": model, "tokenizer": tokenizer}

       def invoke(self, input_object, model):
           text = input_object["inputs"]
           inputs = model["tokenizer"].encode(text, return_tensors="pt")
           outputs = model["model"].generate(inputs, max_length=inputs.shape[1] + 20)
           return [{"generated_text": model["tokenizer"].decode(outputs[0], skip_special_tokens=True)}]

   model_builder = ModelBuilder(
       inference_spec=HFSpec(),
       model_server=ModelServer.MMS,
       schema_builder=SchemaBuilder(
           {"inputs": "Hello, how are you?"},
           [{"generated_text": "I'm doing well!"}]
       ),
   )

   core_model = model_builder.build(model_name="hf-dialogpt")
   endpoint = model_builder.deploy(endpoint_name="hf-endpoint")

:doc:`Full example notebook <../v3-examples/inference-examples/huggingface-example>`



In-Process Mode
----------------


Run inference entirely in your Python process with no containers or AWS resources. Use ``Mode.IN_PROCESS`` and ``deploy_local()``.

.. code-block:: python

   from sagemaker.serve.model_builder import ModelBuilder
   from sagemaker.serve.spec.inference_spec import InferenceSpec
   from sagemaker.serve.builder.schema_builder import SchemaBuilder
   from sagemaker.serve.mode.function_pointers import Mode

   class MathSpec(InferenceSpec):
       def load(self, model_dir):
           return {"factor": 2.0}

       def invoke(self, input_object, model):
           numbers = input_object["numbers"]
           return {"result": [n * model["factor"] for n in numbers]}

   model_builder = ModelBuilder(
       inference_spec=MathSpec(),
       schema_builder=SchemaBuilder({"numbers": [1.0, 2.0]}, {"result": [2.0, 4.0]}),
       mode=Mode.IN_PROCESS,
   )

   core_model = model_builder.build(model_name="math-model")
   local_endpoint = model_builder.deploy_local(endpoint_name="math-local")

   result = local_endpoint.invoke(body={"numbers": [3.0, 5.0]}, content_type="application/json")

:doc:`Full example notebook <../v3-examples/inference-examples/in-process-mode-example>`



Local Container Mode
---------------------


Test models in Docker containers locally using ``Mode.LOCAL_CONTAINER`` and ``deploy_local()``. Same container environment as SageMaker endpoints.

.. code-block:: python

   from sagemaker.serve.model_builder import ModelBuilder
   from sagemaker.serve.spec.inference_spec import InferenceSpec
   from sagemaker.serve.builder.schema_builder import SchemaBuilder
   from sagemaker.serve.utils.types import ModelServer
   from sagemaker.serve.mode.function_pointers import Mode

   model_builder = ModelBuilder(
       inference_spec=MyPyTorchSpec(model_path="./model"),
       model_server=ModelServer.TORCHSERVE,
       schema_builder=SchemaBuilder([[1.0, 2.0, 3.0, 4.0]], [[0.6, 0.4]]),
       mode=Mode.LOCAL_CONTAINER,
   )

   local_model = model_builder.build(model_name="local-pytorch")
   local_endpoint = model_builder.deploy_local(
       endpoint_name="local-pytorch-ep",
       wait=True,
       container_timeout_in_seconds=1200,
   )

   response = local_endpoint.invoke(
       body=json.dumps([[1.0, 2.0, 3.0, 4.0]]),
       content_type="application/json"
   )

:doc:`Full example notebook <../v3-examples/inference-examples/local-mode-example>`



Inference Pipelines (Multi-Container)
--------------------------------------


Chain multiple containers into a serial inference pipeline. Pass a list of ``Model`` objects to ``ModelBuilder``.

.. code-block:: python

   from sagemaker.core.resources import Model
   from sagemaker.core.shapes import ContainerDefinition
   from sagemaker.core.utils import repack_model
   from sagemaker.serve import ModelBuilder

   # Create individual models with primary_container
   sklearn_model = Model.create(
       model_name="sklearn-preprocess",
       primary_container=ContainerDefinition(
           image=sklearn_image,
           model_data_url=sklearn_repacked_uri,
           environment={"SAGEMAKER_PROGRAM": "inference.py"},
       ),
       execution_role_arn=role,
   )

   xgboost_model = Model.create(
       model_name="xgboost-classifier",
       primary_container=ContainerDefinition(
           image=xgboost_image,
           model_data_url=xgboost_model_uri,
       ),
       execution_role_arn=role,
   )

   # Build and deploy pipeline
   pipeline_builder = ModelBuilder(
       model=[sklearn_model, xgboost_model],
       role_arn=role,
   )
   pipeline_model = pipeline_builder.build()
   endpoint = pipeline_builder.deploy(
       endpoint_name="pipeline-ep",
       instance_type="ml.m5.large",
       initial_instance_count=1,
   )

   response = endpoint.invoke(body=csv_data, content_type="text/csv", accept="text/csv")

:doc:`Full example notebook <../v3-examples/inference-examples/inference-pipeline-modelbuilder-vs-core-example>`



Migration from V2
------------------


Inference Classes and Imports
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - V2
     - V3
   * - ``sagemaker.model.Model``
     - ``sagemaker.serve.model_builder.ModelBuilder``
   * - ``sagemaker.pytorch.PyTorchModel``
     - ``sagemaker.serve.model_builder.ModelBuilder``
   * - ``sagemaker.tensorflow.TensorFlowModel``
     - ``sagemaker.serve.model_builder.ModelBuilder``
   * - ``sagemaker.huggingface.HuggingFaceModel``
     - ``sagemaker.serve.model_builder.ModelBuilder``
   * - ``sagemaker.sklearn.SKLearnModel``
     - ``sagemaker.serve.model_builder.ModelBuilder``
   * - ``sagemaker.xgboost.XGBoostModel``
     - ``sagemaker.serve.model_builder.ModelBuilder``
   * - ``sagemaker.jumpstart.JumpStartModel``
     - ``ModelBuilder.from_jumpstart_config(JumpStartConfig(...))``
   * - ``sagemaker.predictor.Predictor``
     - ``sagemaker.core.resources.Endpoint``
   * - ``sagemaker.serializers.*``
     - Handle serialization directly (e.g., ``json.dumps()``)
   * - ``sagemaker.deserializers.*``
     - Handle deserialization directly (e.g., ``json.loads()``)


Methods and Patterns
~~~~~~~~~~~~~~~~~~~~~


.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - V2
     - V3
   * - ``model.deploy(instance_type=..., initial_instance_count=...)``
     - ``model_builder.deploy(endpoint_name=..., instance_type=..., initial_instance_count=...)``
   * - ``estimator.deploy()``
     - ``ModelBuilder(model=trainer).deploy()``
   * - ``predictor.predict(data)``
     - ``endpoint.invoke(body=data, content_type="application/json")``
   * - ``model = Model(image_uri=..., model_data=...)``
     - ``model_builder = ModelBuilder(model=..., model_path=...)``
   * - ``model.deploy()`` returns ``Predictor``
     - ``model_builder.deploy()`` returns ``Endpoint``
   * - ``Transformer(model_name=...).transform(...)``
     - ``sagemaker.core.resources.TransformJob.create(...)``


Session and Utilities
~~~~~~~~~~~~~~~~~~~~~~


.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - V2
     - V3
   * - ``sagemaker.session.Session()``
     - ``sagemaker.core.helper.session_helper.Session()``
   * - ``sagemaker.get_execution_role()``
     - ``sagemaker.core.helper.session_helper.get_execution_role()``
   * - ``sagemaker.image_uris.retrieve(...)``
     - ``sagemaker.core.image_uris.retrieve(...)``
   * - ``boto3.client('sagemaker')``
     - ``sagemaker.core.resources.*`` (Model, Endpoint, EndpointConfig, etc.)


V3 Package Structure
~~~~~~~~~~~~~~~~~~~~~


.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - V3 Package
     - Purpose
   * - ``sagemaker-core``
     - Low-level resource management (Model, Endpoint, EndpointConfig), session, image URIs
   * - ``sagemaker-train``
     - ModelTrainer for training (used with ``ModelBuilder(model=trainer)``)
   * - ``sagemaker-serve``
     - ModelBuilder, InferenceSpec, SchemaBuilder, ModelServer, deployment modes
   * - ``sagemaker-mlops``
     - Pipelines, processing, model registry, monitoring, Clarify


Inference Examples
-----------------

Explore comprehensive inference examples that demonstrate V3 capabilities:

.. toctree::
   :maxdepth: 1

   Custom InferenceSpec <../v3-examples/inference-examples/inference-spec-example>
   ModelBuilder with JumpStart models <../v3-examples/inference-examples/jumpstart-example>
   Optimize a JumpStart model <../v3-examples/inference-examples/optimize-example>
   Train-to-Inference E2E <../v3-examples/inference-examples/train-inference-e2e-example>
   JumpStart E2E <../v3-examples/inference-examples/jumpstart-e2e-training-example>
   Local Container Mode <../v3-examples/inference-examples/local-mode-example>
   Deploy HuggingFace Models <../v3-examples/inference-examples/huggingface-example>
   ModelBuilder in In-Process mode <../v3-examples/inference-examples/in-process-mode-example>
   Inference Pipeline <../v3-examples/inference-examples/inference-pipeline-modelbuilder-vs-core-example>
