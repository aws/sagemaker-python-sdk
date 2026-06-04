Model Training
===============

SageMaker Python SDK V3 provides a unified **ModelTrainer** class that replaces the framework-specific estimators from V2. This single class handles PyTorch, TensorFlow, Scikit-learn, XGBoost, and custom containers through a consistent interface.

Key Benefits of V3 Training
---------------------------

* **Unified Interface**: Single ``ModelTrainer`` class replaces multiple framework-specific estimators
* **Simplified Configuration**: Object-oriented API with auto-generated configs aligned with AWS APIs
* **Reduced Boilerplate**: Streamlined workflows with intuitive interfaces

Quick Start Example
-------------------

**SageMaker Python SDK V2:**

.. code-block:: python

   from sagemaker.estimator import Estimator

   estimator = Estimator(
       image_uri="my-training-image",
       role="arn:aws:iam::123456789012:role/SageMakerRole",
       instance_count=1,
       instance_type="ml.m5.xlarge",
       output_path="s3://my-bucket/output"
   )
   estimator.fit({"training": "s3://my-bucket/train"})

**SageMaker Python SDK V3:**

.. code-block:: python

   from sagemaker.train import ModelTrainer
   from sagemaker.train.configs import InputData

   trainer = ModelTrainer(
       training_image="my-training-image",
       role="arn:aws:iam::123456789012:role/SageMakerRole"
   )

   train_data = InputData(
       channel_name="training",
       data_source="s3://my-bucket/train"
   )

   trainer.train(input_data_config=[train_data])



Local Container Training
------------------------


Run training jobs in Docker containers on your local machine for rapid development and debugging before deploying to SageMaker cloud instances. Local mode requires Docker to be installed and running.

**Session Setup and Image Retrieval:**

.. code-block:: python

   from sagemaker.core.helper.session_helper import Session
   from sagemaker.core import image_uris

   sagemaker_session = Session()
   region = sagemaker_session.boto_region_name

   training_image = image_uris.retrieve(
       framework="pytorch",
       region=region,
       version="2.0.0",
       py_version="py310",
       instance_type="ml.m5.xlarge",
       image_scope="training"
   )

**Configuring Local Container Training:**

.. code-block:: python

   from sagemaker.train.model_trainer import ModelTrainer, Mode
   from sagemaker.train.configs import SourceCode, Compute, InputData

   source_code = SourceCode(
       source_dir="./source",
       entry_script="train.py",
   )

   compute = Compute(
       instance_type="local_cpu",
       instance_count=1,
   )

   train_data = InputData(
       channel_name="train",
       data_source="./data/train",
   )

   model_trainer = ModelTrainer(
       training_image=training_image,
       sagemaker_session=sagemaker_session,
       source_code=source_code,
       compute=compute,
       input_data_config=[train_data],
       base_job_name="local-training",
       training_mode=Mode.LOCAL_CONTAINER,
   )

   model_trainer.train()

Key points:

- Use ``instance_type="local_cpu"`` or ``"local_gpu"`` for local execution
- Set ``training_mode=Mode.LOCAL_CONTAINER`` to run in Docker
- Local data paths are mounted directly into the container
- Training artifacts are saved to the current working directory

:doc:`Full example notebook <../v3-examples/training-examples/local-training-example>`



Distributed Local Training
--------------------------


Test multi-node distributed training locally using multiple Docker containers before deploying to cloud. This uses the ``Torchrun`` distributed driver to coordinate training across containers.

**Configuring Distributed Local Training:**

.. code-block:: python

   from sagemaker.train.model_trainer import ModelTrainer, Mode
   from sagemaker.train.configs import SourceCode, Compute, InputData
   from sagemaker.train.distributed import Torchrun

   source_code = SourceCode(
       source_dir="./source",
       entry_script="train.py",
   )

   distributed = Torchrun(
       process_count_per_node=1,
   )

   compute = Compute(
       instance_type="local_cpu",
       instance_count=2,  # Two containers for distributed training
   )

   model_trainer = ModelTrainer(
       training_image=training_image,
       sagemaker_session=sagemaker_session,
       source_code=source_code,
       distributed=distributed,
       compute=compute,
       input_data_config=[train_data, test_data],
       base_job_name="distributed-local-training",
       training_mode=Mode.LOCAL_CONTAINER,
   )

   model_trainer.train()

Key points:

- ``instance_count=2`` launches two Docker containers
- ``Torchrun`` handles process coordination across containers
- ``process_count_per_node`` controls how many training processes run per container
- Temporary directories (``shared``, ``algo-1``, ``algo-2``) are cleaned up automatically after training

:doc:`Full example notebook <../v3-examples/training-examples/distributed-local-training-example>`



Hyperparameter Management
-------------------------


ModelTrainer supports loading hyperparameters from JSON files, YAML files, or Python dictionaries. File-based hyperparameters provide better version control and support for complex nested structures.

**Loading Hyperparameters from JSON:**

.. code-block:: python

   from sagemaker.train.model_trainer import ModelTrainer
   from sagemaker.train.configs import SourceCode

   source_code = SourceCode(
       source_dir="./source",
       requirements="requirements.txt",
       entry_script="train.py",
   )

   trainer = ModelTrainer(
       training_image=training_image,
       hyperparameters="hyperparameters.json",  # Path to JSON file
       source_code=source_code,
       base_job_name="hp-json-training",
   )

   trainer.train()

**Loading Hyperparameters from YAML:**

.. code-block:: python

   trainer = ModelTrainer(
       training_image=training_image,
       hyperparameters="hyperparameters.yaml",  # Path to YAML file
       source_code=source_code,
       base_job_name="hp-yaml-training",
   )

   trainer.train()

**Using a Python Dictionary:**

.. code-block:: python

   trainer = ModelTrainer(
       training_image=training_image,
       hyperparameters={
           "epochs": 10,
           "learning_rate": 0.001,
           "batch_size": 32,
           "model_config": {"hidden_size": 256, "num_layers": 3},
       },
       source_code=source_code,
       base_job_name="hp-dict-training",
   )

   trainer.train()

Key points:

- JSON and YAML files support complex nested structures (dicts, lists, booleans, floats)
- Hyperparameters are passed to the training script as command-line arguments
- They are also available via the ``SM_HPS`` environment variable as a JSON string
- All three approaches (JSON, YAML, dict) produce identical training behavior

:doc:`Full example notebook <../v3-examples/training-examples/hyperparameter-training-example>`



JumpStart Training
------------------


Train pre-configured models from the SageMaker JumpStart hub using ``ModelTrainer.from_jumpstart_config()``. JumpStart provides optimized training scripts, default hyperparameters, and curated datasets for hundreds of models.

**Training a HuggingFace BERT Model:**

.. code-block:: python

   from sagemaker.train.model_trainer import ModelTrainer
   from sagemaker.core.jumpstart import JumpStartConfig
   from sagemaker.core.helper.session_helper import Session, get_execution_role

   sagemaker_session = Session()
   role = get_execution_role()

   bert_config = JumpStartConfig(
       model_id="huggingface-spc-bert-base-cased",
   )

   bert_trainer = ModelTrainer.from_jumpstart_config(
       jumpstart_config=bert_config,
       base_job_name="jumpstart-bert",
       hyperparameters={
           "epochs": 1,
           "learning_rate": 5e-5,
           "train_batch_size": 32,
       },
       sagemaker_session=sagemaker_session,
   )

   bert_trainer.train()

**Training an XGBoost Classification Model:**

.. code-block:: python

   xgboost_config = JumpStartConfig(
       model_id="xgboost-classification-model",
   )

   xgboost_trainer = ModelTrainer.from_jumpstart_config(
       jumpstart_config=xgboost_config,
       base_job_name="jumpstart-xgboost",
       hyperparameters={
           "num_round": 10,
           "max_depth": 5,
           "eta": 0.2,
           "objective": "binary:logistic",
       },
       sagemaker_session=sagemaker_session,
   )

   xgboost_trainer.train()

**Discovering Available JumpStart Models:**

.. code-block:: python

   from sagemaker.core.jumpstart.notebook_utils import list_jumpstart_models
   from sagemaker.core.jumpstart.search import search_public_hub_models

   # List all available models
   models = list_jumpstart_models()

   # Filter by framework
   hf_models = list_jumpstart_models(filter="framework == huggingface")

   # Search with queries
   results = search_public_hub_models(query="bert")

   # Complex queries with filters
   text_gen = search_public_hub_models(query="@task:text-generation")

Key points:

- ``from_jumpstart_config()`` auto-configures training image, instance type, and default hyperparameters
- Override any default hyperparameters while keeping proven defaults for the rest
- JumpStart provides built-in datasets so you can start training immediately
- Supports HuggingFace, XGBoost, CatBoost, LightGBM, and many more frameworks
- Use ``list_jumpstart_models()`` and ``search_public_hub_models()`` to discover available models

:doc:`Full example notebook <../v3-examples/training-examples/jumpstart-training-example>`



Custom Distributed Training Drivers
------------------------------------


Create custom distributed training drivers by extending ``DistributedConfig`` for specialized coordination logic, framework integration, or advanced debugging.

**Defining a Custom Driver:**

.. code-block:: python

   from sagemaker.train.distributed import DistributedConfig

   class CustomDriver(DistributedConfig):
       process_count_per_node: int = None

       @property
       def driver_dir(self) -> str:
           return "./custom_drivers"

       @property
       def driver_script(self) -> str:
           return "driver.py"

The driver script (``driver.py``) receives environment variables including ``SM_DISTRIBUTED_CONFIG``, ``SM_HPS``, ``SM_SOURCE_DIR``, and ``SM_ENTRY_SCRIPT`` to coordinate training.

**Using the Custom Driver with ModelTrainer:**

.. code-block:: python

   from sagemaker.train.model_trainer import ModelTrainer
   from sagemaker.train.configs import SourceCode

   source_code = SourceCode(
       source_dir="./scripts",
       entry_script="entry_script.py",
   )

   custom_driver = CustomDriver(process_count_per_node=2)

   model_trainer = ModelTrainer(
       training_image=training_image,
       hyperparameters={"epochs": 10},
       source_code=source_code,
       distributed=custom_driver,
       base_job_name="custom-distributed",
   )

   model_trainer.train()

Key points:

- Extend ``DistributedConfig`` and implement ``driver_dir`` and ``driver_script`` properties
- The driver script manages process launching and coordination
- Environment variables provide access to hyperparameters, source code location, and distributed config
- Useful for custom frameworks, specialized coordination patterns, or advanced debugging

:doc:`Full example notebook <../v3-examples/training-examples/custom-distributed-training-example>`



AWS Batch Training Queues
-------------------------


Submit training jobs to AWS Batch job queues for automatic scheduling and resource management. Batch handles capacity allocation and job execution order.

**Setting Up and Submitting Jobs:**

.. code-block:: python

   from sagemaker.train.model_trainer import ModelTrainer
   from sagemaker.train.configs import SourceCode, Compute, StoppingCondition
   from sagemaker.train.aws_batch.training_queue import TrainingQueue

   source_code = SourceCode(command="echo 'Hello World'")

   model_trainer = ModelTrainer(
       training_image=image_uri,
       source_code=source_code,
       base_job_name="batch-training-job",
       compute=Compute(instance_type="ml.g5.xlarge", instance_count=1),
       stopping_condition=StoppingCondition(max_runtime_in_seconds=300),
   )

   # Create a queue reference and submit jobs
   queue = TrainingQueue("my-sm-training-fifo-jq")
   queued_job = queue.submit(training_job=model_trainer, inputs=None)

**Creating Batch Resources Programmatically:**

.. code-block:: python

   from sagemaker.train.aws_batch.boto_client import get_batch_boto_client
   from utils.aws_batch_resource_management import AwsBatchResourceManager, create_resources

   resource_manager = AwsBatchResourceManager(get_batch_boto_client())
   resources = create_resources(
       resource_manager,
       job_queue_name="my-sm-training-fifo-jq",
       service_environment_name="my-sm-training-fifo-se",
       max_capacity=1,
   )

Key points:

- ``TrainingQueue`` wraps AWS Batch job queues for SageMaker training
- ``queue.submit()`` submits a ModelTrainer job to the queue
- Batch manages capacity allocation and job scheduling automatically
- Resources (Service Environments, Job Queues) can be created via console or programmatically
- Supports FIFO and priority-based scheduling

:doc:`Full example notebook <../v3-examples/training-examples/aws_batch/sm-training-queues_getting_started_with_model_trainer>`


Migration from V2
------------------


Training Classes and Imports
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - V2
     - V3
   * - ``sagemaker.estimator.Estimator``
     - ``sagemaker.train.model_trainer.ModelTrainer``
   * - ``sagemaker.pytorch.PyTorch``
     - ``sagemaker.train.model_trainer.ModelTrainer``
   * - ``sagemaker.tensorflow.TensorFlow``
     - ``sagemaker.train.model_trainer.ModelTrainer``
   * - ``sagemaker.huggingface.HuggingFace``
     - ``sagemaker.train.model_trainer.ModelTrainer``
   * - ``sagemaker.sklearn.SKLearn``
     - ``sagemaker.train.model_trainer.ModelTrainer``
   * - ``sagemaker.xgboost.XGBoost``
     - ``sagemaker.train.model_trainer.ModelTrainer``
   * - ``sagemaker.jumpstart.JumpStartEstimator``
     - ``ModelTrainer.from_jumpstart_config(JumpStartConfig(...))``
   * - ``sagemaker.tuner.HyperparameterTuner``
     - ``sagemaker.core.resources.HyperParameterTuningJob``


Methods and Patterns
~~~~~~~~~~~~~~~~~~~~~


.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - V2
     - V3
   * - ``estimator.fit({"train": "s3://..."})``
     - ``trainer.train(input_data_config=[InputData(...)])``
   * - ``estimator.deploy()``
     - ``ModelBuilder(model=trainer).deploy()``
   * - ``instance_type="ml.m5.xlarge"``
     - ``Compute(instance_type="ml.m5.xlarge")``
   * - ``entry_point="train.py"``
     - ``SourceCode(entry_script="train.py")``
   * - ``source_dir="./src"``
     - ``SourceCode(source_dir="./src")``
   * - ``sagemaker.inputs.TrainingInput(s3_data=...)``
     - ``InputData(channel_name=..., data_source=...)``
   * - ``hyperparameters={"lr": 0.01}``
     - ``hyperparameters={"lr": 0.01}`` or ``hyperparameters="config.json"``
   * - ``max_run=3600``
     - ``StoppingCondition(max_runtime_in_seconds=3600)``


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
   * - ``import sagemaker`` (bare import)
     - Use explicit imports from subpackages
   * - ``boto3.client('sagemaker')``
     - ``sagemaker.core.resources.*`` (TrainingJob, Model, Endpoint, etc.)


V3 Package Structure
~~~~~~~~~~~~~~~~~~~~~


.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - V3 Package
     - Purpose
   * - ``sagemaker-core``
     - Low-level resource management, session, image URIs, lineage, JumpStart
   * - ``sagemaker-train``
     - ModelTrainer, Compute, SourceCode, InputData, distributed training
   * - ``sagemaker-serve``
     - ModelBuilder, InferenceSpec, SchemaBuilder, deployment
   * - ``sagemaker-mlops``
     - Pipelines, processing, model registry, monitoring, Clarify


Training Examples
-----------------

.. toctree::
   :maxdepth: 1

   Local Container mode <../v3-examples/training-examples/local-training-example>
   Distributed Local Training <../v3-examples/training-examples/distributed-local-training-example>
   Hyperparameter Training <../v3-examples/training-examples/hyperparameter-training-example>
   Training with JumpStart Models <../v3-examples/training-examples/jumpstart-training-example>
   Custom Distributed Training <../v3-examples/training-examples/custom-distributed-training-example>
   AWS Batch for Training <../v3-examples/training-examples/aws_batch/sm-training-queues_getting_started_with_model_trainer>
