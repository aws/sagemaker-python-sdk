Lineage Tracking
================

Amazon SageMaker Lineage enables events that happen within SageMaker to be traced via a graph structure. The data simplifies generating reports, making comparisons, or discovering relationships between events. For example, you can easily trace both how a model was generated and where the model was deployed.

The lineage graph is created automatically by SageMaker and you can directly create or modify your own graphs.

Key Concepts
------------

- **Lineage Graph** - A connected graph tracing your machine learning workflow end to end.
- **Artifacts** - Represents a URI addressable object or data. Artifacts are typically inputs or outputs to Actions.
- **Actions** - Represents an action taken such as a computation, transformation, or job.
- **Contexts** - Provides a method to logically group other entities.
- **Associations** - A directed edge in the lineage graph that links two entities.
- **Lineage Traversal** - Starting from an arbitrary point, trace the lineage graph to discover and analyze relationships between steps in your workflow.
- **Experiments** - Experiment entities (Experiments, Trials, and Trial Components) are also part of the lineage graph and can be associated with Artifacts, Actions, or Contexts.

Use Cases
---------

The notebook from the `SageMaker examples repository <https://github.com/aws/amazon-sagemaker-examples>`_ demonstrates the following major use cases for lineage tracking:

1. **Creating Lineage Contexts** - Group related lineage entities under a logical workflow context
2. **Listing Lineage Entities** - Query and enumerate existing contexts, actions, artifacts, and associations
3. **Creating Actions** - Record computational steps such as model builds, transformations, or training jobs
4. **Creating Artifacts** - Register data inputs (datasets, labels) and outputs (trained models) as lineage artifacts
5. **Creating Associations** - Link entities together with directed edges to form the lineage graph
6. **Traversing Associations** - Query incoming and outgoing associations to understand entity relationships
7. **Cleaning Up Lineage Data** - Delete associations and entities when they are no longer needed

V3 Migration Notes
------------------

In SageMaker Python SDK V3, lineage classes have moved from ``sagemaker.lineage`` to ``sagemaker.core.lineage``. The old import paths still work via compatibility shims but emit deprecation warnings.

.. list-table:: Import Path Changes
   :header-rows: 1
   :widths: 50 50

   * - V2 Import
     - V3 Import
   * - ``from sagemaker.lineage.context import Context``
     - ``from sagemaker.core.lineage.context import Context``
   * - ``from sagemaker.lineage.action import Action``
     - ``from sagemaker.core.lineage.action import Action``
   * - ``from sagemaker.lineage.artifact import Artifact``
     - ``from sagemaker.core.lineage.artifact import Artifact``
   * - ``from sagemaker.lineage.association import Association``
     - ``from sagemaker.core.lineage.association import Association``
   * - ``import sagemaker`` / ``sagemaker.session.Session()``
     - ``from sagemaker.core.helper.session_helper import Session``

The API signatures for ``create``, ``list``, ``delete``, and association management remain the same. The key change is the import path.


Use Case 1: Session Setup
-------------------------

Initialize a SageMaker session and set up common variables.

**V2 (Legacy):**

.. code-block:: python

   import boto3
   import sagemaker

   region = boto3.Session().region_name
   sagemaker_session = sagemaker.session.Session()
   default_bucket = sagemaker_session.default_bucket()

**V3:**

.. code-block:: python

   import boto3
   from sagemaker.core.helper.session_helper import Session

   region = boto3.Session().region_name
   sagemaker_session = Session()
   default_bucket = sagemaker_session.default_bucket()


Use Case 2: Creating a Lineage Context
---------------------------------------

Contexts provide a method to logically group other lineage entities. Each context name must be unique across all other contexts.

**V2 (Legacy):**

.. code-block:: python

   from datetime import datetime
   from sagemaker.lineage.context import Context

   unique_id = str(int(datetime.now().replace(microsecond=0).timestamp()))
   context_name = f"machine-learning-workflow-{unique_id}"

   ml_workflow_context = Context.create(
       context_name=context_name,
       context_type="MLWorkflow",
       source_uri=unique_id,
       properties={"example": "true"},
   )

**V3:**

.. code-block:: python

   from datetime import datetime
   from sagemaker.core.lineage.context import Context

   unique_id = str(int(datetime.now().replace(microsecond=0).timestamp()))
   context_name = f"machine-learning-workflow-{unique_id}"

   ml_workflow_context = Context.create(
       context_name=context_name,
       context_type="MLWorkflow",
       source_uri=unique_id,
       properties={"example": "true"},
   )


Use Case 3: Listing Contexts
-----------------------------

Enumerate existing contexts sorted by creation time.

**V2 (Legacy):**

.. code-block:: python

   from sagemaker.lineage.context import Context

   contexts = Context.list(sort_by="CreationTime", sort_order="Descending")
   for ctx in contexts:
       print(ctx.context_name)

**V3:**

.. code-block:: python

   from sagemaker.core.lineage.context import Context

   contexts = Context.list(sort_by="CreationTime", sort_order="Descending")
   for ctx in contexts:
       print(ctx.context_name)


Use Case 4: Creating an Action
-------------------------------

Actions represent computational steps such as model builds, transformations, or training jobs.

**V2 (Legacy):**

.. code-block:: python

   from sagemaker.lineage.action import Action

   model_build_action = Action.create(
       action_name=f"model-build-step-{unique_id}",
       action_type="ModelBuild",
       source_uri=unique_id,
       properties={"Example": "Metadata"},
   )

**V3:**

.. code-block:: python

   from sagemaker.core.lineage.action import Action

   model_build_action = Action.create(
       action_name=f"model-build-step-{unique_id}",
       action_type="ModelBuild",
       source_uri=unique_id,
       properties={"Example": "Metadata"},
   )


Use Case 5: Creating Associations
-----------------------------------

Associations are directed edges in the lineage graph. The ``association_type`` can be ``Produced``, ``DerivedFrom``, ``AssociatedWith``, or ``ContributedTo``.

**V2 (Legacy):**

.. code-block:: python

   from sagemaker.lineage.association import Association

   context_action_association = Association.create(
       source_arn=ml_workflow_context.context_arn,
       destination_arn=model_build_action.action_arn,
       association_type="AssociatedWith",
   )

**V3:**

.. code-block:: python

   from sagemaker.core.lineage.association import Association

   context_action_association = Association.create(
       source_arn=ml_workflow_context.context_arn,
       destination_arn=model_build_action.action_arn,
       association_type="AssociatedWith",
   )



Use Case 6: Traversing Associations
-------------------------------------

Query incoming and outgoing associations to understand how entities are related in the lineage graph.

**V2 (Legacy):**

.. code-block:: python

   from sagemaker.lineage.association import Association

   # List incoming associations to an action
   incoming = Association.list(destination_arn=model_build_action.action_arn)
   for association in incoming:
       print(f"{model_build_action.action_name} has incoming association from {association.source_name}")

   # List outgoing associations from a context
   outgoing = Association.list(source_arn=ml_workflow_context.context_arn)
   for association in outgoing:
       print(f"{ml_workflow_context.context_name} has outgoing association to {association.destination_name}")

**V3:**

.. code-block:: python

   from sagemaker.core.lineage.association import Association

   # List incoming associations to an action
   incoming = Association.list(destination_arn=model_build_action.action_arn)
   for association in incoming:
       print(f"{model_build_action.action_name} has incoming association from {association.source_name}")

   # List outgoing associations from a context
   outgoing = Association.list(source_arn=ml_workflow_context.context_arn)
   for association in outgoing:
       print(f"{ml_workflow_context.context_name} has outgoing association to {association.destination_name}")


Use Case 7: Creating Artifacts
-------------------------------

Artifacts represent URI-addressable objects or data, such as datasets, labels, or trained models.

**V2 (Legacy):**

.. code-block:: python

   from sagemaker.lineage.artifact import Artifact

   input_test_images = Artifact.create(
       artifact_name="mnist-test-images",
       artifact_type="TestData",
       source_types=[{"SourceIdType": "Custom", "Value": unique_id}],
       source_uri=f"https://sagemaker-example-files-prod-{region}.s3.amazonaws.com/datasets/image/MNIST/t10k-images-idx3-ubyte.gz",
   )

   input_test_labels = Artifact.create(
       artifact_name="mnist-test-labels",
       artifact_type="TestLabels",
       source_types=[{"SourceIdType": "Custom", "Value": unique_id}],
       source_uri=f"https://sagemaker-example-files-prod-{region}.s3.amazonaws.com/datasets/image/MNIST/t10k-labels-idx1-ubyte.gz",
   )

   output_model = Artifact.create(
       artifact_name="mnist-model",
       artifact_type="Model",
       source_types=[{"SourceIdType": "Custom", "Value": unique_id}],
       source_uri=f"https://sagemaker-example-files-prod-{region}.s3.amazonaws.com/datasets/image/MNIST/model/tensorflow-training-2020-11-20-23-57-13-077/model.tar.gz",
   )

**V3:**

.. code-block:: python

   from sagemaker.core.lineage.artifact import Artifact

   input_test_images = Artifact.create(
       artifact_name="mnist-test-images",
       artifact_type="TestData",
       source_types=[{"SourceIdType": "Custom", "Value": unique_id}],
       source_uri=f"https://sagemaker-example-files-prod-{region}.s3.amazonaws.com/datasets/image/MNIST/t10k-images-idx3-ubyte.gz",
   )

   input_test_labels = Artifact.create(
       artifact_name="mnist-test-labels",
       artifact_type="TestLabels",
       source_types=[{"SourceIdType": "Custom", "Value": unique_id}],
       source_uri=f"https://sagemaker-example-files-prod-{region}.s3.amazonaws.com/datasets/image/MNIST/t10k-labels-idx1-ubyte.gz",
   )

   output_model = Artifact.create(
       artifact_name="mnist-model",
       artifact_type="Model",
       source_types=[{"SourceIdType": "Custom", "Value": unique_id}],
       source_uri=f"https://sagemaker-example-files-prod-{region}.s3.amazonaws.com/datasets/image/MNIST/model/tensorflow-training-2020-11-20-23-57-13-077/model.tar.gz",
   )


Use Case 8: Linking Artifacts to Actions
------------------------------------------

Associate data artifacts as inputs to an action, and the action's output to a model artifact, forming a complete lineage chain.

**V2 (Legacy):**

.. code-block:: python

   from sagemaker.lineage.association import Association

   # Link input data to the model build action
   Association.create(
       source_arn=input_test_images.artifact_arn,
       destination_arn=model_build_action.action_arn,
   )
   Association.create(
       source_arn=input_test_labels.artifact_arn,
       destination_arn=model_build_action.action_arn,
   )

   # Link the action output to the model artifact
   Association.create(
       source_arn=model_build_action.action_arn,
       destination_arn=output_model.artifact_arn,
   )

**V3:**

.. code-block:: python

   from sagemaker.core.lineage.association import Association

   # Link input data to the model build action
   Association.create(
       source_arn=input_test_images.artifact_arn,
       destination_arn=model_build_action.action_arn,
   )
   Association.create(
       source_arn=input_test_labels.artifact_arn,
       destination_arn=model_build_action.action_arn,
   )

   # Link the action output to the model artifact
   Association.create(
       source_arn=model_build_action.action_arn,
       destination_arn=output_model.artifact_arn,
   )


Use Case 9: Cleaning Up Lineage Data
--------------------------------------

Delete associations first, then delete the entities themselves. Associations must be removed before their source or destination entities can be deleted.

**V2 (Legacy):**

.. code-block:: python

   import sagemaker
   from sagemaker.lineage.association import Association
   from sagemaker.lineage.context import Context
   from sagemaker.lineage.action import Action
   from sagemaker.lineage.artifact import Artifact

   sagemaker_session = sagemaker.session.Session()

   def delete_associations(arn):
       for summary in Association.list(destination_arn=arn):
           assct = Association(
               source_arn=summary.source_arn,
               destination_arn=summary.destination_arn,
               sagemaker_session=sagemaker_session,
           )
           assct.delete()
       for summary in Association.list(source_arn=arn):
           assct = Association(
               source_arn=summary.source_arn,
               destination_arn=summary.destination_arn,
               sagemaker_session=sagemaker_session,
           )
           assct.delete()

   # Delete context
   delete_associations(ml_workflow_context.context_arn)
   Context(context_name=ml_workflow_context.context_name, sagemaker_session=sagemaker_session).delete()

   # Delete action
   delete_associations(model_build_action.action_arn)
   Action(action_name=model_build_action.action_name, sagemaker_session=sagemaker_session).delete()

   # Delete artifacts
   for artifact in [input_test_images, input_test_labels, output_model]:
       delete_associations(artifact.artifact_arn)
       Artifact(artifact_arn=artifact.artifact_arn, sagemaker_session=sagemaker_session).delete()

**V3:**

.. code-block:: python

   from sagemaker.core.helper.session_helper import Session
   from sagemaker.core.lineage.association import Association
   from sagemaker.core.lineage.context import Context
   from sagemaker.core.lineage.action import Action
   from sagemaker.core.lineage.artifact import Artifact

   sagemaker_session = Session()

   def delete_associations(arn):
       for summary in Association.list(destination_arn=arn):
           assct = Association(
               source_arn=summary.source_arn,
               destination_arn=summary.destination_arn,
               sagemaker_session=sagemaker_session,
           )
           assct.delete()
       for summary in Association.list(source_arn=arn):
           assct = Association(
               source_arn=summary.source_arn,
               destination_arn=summary.destination_arn,
               sagemaker_session=sagemaker_session,
           )
           assct.delete()

   # Delete context
   delete_associations(ml_workflow_context.context_arn)
   Context(context_name=ml_workflow_context.context_name, sagemaker_session=sagemaker_session).delete()

   # Delete action
   delete_associations(model_build_action.action_arn)
   Action(action_name=model_build_action.action_name, sagemaker_session=sagemaker_session).delete()

   # Delete artifacts
   for artifact in [input_test_images, input_test_labels, output_model]:
       delete_associations(artifact.artifact_arn)
       Artifact(artifact_arn=artifact.artifact_arn, sagemaker_session=sagemaker_session).delete()


Caveats
-------

- Associations cannot be created between two experiment entities (e.g., between an Experiment and Trial).
- Associations can only be created between Action, Artifact, or Context resources.
- Maximum number of manually created lineage entities:

  - Artifacts: 6000
  - Contexts: 500
  - Actions: 3000
  - Associations: 6000

- There is no limit on the number of lineage entities created automatically by SageMaker.


Lineage Tracking Example
-------------------------

For a complete end-to-end V3 example, see the lineage tracking notebook:

.. toctree::
   :maxdepth: 1

   ../v3-examples/ml-ops-examples/v3-lineage-tracking-example
