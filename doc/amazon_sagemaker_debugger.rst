
#########################
Amazon SageMaker Debugger
#########################


.. warning::

  This page is no longer supported for maintenence. The live documentation is at `Debug and Profile Training Jobs Using Amazon SageMaker Debugger <https://docs.aws.amazon.com/sagemaker/latest/dg/train-debugger.html>`_
  and `Debugger API <https://sagemaker.readthedocs.io/en/stable/api/training/debugger.html>`_.


Amazon SageMaker Debugger allows you to detect anomalies while training your machine learning model by emitting relevant data during training, storing the data and then analyzing it.

.. contents::

Background
==========

Amazon SageMaker provides every developer and data scientist with the ability to build, train, and deploy machine learning models quickly. Amazon SageMaker is a fully-managed service that encompasses the entire machine learning workflow. You can label and prepare your data, choose an algorithm, train a model, and then tune and optimize it for deployment. You can deploy your models to production with Amazon SageMaker to make predictions at lower costs than was previously possible.

`SageMaker Debugger <https://docs.aws.amazon.com/sagemaker/latest/dg/train-debugger.html>`__ provides a way to hook into the training process and emit debug artifacts (a.k.a. "tensors") that represent the training state at each point in the training lifecycle. Debugger then stores the data in real time and uses rules that encapsulate logic to analyze tensors and react to anomalies. Debugger provides built-in rules and allows you to write custom rules for analysis.

Setup
=====

To get started, you must satisfy the following prerequisites:

* Specify an AWS Region where you'll train your model.
* Give Amazon SageMaker the access to your data in Amazon Simple Storage Service (Amazon S3) needed to train your model by creating an IAM role ARN. See the `AWS IAM documentation <https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles.html>`__ for how to fine tune the permissions needed.

Capture real-time debugging data during model training in Amazon SageMaker
==========================================================================

To enable data emission for debugging model training, Amazon SageMaker initializes a "hook" which attaches itself to the training process and emits data necessary for debugging, i.e. tensors. To provide the hook's configuration, specify the option called ``DebuggerHookConfig`` when training your model. For more about the ``DebuggerHookConfig`` object, see the `API documentation <https://sagemaker.readthedocs.io/en/stable/debugger.html#sagemaker.debugger.DebuggerHookConfig>`__.

The ``DebuggerHookConfig`` accepts one or more objects of type ``CollectionConfig``, which defines the configuration around the tensor collection you intend to emit and save during model training. The concept of a "collection" helps group tensors for easier handling.

.. code:: python

    from sagemaker.debugger import CollectionConfig, DebuggerHookConfig

    collection_config = CollectionConfig(
        name='collection_name',
        parameters={
            'key': 'value'
        }
    )

    debugger_hook_config = DebuggerHookConfig(
        s3_output_path='s3://path/for/data/emission',
        container_local_output_path='/local/path/for/data/emission',
        hook_parameters={
            'key': 'value'
        },
        collection_configs=[
            collection_config
        ]
    )

    estimator = TensorFlow(
        role=role,
        instance_count=1,
        instance_type=instance_type,
        debugger_hook_config=debugger_hook_config
    )

Specifying configurations for collections
-----------------------------------------

Collection Name
~~~~~~~~~~~~~~~

``name`` in ``CollectionConfig`` is used to specify the name of the tensor collection you wish to emit and store. This name is used by SageMaker Debugger to refer to all the tensors in this collection. You can supply any valid string for the collection name. In addition, there are "built-in" collections, whose names are recognized by the hook, that you can emit simply by specifying their names. Examples of these collections are "gradients", "weights", "biases", etc. A full list is available at `SageMaker Debugger Built-in Collections <https://github.com/awslabs/sagemaker-debugger/blob/master/docs/api.md#built-in-collections>`__.

To emit and store one of the built-in collections:

.. code:: python

    collection_config_biases = CollectionConfig(name='biases')

Collection Parameters
~~~~~~~~~~~~~~~~~~~~~

To specify additional configuration for a particular collection, use ``parameters`` in the ``CollectionConfig``. This parameter provides a mapping that defines what group of tensors are saved and how frequently they are to be saved.

For instance, suppose you want to save a collection of tensors with the following properties:

========================================================= =========
**Desired Property**                                      **Value**
--------------------------------------------------------- ---------
regex of tensors which should be saved                    ``relu``
step frequency at which the said tensors should be saved  20
starting at step                                          5
ending at step                                            100
========================================================= =========

You should configure the ``CollectionConfig`` as:

.. code:: python

    collection_config_for_relu = CollectionConfig(
        name='custom_relu_collection',
        parameters={
            'include_regex': 'relu',
            'save_interval': '20',
            'start_step': '5',
            'end_step': '100'
        }
    }

The possible values of ``parameters`` in ``CollectionConfig`` can be viewed at `CollectionParameters <https://docs.aws.amazon.com/sagemaker/latest/dg/API_CollectionConfiguration.html#SageMaker-Type-CollectionConfiguration-CollectionParameters>`__.

Hook Parameters
~~~~~~~~~~~~~~~

To apply properties across all collections, use ``hook_parameters`` within the ``DebuggerHookConfig`` object. For example, to apply a value of ``10`` for ``save_interval`` across all collections:

.. code:: python

    from sagemaker.debugger import CollectionConfig, DebuggerHookConfig

    collection_config_1 = CollectionConfig(
        name='collection_name_1',
        parameters={
            'include_regex': '.*'
        }
    )
    collection_config_2 = CollectionConfig(
        name='collection_name_2',
        parameters={
            'include_regex': '.*'
        }
    }

    debugger_hook_config = DebuggerHookConfig(
        s3_output_path='s3://path/for/data/emission',
        container_local_output_path='/local/path/for/data/emission',
        hook_parameters={
            'save_interval': '10'
        },
        collection_configs=[
            collection_config_1, collection_config_2
        ]
    )

In the above sample code, the ``save_interval`` of ``10`` will be applied for storing both collections.

Note that the ``save_interval`` value set in the ``collection_parameters`` will override the value for ``save_interval`` in the ``hook_parameters``. For example, in the above sample code, if ``collection_config_2`` had a ``save_interval`` value set to ``20``, then the tensors for that collection would be saved with step interval ``20`` while those for ``collection_config_1`` would still be saved with a step interval of ``10``. This holds true for any parameters common in ``hook_parameters`` and ``parameters`` in ``CollectionConfig``.

The possible values of ``hook_parameters`` in ``DebuggerHookConfig`` can be viewed at `SageMaker Debugger Hook <https://github.com/awslabs/sagemaker-debugger/blob/master/docs/api.md#creating-a-hook>`__.

Begin model training
--------------------

To create a training job that initializes the debugging hook with the value of the ``DebuggerHookConfig`` object, call ``fit()`` on the ``estimator``. The hook starts emitting the relevant debugging data, i.e. the tensor collections, in real time and stores the data locally in the local path provided in ``DebuggerHookConfig``. This data is then uploaded in near real time to an S3 path derived from the path provided in the hook configuration.

.. code::

    s3://{destination-bucket-prefix}/{training-job-name}/debug-output/

The path is derived from the value of ``s3_output_path``, and not used verbatim, to ensure that artifacts from different training jobs are placed in different Amazon S3 paths. To enable correct analyses of different training jobs, it is essential to keep the debug artifacts from these jobs separate.

To access the above Amazon S3 path through the estimator object, you can use the following command:

.. code:: python

    tensors_s3_output_path = estimator.latest_job_debugger_artifacts_path()

You can use the ``S3Downloader`` utility to view and download the debugging data emitted during training in Amazon S3. (Note that data is stored in a streaming fashion so the data you download locally through ``S3Downloader`` will be a snapshot of the data generated until that time.) Here is the code:

.. code:: python

    from sagemaker.s3 import S3Downloader

    # Start the training by calling fit
    # Setting the wait to `False` would make the fit asynchronous
    estimator.fit(wait=False)

    # Get a list of S3 URIs
    S3Downloader.list(estimator.latest_job_debugger_artifacts_path())

Continuous analyses through rules
=================================

In addition to collecting the debugging data, Amazon SageMaker Debugger provides the capability for you to analyze it in a streaming fashion using "rules". A SageMaker Debugger "rule" is a piece of code which encapsulates the logic for analyzing debugging data.

SageMaker Debugger provides a set of built-in rules curated by data scientists and engineers at Amazon to identify common problems while training machine learning models. There is also support for using custom rule source codes for evaluation. In the following sections, you'll learn how to use both the built-in and custom rules while training your model.

Relationship between debugger hook and rules
--------------------------------------------

Using SageMaker Debugger is, broadly, a two-pronged approach. On one hand you have the production of debugging data, which is done through the Debugger Hook, and on the other hand you have the consumption of this data, which can be with rules (for continuous analyses) or by using the SageMaker Debugger SDK (for interactive analyses).

The production and consumption of data are defined independently. For example, you could configure the debugging hook to store only the collection "gradients" and then configure the rules to operate on some other collection, say, "weights". While this is possible, it's quite useless as it gives you no meaningful insight into the training process. This is because the rule will do nothing in this example scenario since it will wait for the tensors in the collection "gradients" which are never be emitted.

For more useful and efficient debugging, configure your debugging hook to produce and store the debugging data that you care about and employ rules that operate on that particular data. This way, you ensure that the Debugger is utilized to its maximum potential in detecting anomalies. In this sense, there is a loose binding between the hook and the rules.

Normally, you'd achieve this binding for a training job by providing values for both ``debugger_hook_config`` and ``rules`` in your estimator. However, SageMaker Debugger simplifies this by allowing you to specify the collection configuration within the ``Rule`` object itself. This way, you don't have to specify the ``debugger_hook_config`` in your estimator separately.

Using built-in rules
--------------------

SageMaker Debugger comes with a set of built-in rules which can be used to identify common problems in model training, for example vanishing gradients or exploding tensors. You can choose to evaluate one or more of these rules while training your model to obtain meaningful insight into the training process. To learn more about these built in rules, see `SageMaker Debugger Built-in Rules <https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-built-in-rules.html>`__.

Pre-defined debugger hook configuration for built-in rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
As mentioned earlier, for efficient analyses, it's important that the debugging data that is emitted by the hook is relevant to the rules used to operate and analyze the data. For example, if the hook is configured to emit the collection "weights", you should evaluate a rule that operates on this collection and not some other collection.

Determining the types of data to emit for debugging with the built-in rules during the model training can be tricky. To guide you in this choice, Amazon SageMaker provides you with predefined collection configurations best suited for each of the built-in rules. So when you use built-in Debugger rules, you just need to specify the names of the built-in rule and SageMaker Debugger configures the collection(s) to emit that the rules need to operate on. To learn more about the mapping of each rule to the appropriate collection configuration, see `Amazon SageMaker Debugger Rules Config <https://github.com/awslabs/sagemaker-debugger-rulesconfig>`__.

Sample Usages
~~~~~~~~~~~~~

**Example 1**: Using a single built-in rule without any customization.

.. code:: python

    from sagemaker.debugger import Rule
    from smdebug_rulesconfig import vanishing_gradient

    estimator = TensorFlow(
            role=role,
            instance_count=1,
            instance_type=instance_type,
            rules=[Rule.sagemaker(vanishing_gradient())]
    )


In the example above, Amazon SageMaker pulls the collection configuration best suited for the built-in rule Vanishing Gradient from `SageMaker Debugger Rules Config <https://github.com/awslabs/sagemaker-debugger-rulesconfig>`__ and configures the debugging data to be stored in the manner specified in the configuration.

**Example 2**: Using more than one built-in rules without any customization.

.. code:: python

    from sagemaker.debugger import Rule
    from smdebug_rulesconfig import vanishing_gradient, weight_update_ratio

    estimator = TensorFlow(
            role=role,
            instance_count=1,
            instance_type=instance_type,
            rules=[Rule.sagemaker(vanishing_gradient()), Rule.sagemaker(weight_update_ratio())]
    )

In the example above, Amazon SageMaker pulls the hook configurations for Vanishing Gradient and Weight Update Ratio rules from `SageMaker Debugger Rules Config <https://github.com/awslabs/sagemaker-debugger-rulesconfig>`__  and configures the collections to be stored in the manner specified in each configuration.

**Example 3**: Using a built-in rule with no customization and another built-in rule with customization.

Here we modify the ``weight_update_ratio`` rule to store a custom collection rather than "weights" which it would normally do if the behavior is not overridden.


.. code:: python

    from sagemaker.debugger import Rule
    from smdebug_rulesconfig import vanishing_gradient, weight_update_ratio

    wur_with_customization = Rule.sagemaker(
        base_config=weight_update_ratio(),
        name="custom_wup_rule_name",
        rule_parameters={
            'key1': 'value1',
            'key2': 'value2'
        },
        collections_to_save=[
            CollectionConfig(
                name="custom_collection_name",
                parameters= {
                    'key1': 'value1',
                    'key2': 'value2'
                }
            )
        ]
    )

    estimator = TensorFlow(
            role=role,
            instance_count=1,
            instance_type=instance_type,
            rules=[
                Rule.sagemaker(vanishing_gradient()),
                wur_with_customization
            ]
    )


In the example above, the collection configuration for Vanishing Gradient is pulled from `SageMaker Debugger Rules Config <https://github.com/awslabs/sagemaker-debugger-rulesconfig>`__  and the user supplied configuration is used for the Weight Update Ratio rule.

Using custom rules
------------------

SageMaker Debugger also allows the users to create custom rules and have those evaluated against the debugging data. To use custom rules, you must provide two items:

* Custom rule source file and its local or S3 location. You can learn more about how to write custom rules at `How to Write Custom Debugger Rules <https://github.com/awslabs/sagemaker-debugger/blob/master/docs/analysis.md#writing-a-custom-rule>`__
* Rule evaluator image for the corresponding region available from `Amazon SageMaker Debugger Custom Rule Images <https://docs.aws.amazon.com/sagemaker/latest/dg/debuger-custom-rule-registry-ids.html>`__

To learn more about how to write your custom rules and use them see `SageMaker Debugger Custom Rules <https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-custom-rules.html>`__.

Sample Usage
~~~~~~~~~~~~

For this example, we evaluate an altered version of the Vanishing Gradient rule against our model training. The rule checks the gradients and asserts that the mean value of the gradients at any step is always above a certain threshold. The source code for the rule is available `here <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-debugger/tensorflow_keras_custom_rule/rules/my_custom_rule.py>`__ and is assumed to be in the relative directory path ``rules/custom_gradient_rule.py``.

To evaluate the custom rule against the training:

.. code:: python

    from sagemaker.debugger import Rule

    region = 'us-east-1' # the AWS region of the training job
    custom_gradient_rule = Rule.custom(
        name='MyCustomRule',
        image_uri='864354269164.dkr.ecr.{}.amazonaws.com/sagemaker-debugger-rule-evaluator:latest'.format(region),
        instance_type='ml.t3.medium', # instance type to run the rule evaluation on
        source='rules/custom_gradient_rule.py', # path to the rule source file
        rule_to_invoke='CustomGradientRule', # name of the class to invoke in the rule source file
        volume_size_in_gb=30, # EBS volume size required to be attached to the rule evaluation instance
        collections_to_save=[CollectionConfig("gradients")], # collections to be analyzed by the rule
        rule_parameters={
          'threshold': '20.0' # this will be used to initialize 'threshold' param in your rule constructor
        }
    )

    estimator = TensorFlow(
        role=role,
        instance_count=1,
        instance_type=instance_type,
        rules=[
            custom_gradient_rule
        ]
    )

While initializing the custom rule through ``Rules.custom()``, you must specify a valid S3 location for rule source location as the value of ``source``.

Capture real-time TensorBoard data from the debugging hook
==========================================================

In addition to emitting and storing the debugging data useful for analyses, the debugging hook is also capable of emitting `TensorBoard <https://www.tensorflow.org/tensorboard>`__ data for you to point your TensorBoard application at and to visualize.

To enable the debugging hook to emit TensorBoard data, you need to specify the new option ``TensorBoardOutputConfig`` as follows:

.. code:: python

    from sagemaker.debugger import TensorBoardOutputConfig

    tensorboard_output_config = TensorBoardOutputConfig(
        s3_output_path='s3://path/for/tensorboard/data/emission',
        container_local_output_path='/local/path/for/tensorboard/data/emission'
    )

    estimator = TensorFlow(
        role=role,
        instance_count=1,
        instance_type=instance_type,
        tensorboard_output_config=tensorboard_output_config
    )

To create a training job where the debugging hook emits and stores TensorBoard data using the configuration specified in the ``TensorBoardOutputConfig`` object, call ``fit()`` on the ``estimator``. The debugging hook uploads the generated TensorBoard data in near real-time to an S3 path derived from the value of ``s3_output_path`` provided in the configuration:

.. code::

    s3://{destination-bucket-prefix}/{training-job-name}/tensorboard-output/

To access the S3 path where the tensorboard data is stored, you can do:

.. code:: python

    tensorboard_s3_output_path = estimator.latest_job_tensorboard_artifacts_path()

The reason for deriving the path from the value supplied to ``s3_output_path`` is the same as that provided for ``DebuggerHookConfig`` case - the directory for TensorBoard artifact storage needs be different for each training job.

Note that having the TensorBoard data emitted from the hook in addition to the tensors will incur a cost to the training and may slow it down.

Interactive analysis using SageMaker Debugger SDK and visualizations
====================================================================

`Amazon SageMaker Debugger SDK <https://github.com/awslabs/sagemaker-debugger>`__ also allows you to do interactive analyses on the debugging data produced from a training job run and to render visualizations of it. After calling ``fit()`` on the estimator, you can use the SDK to load the saved data in a SageMaker Debugger ``trial`` and do an analysis on the data:

.. code:: python

    from smdebug.trials import create_trial

    s3_output_path = estimator.latest_job_debugger_artifacts_path()
    trial = create_trial(s3_output_path)

To learn more about the programming model for analysis using the SageMaker Debugger SDK, see `SageMaker Debugger Analysis <https://github.com/awslabs/sagemaker-debugger/blob/master/docs/analysis.md>`__.

For a tutorial on what you can do after creating the trial and how to visualize the results, see `SageMaker Debugger - Visualizing Debugging Results <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-debugger/mnist_tensor_plot/mnist-tensor-plot.ipynb>`__.

Default behavior and opting out
===============================

For ``TensorFlow``, ``Keras``, ``MXNet``, ``PyTorch`` and ``XGBoost`` estimators, the ``DebuggerHookConfig`` is always initialized regardless of specification while initializing the estimator. This is done to minimize code changes needed to get useful debugging information.

To disable the hook initialization, you can do so by specifying ``False`` for value of ``debugger_hook_config`` in your framework estimator's initialization:

.. code:: python

    estimator = TensorFlow(
        role=role,
        instance_count=1,
        instance_type=instance_type,
        debugger_hook_config=False
    )

Learn More
==========

Further documentation
---------------------

* API documentation: https://sagemaker.readthedocs.io/en/stable/debugger.html
* AWS documentation: https://docs.aws.amazon.com/sagemaker/latest/dg/train-debugger.html
* SageMaker Debugger SDK: https://github.com/awslabs/sagemaker-debugger
* ``S3Downloader``: https://sagemaker.readthedocs.io/en/stable/s3.html#sagemaker.s3.S3Downloader

Notebook examples
-----------------

Consult our notebook examples for in-depth tutorials: https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-debugger
