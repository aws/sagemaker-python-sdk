#########################################
Amazon SageMaker Model Building Pipeline
#########################################


.. contents::

==========
Background
==========

Amazon SageMaker Model Building Pipelines is the first purpose-built, easy-to-use continuous integration and continuous delivery (CI/CD) service for machine learning (ML). With SageMaker Pipelines, you can create, automate, and manage end-to-end ML workflows at scale.

A pipeline is a series of interconnected steps that is defined by a JSON pipeline definition. The SageMaker Python SDK offers convenient abstractions to help construct a pipeline with ease. The following pages include code examples for some of the most common SageMaker Pipelines use cases.

For a higher-level conceptual guide and more information on integration with additional AWS services and features, see :`Amazon SageMaker Model Building Pipelines`_ in the Amazon SageMaker Developer Guide.

.. _Amazon SageMaker Model Building Pipelines: https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html


================
Code Examples
================

Pipeline Session
==================
Pipeline Session helps manage AWS service integrations during pipeline creation.

Pipeline Session is an extension of SageMaker Session, which manages the interactions between SageMaker APIs and AWS services like Amazon S3. SageMaker Session provides convenient methods for manipulating entities and resources that Amazon SageMaker uses, such as training jobs, endpoints, and S3 input datasets. AWS service calls are delegated to an underlying Boto3 session, which by default is initialized using the AWS configuration chain.

The following example shows how to construct an estimator and start a :code:`TrainingJob` with SageMaker Session:

.. code-block:: python

    pytorch_estimator = PyTorch(
        *sagemaker_session=sagemaker.Session(),*
        role=sagemaker.get_execution_role(),
        instance_type="ml.c5.xlarge",
        instance_count=1,
        framework_version="1.8.0",
        py_version="py36",
        entry_point="./entry_point.py",
    )
    pytorch_estimator.fit(
        inputs=TrainingInput(s3_data="s3://my-bucket/my-data/train"),
    )

The :class:`sagemaker.estimator.EstimatorBase.fit` method will call the underlying SageMaker :code:`CreateTrainingJob` API to start a TrainingJob immediately. When composing a pipeline to run a training job, one need to define a :class:`sagemaker.workflow.steps.TrainingStep` first, and we need the training job to be started only when this :class:`sagemaker.workflow.steps.TrainingStep` gets executed during a pipeline execution. This is where the pipeline session :class:`sagemaker.workflow.pipeline_context.PipelineSession` came in.

.. code-block:: python

    pytorch_estimator = PyTorch(
        sagemaker_session=sagemaker.Session(),
        role=sagemaker.get_execution_role(),
        instance_type="ml.c5.xlarge",
        instance_count=1,
        framework_version="1.8.0",
        py_version="py36",
        entry_point="./entry_point.py",
    )
    pytorch_estimator.fit(
        inputs=TrainingInput(s3_data="s3://my-bucket/my-data/train"),
    )


.. code-block:: python

    from sagemaker.workflow.pipeline_context import PipelineSession

    pytorch_estimator = PyTorch(
        sagemaker_session=PipelineSession(),
        role=sagemaker.get_execution_role(),
        instance_type="ml.c5.xlarge",
        instance_count=1,
        framework_version="1.8.0",
        py_version="py36",
        entry_point="./entry_point.py",
    )

    step = TrainingStep(
        name="MyTrainingStep",
        // code just like how you trigger a training job before,
        // pipeline session will take care of delaying the start
        // of the training job during pipeline execution.
        step_args=pytorch_estimator.fit(
            inputs=TrainingInput(s3_data="s3://my-bucket/my-data/train"),
        ),
        displayName="MyTrainingStepDisplayName",
        description="This is MyTrainingStep",
        cache_config=CacheConfig(...),
        retry_policies=[...],
        depends_on=[...],
    )

When you use :class:`sagemaker.workflow.pipeline_context.PipelineSession` rather than :class:`sagemaker.session.Session`, the :code:`.fit` method does not immediately start a training job. Instead, the :code:`.fit` method delays the request to call :code:`CreateTrainingJob`, so that you can first define your :class:`sagemaker.workflow.steps.TrainingStep`.

.. warning::
   A :class:`sagemaker.workflow.pipeline_context.PipelineSession` must be given in order to start the job during pipeline execution time. Otherwise, a training job will get started immediately.

Local Pipeline Session
======================

Like Pipeline Session, Local Pipeline Session provides a convenient way to capture input job arguments without starting the job. These input arguments can be provided in the :code:`step_args` parameter to their corresponding `Pipelines step type <https://sagemaker.readthedocs.io/en/stable/workflows/pipelines/sagemaker.workflow.pipelines.html#sagemaker.workflow.steps.Step>`__. The difference between :class:`sagemaker.workflow.pipeline_context.PipelineSession` and :class:`sagemaker.workflow.pipeline_context.LocalPipelineSession` is that :class:`sagemaker.workflow.pipeline_context.LocalPipelineSession` is used to run SageMaker pipelines locally (in local mode) whereas using :class:`sagemaker.workflow.pipeline_context.PipelineSession` runs the job on the managed service.

.. code-block:: python

    from sagemaker.workflow.pipeline_context import LocalPipelineSession

    local_pipeline_session = LocalPipelineSession()

    pytorch_estimator = PyTorch(
        sagemaker_session=local_pipeline_session,
        role=sagemaker.get_execution_role(),
        instance_type="ml.c5.xlarge",
        instance_count=1,
        framework_version="1.8.0",
        py_version="py36",
        entry_point="./entry_point.py",
    )

    step = TrainingStep(
        name="MyTrainingStep",
        step_args=pytorch_estimator.fit(
            inputs=TrainingInput(s3_data="s3://my-bucket/my-data/train"),
        )
    )

    pipeline = Pipeline(
        name="MyPipeline",
        steps=[step],
        sagemaker_session=local_pipeline_session
    )

    pipeline.create(
        role_arn=sagemaker.get_execution_role(),
        description="local pipeline example"
    )

    // pipeline will execute locally
    pipeline.start()

    steps = pipeline.list_steps()

    training_job_name = steps['PipelineExecutionSteps'][0]['Metadata']['TrainingJob']['Arn']

    step_outputs = pipeline_session.sagemaker_client.describe_training_job(TrainingJobName = training_job_name)


Pipeline Parameters
======================

You can parameterize your pipeline definition using parameters. You can reference parameters that you define throughout your pipeline definition. Parameters have a default value, which you can override by specifying parameter values when starting a pipeline execution.

- :class:`sagemaker.workflow.parameters.ParameterString` – Representing a string parameter.
- :class:`sagemaker.workflow.parameters.ParameterInteger` – Representing an integer parameter.
- :class:`sagemaker.workflow.parameters.ParameterFloat` – Representing a float parameter.
- :class:`sagemaker.workflow.parameters.ParameterBoolean` – Representing a Boolean Python type.

Here is an example:

.. code-block:: python

    from sagemaker.workflow.parameters import (
        ParameterInteger,
        ParameterString,
        ParameterFloat,
        ParameterBoolean,
    )
    from sagemaker.workflow.pipeline_context import PipelineSession

    session = PipelineSession()

    instance_count = ParameterInteger(name="InstanceCount", default_value=2)
    app_managed = ParameterBoolean(name="AppManaged", default_value=False)

    inputs = [
        ProcessingInput(
            source="s3://my-bucket/sourcefile",
            destination="/opt/ml/processing/inputs/",
            app_managed=app_managed
        ),
    ]

    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        role=sagemaker.get_execution_role(),
        instance_type="ml.m5.xlarge",
        instance_count=instance_count,
        command=["python3"],
        sagemaker_session=session,
        base_job_name="test-sklearn",
    )

    step_sklearn = ProcessingStep(
        name="MyProcessingStep",
        step_args=sklearn_processor.run(
            inputs=inputs, code="./my-local/script.py"
        ),
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count, app_managed],
        steps=[step_sklearn],
        sagemaker_session=session,
    )

    # you can override the default parameter values
    pipeline.start({
       "InstanceCount": 2,
       "AppManaged": True,
    })

Step Dependencies
====================
There are two types of step dependencies: a `data dependency`_ and a `custom dependency`_. To create data dependencies between steps, pass the properties or the outputs of one step as the input to another step in the pipeline. This is called property reference. Alternatively, you can specify a custom dependency to make sure that a pipeline execution does not start a new step until all dependent steps are completed.

.. _data dependency: https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html#build-and-manage-data-dependency
.. _custom dependency: https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html#build-and-manage-custom-dependency

Data Dependency — Property Reference
--------------------------------------------

A step property is an attribute of a step that represents the output values from a step execution. For example, :code:`TrainingStep.Properties.TrainingJobName` is a property of a :class:`sagemaker.workflow.steps.TrainingStep`.

For a step that references a SageMaker job (e.g. :class:`sagemaker.workflow.steps.ProcessingStep`, :class:`sagemaker.workflow.steps.TrainingStep`, or :class:`sagemaker.workflow.steps.TransformStep`), the step property matches the attributes of that SageMaker job. For example, :class:`sagemaker.workflow.steps.TrainingStep`. properties match the attributes that result from calling :code:`DescribeTrainingJob`.  :code:`TrainingJobName` is an attribute from a :code:`DescribeTrainingJob` result. Therefore, :code:`TrainingJobName` is a :class:`sagemaker.workflow.steps.TrainingStep` property, and can be referenced as :code:`TrainingStep.Properties.TrainingJobName`.

You can build data dependencies from one step to another using this kind of property reference. These data dependencies are then used by SageMaker Pipelines to construct the directed acyclic graph (DAG) from the pipeline definition. These properties can be referenced as placeholder values and are resolved at runtime.

For each step type you can refer to the following properties for data dependency creation:

TrainingStep
`````````````
Referable Property List:

- `DescribeTrainingJob`_

.. _DescribeTrainingJob: https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_DescribeTrainingJob.html#API_DescribeTrainingJob_ResponseSyntax

Example:

.. code-block:: python

    step_train = TrainingStep(...)
    model = Model(
        image_uri="my-dummy-image",
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        ...
    )
    # assume your training job will produce a metric called "val:acc"
    # and you would like to use it to demtermine if you want to create
    # a SageMaker Model for it.
    step_condition = ConditionStep(
        conditions = [
            ConditionGreaterThanOrEqualTo(
                left=step_train.properties.FinalMetricDataList['val:acc'].Value
                right=0.95
        )],
        if_steps = [step_model_create],
    )

ProcessingStep
````````````````
Referable Property List:

- `DescribeProcessingJob`_

.. _DescribeProcessingJob: https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_DescribeProcessingJob.html#API_DescribeProcessingJob_ResponseSyntax

.. code-block:: python

    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type="ml.m5.xlarge",
        instance_count=1,
        base_job_name="sklearn-abalone-preprocess",
        sagemaker_session=PipelineSession(),
        role=sagemaker.get_execution_role(),
    )

    step_process = ProcessingStep(
        name="MyProcessingStep",
        ...,
        step_args = sklearn_processor.run(
            ...,
            outputs=[
                ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ],
            code="./local/preprocess.py",
            arguments=["--input-data", "s3://my-input"]
        ),
    )

    step_args = estimator.fit(inputs=TrainingInput(
        s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
    ))

TransformStep
````````````````
Referable Property List:

`DescribeTransformJob`_

.. _DescribeTransformJob: https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_DescribeTransformJob.html#API_DescribeTransformJob_ResponseSyntax

.. code-block:: python

    step_transform = TransformStep(...)
    transform_output = step_transform.TransformOutput.S3OutputPath

TuningStep
`````````````
Referable Property List:

- `DescribeHyperParameterTuningJob`_
- `ListTrainingJobsForHyperParameterTuningJob`_

.. _DescribeHyperParameterTuningJob: https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_DescribeHyperParameterTuningJob.html#API_DescribeHyperParameterTuningJob_ResponseSyntax
.. _ListTrainingJobsForHyperParameterTuningJob: https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ListTrainingJobsForHyperParameterTuningJob.html#API_ListTrainingJobsForHyperParameterTuningJob_ResponseSyntax

Example:

.. code-block:: python

    bucket = "my-bucket"
    model_prefix = "my-model"

    step_tune = TuningStep(...)
    # tuning step can launch multiple training jobs, thus producing multiple model artifacts
    # we can create a model with the best performance
    best_model = Model(
        model_data=Join(
            on="/",
            values=[
                f"s3://{bucket}/{model_prefix}",
                # from DescribeHyperParameterTuningJob
                step_tune.properties.BestTrainingJob.TrainingJobName,
                "output/model.tar.gz",
            ],
        )
    )
    # we can also access any top-k best as we wish
    second_best_model = Model(
        model_data=Join(
            on="/",
            values=[
                f"s3://{bucket}/{model_prefix}",
                # from ListTrainingJobsForHyperParameterTuningJob
                step_tune.properties.TrainingJobSummaries[1].TrainingJobName,
                "output/model.tar.gz",
            ],
        )
    )

:class:`sagemaker.workflow.steps.TuningStep` also has a helper function to generate any :code:`top-k` model data URI easily:

.. code-block:: python

    model_data = step_tune.get_top_model_s3_uri(
        top_k=0, # best model
        s3_bucket=bucket,
        prefix=model_prefix
    )

AutoMLStep
`````````````
Referable Property List:

- `DescribeAutoMLJob`_
- `BestCandidateProperties.ModelInsightsJsonReportPath`_
- `BestCandidateProperties.ExplainabilityJsonReportPath`_

.. _DescribeAutoMLJob: https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_DescribeAutoMLJob
.. _BestCandidateProperties.ModelInsightsJsonReportPath: https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CandidateArtifactLocations.html#sagemaker-Type-CandidateArtifactLocations-ModelInsights
.. _BestCandidateProperties.ExplainabilityJsonReportPath: https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CandidateArtifactLocations.html#sagemaker-Type-CandidateArtifactLocations-Explainability

Example:

.. code-block:: python

    step_automl = AutoMLStep(...)

    auto_ml_model = step_automl.get_best_model(<role>)

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=auto_ml_step.properties.BestCandidateProperties.ModelInsightsJsonReportPath,
            content_type="application/json",
        ),
        explainability=MetricsSource(
            s3_uri=auto_ml_step.properties.BestCandidateProperties.ExplainabilityJsonReportPath,
            content_type="application/json",
        )
    )

    step_args_register_model = auto_ml_model.register(
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.t2.medium", "ml.m5.large"],
    transform_instances=["ml.m5.large"],
    model_package_group_name="My model package group name",
    approval_status="PendingManualApproval",
    model_metrics=model_metrics,
    )

    step_register_model = ModelStep(
        name="auto-ml-model-register",
        step_args=step_args_register_model,
    )

ModelStep
````````````````
Referable Property List:

- `DescribeModel`_

  OR
- `DescribeModelPackage`_

.. _DescribeModel: https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_DescribeModel.html#API_DescribeModel_ResponseSyntax
.. _DescribeModelPackage: https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_DescribeModelPackage.html#API_DescribeModelPackage_ResponseSyntax

Example:

For model creation usecase:

.. code-block:: python

    create_model_step = ModelStep(
        name="MyModelCreationStep",
        step_args = model.create(...)
    )
    model_data = create_model_step.properties.PrimaryContainer.ModelDataUrl

For model registration usercase:

.. code-block:: python

    register_model_step = ModelStep(
        name="MyModelRegistrationStep",
        step_args=model.register(...)
    )
    approval_status=register_model_step.properties.ModelApprovalStatus

LambdaStep
`````````````
Referable Property List:

- :code:`OutputParameters`: A list of key-value pairs `OutputParameter`_ as the output of the Lambda execution.

.. _OutputParameter: https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_OutputParameter.html


Example:

.. code-block:: python

    str_outputParam = LambdaOutput(output_name="output1", output_type=LambdaOutputTypeEnum.String)
    int_outputParam = LambdaOutput(output_name"output2", output_type=LambdaOutputTypeEnum.Integer)
    bool_outputParam = LambdaOutput(output_name"output3", output_type=LambdaOutputTypeEnum.Boolean)
    float_outputParam = LambdaOutput(output_name"output4", output_type=LambdaOutputTypeEnum.Float)

    step_lambda = LambdaStep(
        name="MyLambdaStep",
        lambda_func=Lambda(
            function_arn="arn:aws:lambda:us-west-2:123456789012:function:sagemaker_test_lambda",
            session=PipelineSession(),
        ),
        inputs={"arg1": "foo", "arg2": 5},
        outputs=[
            str_outputParam, int_outputParam, bool_outputParam, float_outputParam
       ],
    )
    output_ref = step_lambda.properties.Outputs["output1"]

Where the lambda function with :code:`arn arn:aws:lambda:us-west-2:123456789012:function:sagemaker_test_lambda`
should output like this:

.. code-block:: python

    def handler(event, context):
        ...
        return {
            "output1": "string_value",
            "output2": 1,
            "output3": True,
            "output4": 2.0,
        }

Note that the output parameters can not be nested. Otherwise, the value will be treated as a single string. For instance, if your lambda outputs

.. code-block:: json

    {
        "output1": {
            "nested_output1": "my-output"
        }
    }

This will be resolved as :code:`{"output1": "{\"nested_output1\":\"my-output\"}"}` by which if you refer :code:`step_lambda.properties.Outputs["output1"]["nested_output1"]` later, a non-retryable client error will be thrown.

CallbackStep
`````````````

Referable Property List:

- :code:`OutputParameters`: A list of key-value pairs `OutputParameter`_ defined by `SendPipelineExecutionStepSuccess`_ call.

.. _SendPipelineExecutionStepSuccess: https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_SendPipelineExecutionStepSuccess.html call.

Example:

.. code-block:: python

    param = ParameterInteger(name="MyInt")
    outputParam = CallbackOutput(output_name="output1", output_type=CallbackOutputTypeEnum.String)
    step_callback = CallbackStep(
        name="MyCallbackStep",
        depends_on=["TestStep"],
        sqs_queue_url="https://sqs.us-east-2.amazonaws.com/123456789012/MyQueue",
        inputs={"arg1": "foo", "arg2": 5, "arg3": param},
        outputs=[outputParam],
    )
    output_ref = step_callback.properties.Outputs["output1]

The output parameters cannot be nested. If the values are nested, they will be treated as a single string value. For example, a nested output value of

.. code-block:: json

    {
        "output1": {
            "nested_output1": "my-output"
        }
    }

is resolved as :code:`{"output1": "{\"nested_output1\":\"my-output\"}"}`. If you try to refer to :code:`step_callback.properties.Outputs["output1"]["nested_output1"]` this will throw a non-retryable client error.


QualityCheckStep
```````````````````

Referable Property List:

- :code:`CalculatedBaselineConstraints`: The baseline constraints file calculated by the underlying Model Monitor container.
- :code:`CalculatedBaselineStatistics`: The baseline statistics file calculated by the underlying Model Monitor container.
- :code:`BaselineUsedForDriftCheckStatistics & BaselineUsedForDriftCheckConstraints`: These are the two properties used to set drift_check_baseline in the Model Registry. The values set in these properties vary depending on the parameters passed to the step.

ClarifyCheckStep
```````````````````

Referable Property List:

- :code:`CalculatedBaselineConstraints`: The baseline constraints file calculated by the underlying Clarify container.
- :code:`BaselineUsedForDriftCheckConstraints`: This property is used to set drift_check_baseline in the Model Registry. The values set in this property will vary depending on the parameters passed to the step.

More examples about QualityCheckStep and ClarifyCheckStep can be found in `SageMaker Pipelines integration with Model Monitor and Clarify`_ notebook

EMRStep
`````````````
Referable Property List:

- :code:`ClusterId`: The Id of the EMR cluster.

You can see more details at `AWS official developer guide for Step Introductions`_

.. _AWS official developer guide for Step Introductions: https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html

Custom Dependency
------------------
To build a custom dependency, simply add the desired step or steps to another step’s :code:`depends_on` attribute as follows:

.. code-block:: python

    step_1 = ProcessingStep(
        name="MyProcessingStep",
        step_args=sklearn_processor.run(
            inputs=inputs,
            code="./my-local/my-first-script.py"
        ),
    )

    step_2 = ProcessingStep(
        name="MyProcessingStep",
        step_args=sklearn_processor.run(
            inputs=inputs,
            code="./my-local/my-second-script.py"
        ),
        depends_on=[step_1.name],
    )

In this case, :code:`step_2` will start only when :code:`step_1` is done.

Property File
==============

A :class:`sagemaker.workflow.properties.PropertyFile` is designed to store information that is output from :class:`sagemaker.workflow.steps.ProcessingStep`. The :class:`sagemaker.workflow.functions.JsonGet` function processes a property file . You can use JsonPath notation to query the information.

.. code-block:: python

    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type="ml.m5.xlarge",
        instance_count=1,
        base_job_name="sklearn-abalone-preprocess",
        sagemaker_session=session,
        role=sagemaker.get_execution_role(),
    )

    step_args = sklearn_processor.run(
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
            ProcessingOutput(output_name="hyperparam", source="/opt/ml/processing/evaluation"),
        ],
        code="./local/preprocess.py",
        arguments=["--input-data", "s3://my-input"],
    )

    hyperparam_report = PropertyFile(
        name="AbaloneHyperparamReport",
        output_name="hyperparam",
        path="hyperparam.json",
    )

    step_process = ProcessingStep(
       name="PreprocessAbaloneData",
       step_args=step_args,
       property_files=[hyperparam_report],
    )

To retrieve a file produced by the :class:`sagemaker.workflow.steps.ProcessingStep` as a property file, the :code:`ProcessingOutput.output_name` and the :code:`PropertyFile.output_name` values must be the same. For this example, assume that the :code:`hyperparam.json` value produced by the ProcessingStep in the :code:`/opt/ml/processing/evaluation` directory looks similar to the following:

.. code-block:: json

    {
        "hyperparam": {
            "eta": {
                "value": 0.6
            }
        }
    }

Then, you can query this value using :class:`sagemaker.workflow.functions.JsonGet` and use the value for any subsequent steps:

.. code-block:: python

    eta = JsonGet(
     step_name=step_process.name,
     property_file=hyperparam_report,
     json_path="hyperparam.eta.value",
    )

Conditions
============

Condition step is used to evaluate the condition of step properties to assess which action should be taken next in the pipeline. It takes a list of conditions, and a list steps to execute if all conditions are evaluated to be true, and another list of steps to execute otherwise. For instance:

.. code-block:: python

    step_condition = ConditionStep(
        # The conditions are evaluated with operator AND
        conditions = [condition_1, condition_2, condition_3, condition_4],
        if_steps = [step_register],
        else_steps = [step_fail],
    )

There are eight types of condition are supported, they are:

- :class:`sagemaker.workflow.conditions.ConditionEquals`
- :class:`sagemaker.workflow.conditions.ConditionGreaterThan`
- :class:`sagemaker.workflow.conditions.ConditionGreaterThanOrEqualTo`
- :class:`sagemaker.workflow.conditions.ConditionLessThan`
- :class:`sagemaker.workflow.conditions.ConditionLessThanOrEqualTo`
- :class:`sagemaker.workflow.conditions.ConditionIn`
- :class:`sagemaker.workflow.conditions.ConditionNot`
- :class:`sagemaker.workflow.conditions.ConditionOr`

:class:`sagemaker.workflow.properties.PropertyFile` and :class:`sagemaker.workflow.functions.JsonGet` introduced above is particularly handy when used together with conditions. Here is an example:

.. code-block:: python

    step_train = TrainingStep(...)
    model = Model(
        image_uri="my-dummy-image",
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        ...
    )

    step_args = sklearn_processor.run(
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
            ProcessingOutput(output_name="hyperparam", source="/opt/ml/processing/evaluation"),
        ],
        code="./local/preprocess.py",
        arguments=["--input-data", "s3://my-input"],
    )

    eval_report = PropertyFile(
        name="AbaloneHyperparamReport",
        output_name="hyperparam",
        path="hyperparam.json",
    )

    step_process = ProcessingStep(
        name="PreprocessAbaloneData",
        step_args=step_args,
        property_files=[eval_report],
    )

    eval_score = JsonGet(
        step_name=step_process.name,
        property_file=eval_report,
        json_path="eval.accuracy",
    )

    # register the model if evaluation score is satisfactory
    register_arg = model.register(
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name="my-model-pkg-name",
        approval_status="Approved",
    )
    step_register = ModelStep(
        name="MyModelCreationStep",
        step_args=register_arg,
    )
    # otherwise, transit to a failure step
    step_fail = FailStep(name="FailStep", ...)

    cond = ConditionStep(
       conditions = [ConditionGreaterThanOrEqualTo(left=eval_score, right=0.95)],
       if_steps = [step_register],
       else_steps = [step_fail],
    )


Pipeline Functions
===================
Several pipeline built-in functions are offered to help you compose your pipeline. Use pipeline functions to assign values to properties that are not available until pipeline execution time.

Join
-----------
Use the :class:`sagemaker.workflow.functions.Join` function to join a list of properties. For example, you can use Join to construct an S3 URI that can only be evaluated at run time, and use that URI to construct the :class:`sagemaker.workflow.steps.TrainingStep` at compile time.

.. code-block:: python

    bucket = ParameterString('bucket', default_value='my-bucket')

    input_uri = Join(
        on="/",
        values=['s3:/', bucket, "my-input")]
    )

    step = TrainingStep(
        name="MyTrainingStep",
        run_args=estimator.fit(inputs=TrainingInput(s3_data=input_uri)),
    )

JsonGet
-----------
Use :class:`sagemaker.workflow.functions.JsonGet` to extract a Json property from a :class:`sagemaker.workflow.properties.PropertyFile` produced by a :class:`sagemaker.workflow.steps.ProcessingStep`, and pass it to subsequent steps. The following example retrieves a hyperparameter value from the :class:`sagemaker.workflow.properties.PropertyFile`, and pass it to a subsequent :class:`sagemaker.workflow.steps.TrainingStep`

.. code-block:: python

    session = PipelineSession()

    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type="ml.m5.xlarge",
        instance_count=1,
        base_job_name="sklearn-abalone-preprocess",
        sagemaker_session=session,
        role=sagemaker.get_execution_role(),
    )

    step_args = sklearn_processor.run(
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
            ProcessingOutput(output_name="hyperparam", source="/opt/ml/processing/evaluation"),
        ],
        code="./local/preprocess.py",
        arguments=["--input-data", "s3://my-input"],
    )

    hyperparam_report = PropertyFile(
        name="AbaloneHyperparamReport",
        output_name="hyperparam",
        path="hyperparam.json",
    )

    step_process = ProcessingStep(
        name="PreprocessAbaloneData",
       step_args=step_args,
        property_files=[hyperparam_report],
    )

    xgb_train = Estimator(
        image_uri="s3://my-image-uri",
        instance_type="ml.c5.xlarge",
        instance_count=1,
        output_path="s3://my-output-path",
        base_job_name="abalone-train",
        sagemaker_session=session,
        role=sagemaker.get_execution_role(),
    )

    eta = JsonGet(
     step_name=step_process.name,
     property_file=hyperparam_report,
     json_path="hyperparam.eta.value",
    )

    xgb_train.set_hyperparameters(
        objective="reg:linear",
        num_round=50,
        max_depth=5,
        eta=eta,
        gamma=4,
        min_child_weight=6,
        subsample=0.7,
        silent=0,
    )

    step_args = xgb_train.fit(inputs={
        "train": TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                "train"
            ].S3Output.S3Uri,
            content_type="text/csv",
        ),
        "validation": TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                "validation"
            ].S3Output.S3Uri,
            content_type="text/csv",
        ),
    },)

    step_train = TrainingStep(
        name="TrainAbaloneModel",
        step_args=step_args,
    )

Execution Variable
====================

There are a number of properties for a pipeline execution that can only be resolved at run time. However, they can be accessed at compile time using execution variables.

- :class:`sagemaker.workflow.execution_variables.ExecutionVariables.START_DATETIME`: The start time of an execution.
- :class:`sagemaker.workflow.execution_variables.ExecutionVariables.CURRENT_DATETIME`: The time when the variable is being evaluated during an execution.
- :class:`sagemaker.workflow.execution_variables.ExecutionVariables.PIPELINE_EXECUTION_ID`: The Id of an execution.
- :class:`sagemaker.workflow.execution_variables.ExecutionVariables.PIPELINE_EXECUTION_ARN`: The execution ARN for an execution.
- :class:`sagemaker.workflow.execution_variables.ExecutionVariables.PIPELINE_NAME`: The name of the pipeline.
- :class:`sagemaker.workflow.execution_variables.ExecutionVariables.PIPELINE_ARN`: The ARN of the pipeline.
- :class:`sagemaker.workflow.execution_variables.ExecutionVariables.TRAINING_JOB_NAME`: The name of the training job launched by the training step.
- :class:`sagemaker.workflow.execution_variables.ExecutionVariables.PROCESSING_JOB_NAME`: The name of the processing job launched by the processing step.

You can use these execution variables as you see fit. The following example uses the :code:`START_DATETIME` execution variable to construct a processing output path:

.. code-block:: python

    bucket = ParameterString('bucket', default_value='my-bucket')

    output_path = Join(
        on="/",
        values=['s3:/', bucket, 'my-train-output-', ExecutionVariables.START_DATETIME])]
    )

    step = ProcessingStep(
        name="MyTrainingStep",
        step_args=processor.fit(
            inputs=ProcessingInput(source="s3://my-input"),
            outputs=[
                ProcessingOutput(
                    output_name="train",
                    source="/opt/ml/processing/train",
                    destination=output_path,
                ),
            ],
        ),
    )


Step Parallelism
===================
When a step does not depend on any other step, it is run immediately upon pipeline execution. However, executing too many pipeline steps in parallel can quickly exhaust available resources. Control the number of concurrent steps for a pipeline execution with :class:`sagemaker.workflow.parallelism_config.ParallelismConfiguration`.

The following example uses :class:`sagemaker.workflow.parallelism_config.ParallelismConfiguration` to set the concurrent step limit to five.

.. code-block:: python

    pipeline.create(
        parallelism_config=ParallelismConfiguration(5),
    )


Caching Configuration
==============================
Executing the step without changing its configurations, inputs, or outputs can be a waste. Thus, we can enable caching for pipeline steps. When you use step signature caching, SageMaker Pipelines tries to use a previous run of your current pipeline step instead of running the step again. When previous runs are considered for reuse, certain arguments from the step are evaluated to see if any have changed. If any of these arguments have been updated, the step will execute again with the new configuration.

When you turn on caching, you supply an expiration time (in `ISO8601 duration string format <https://en.wikipedia.org/wiki/ISO_8601#Durations>`__). The expiration time indicates how old a previous execution can be to be considered for reuse.

.. code-block:: python

    cache_config = CacheConfig(
        enable_caching=True,
        expire_after="P30d" # 30-day
    )

You can format your ISO8601 duration strings like the following examples:

- :code:`p30d`: 30 days
- :code:`P4DT12H`: 4 days and 12 hours
- :code:`T12H`: 12 hours

Caching is supported for the following step types:

- :class:`sagemaker.workflow.steps.TrainingStep`
- :class:`sagemaker.workflow.steps.ProcessingStep`
- :class:`sagemaker.workflow.steps.TransformStep`
- :class:`sagemaker.workflow.steps.TuningStep`
- :class:`sagemaker.workflow.quality_check_step.QualityCheckStep`
- :class:`sagemaker.workflow.clarify_check_step.ClarifyCheckStep`
- :class:`sagemaker.workflow.emr_step.EMRStep`

In order to create pipeline steps and eventually construct a SageMaker pipeline, you provide parameters within a Python script or notebook. The SageMaker Python SDK creates a pipeline definition by translating these parameters into SageMaker job attributes. Some of these attributes, when changed, cause the step to re-run (See `Caching Pipeline Steps <https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-caching.html>`__ for a detailed list). Therefore, if you update a SDK parameter that is used to create such an attribute, the step will rerun. See the following discussion for examples of this in processing and training steps, which are commonly used steps in Pipelines.

The following example creates a processing step:

.. code-block:: python

    from sagemaker.workflow.pipeline_context import PipelineSession
    from sagemaker.sklearn.processing import SKLearnProcessor
    from sagemaker.workflow.steps import ProcessingStep
    from sagemaker.dataset_definition.inputs import S3Input
    from sagemaker.processing import ProcessingInput, ProcessingOutput

    pipeline_session = PipelineSession()

    framework_version = "0.23-1"

    sklearn_processor = SKLearnProcessor(
        framework_version=framework_version,
        instance_type="ml.m5.xlarge",
        instance_count=processing_instance_count,
        role=role,
        sagemaker_session=pipeline_session
    )

    processor_args = sklearn_processor.run(
        inputs=[
            ProcessingInput(
                source="artifacts/data/abalone-dataset.csv",
                input_name="abalone-dataset",
                s3_input=S3Input(
                    local_path="/opt/ml/processing/input",
                    s3_uri="artifacts/data/abalone-dataset.csv",
                    s3_data_type="S3Prefix",
                    s3_input_mode="File",
                    s3_data_distribution_type="FullyReplicated",
                    s3_compression_type="None",
                )
            )
        ],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code="artifacts/code/process/preprocessing.py",
    )

    processing_step = ProcessingStep(
        name="Process",
        step_args=processor_args,
        cache_config=cache_config
    )

The following parameters from the example cause additional processing step iterations when you change them:

- :code:`framework_version`: This parameter is used to construct the :code:`image_uri` for the `AppSpecification <https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_AppSpecification.html>`__ attribute of the processing job.
- :code:`inputs`: Any :class:`ProcessingInputs` are passed through directly as job `ProcessingInputs <https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ProcessingInput.html>`__. Input :code:`source` files that exist in the container’s local file system are uploaded to S3 and given a new :code:`S3_Uri`. If the S3 path changes, a new processing job is initiated. For examples of S3 paths, see the **S3 Artifact Folder Structure** section.
- :code:`code`: The code parameter is also packaged as a `ProcessingInput <https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ProcessingInput.html>`__ job. For local files, a unique hash is created from the file. The file is then uploaded to S3 with the hash included in the path. When a different local file is used, a new hash is created and the S3 path for that `ProcessingInput <https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ProcessingInput.html>`__ changes, initiating a new step run. For examples S3 paths, see the **S3 Artifact Folder Structure** section.

The following example creates a training step:

.. code-block:: python

    from sagemaker.sklearn.estimator import SKLearn
    from sagemaker.workflow.steps import TrainingStep

    pipeline_session = PipelineSession()

    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.0-1",
        py_version="py3",
        instance_type="ml.m5.xlarge",
    )

    hyperparameters = {
        "dataset_frequency": "H",
        "timestamp_format": "yyyy-MM-dd hh:mm:ss",
        "number_of_backtest_windows": "1",
        "role_arn": role_arn,
        "region": region,
    }

    sklearn_estimator = SKLearn(
        entry_point="train.py",
        role=role_arn,
        image_uri=container_image_uri,
        instance_type=training_instance_type,
        sagemaker_session=pipeline_session,
        base_job_name="training_job",
        hyperparameters=hyperparameters,
        enable_sagemaker_metrics=True,
    )

    train_args = xgb_train.fit(
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        }
    )

    training_step = TrainingStep(
        name="Train",
        estimator=sklearn_estimator,
        cache_config=cache_config
    )

The following parameters from the example cause additional training step iterations when you change them:

- :code:`image_uri`: The :code:`image_uri` parameter defines the image used for training, and is used directly in the `AlgorithmSpecification <https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_AlgorithmSpecification.html>`__ attribute of the training job.
- :code:`hyperparameters`: All of the hyperparameters are used directly in the `HyperParameters <https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_DescribeTrainingJob.html#API_DescribeTrainingJob_ResponseSyntax>`__ attribute for the training job.
- :code:`entry_point`: The entry point file is included in the training job’s `InputDataConfig Channel <https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_Channel.html>`__ array. A unique hash is created from the file (and any other dependencies), and then the file is uploaded to S3 with the hash included in the path. When a different entry point file is used, a new hash is created and the S3 path for that `InputDataConfig Channel <https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_Channel.html>`__ object changes, initiating a new step run. For examples of what the S3 paths look like, see the **S3 Artifact Folder Structure** section.
- :code:`inputs`: The inputs are also included in the training job’s `InputDataConfig <https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_Channel.html>`__. Local inputs are uploaded to S3. If the S3 path changes, a new training job is initiated. For examples of S3 paths, see the **S3 Artifact Folder Structure** section.

S3 Artifact Folder Structure
----------------------------

You use the following S3 paths when uploading local input and code artifacts, and when saving output artifacts.

*Processing*

- Code: :code:`s3://bucket_name/pipeline_name/code/<code_hash>/file.py`. The file could also be a tar.gz of source_dir and dependencies.
- Input Data: :code:`s3://bucket_name/pipeline_name/step_name/input/input_name/file.csv`
- Configuration: :code:`s3://bucket_name/pipeline_name/step_name/input/conf/<configuration_hash>/configuration.json`
- Output: :code:`s3://bucket_name/pipeline_name/<execution_id>/step_name/output/output_name`

*Training*

- Code: :code:`s3://bucket_name/code_location/pipeline_name/code/<code_hash>/code.tar.gz`
- Output: The output paths for Training jobs can vary - the default output path is the root of the s3 bucket: :code:`s3://bucket_name`. For Training jobs created from a Tuning job, the default path includes the Training job name created by the Training platform, formatted as :code:`s3://bucket_name/<training_job_name>/output/model.tar.gz`.

*Transform*

- Output: :code:`s3://bucket_name/pipeline_name/<execution_id>/step_name`

.. warning::
    For input artifacts such as data or code files, the actual content of the artifacts is not tracked, only the S3 path. This means that if a file in S3 is updated and re-uploaded directly with an identical name and path, then the step does NOT run again.


Retry Policy
===============

We can configure step wise retry behavior for certain step types. During a pipeline step execution, there are two points in which you might encounter errors.

1. You might encounter errors when trying to create or start a SageMaker job like a :code:`ProcessingJob` or :code:`TrainingJob`.
2. You might encounter errors when a SageMaker job like a :code:`ProcessingJob` or :code:`TrainingJob`. finishes with failures.

There are two types of retry policies to handle these scenarios:

- :class:`sagemaker.workflow.retry.StepRetryPolicy`
- :class:`sagemaker.workflow.retry.SageMakerJobStepRetryPolicy`

The :code:`StepRetryPolicy` is used if service faults (like a network issue) or throttling are recognized when creating a SageMaker job.

.. code-block:: python

    StepRetryPolicy(
        exception_types=[
            StepExceptionTypeEnum.SERVICE_FAULT,
            StepExceptionTypeEnum.THROTTLING,
        ],
        expire_after_min=5,
        interval_seconds=10,
        backoff_rate=2.0
    )


Note: A pipeline step type that supports the :code:`StepRetryPolicy` will attempt exponential retries with a one-second interval by default for service faults and throttling. This behavior can be overridden using the policy above.

The :code:`SageMakerJobStepRetryPolicy` is used if a failure reason is given after a job is done. There are many reasons why a job can fail. The :code:`SageMakerJobStepRetryPolicy` supports retry configuration for the following failures:

- :code:`SageMaker.JOB_INTERNAL_ERROR`
- :code:`SageMaker.CAPACITY_ERROR`
- :code:`SageMaker.RESOURCE_LIMIT`

The following example specifies that a SageMaker job should retry if it fails due to a resource limit exception. The job will retry exponentially, starting at an interval of 60 seconds, and will only attempt to retry this job for two hours total.

.. code-block:: python

    SageMakerJobStepRetryPolicy(
        exception_types=[SageMakerJobExceptionTypeEnum.RESOURCE_LIMIT]
        expire_after_min=120,
        interval_seconds=60,
        backoff_rate=2.0
    )


For more information, see `Retry Policy for Pipeline Steps`_ in the *Amazon SageMaker Developer Guide*.

.. _Retry Policy for Pipeline Steps: https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-retry-policy.html.

Model Step
===============

After model artifacts are produced by either a :code:`TrainingJob` or :code:`TuningJob`, you might want to create a SageMaker Model, or register that model to SageMaker Model Registry. This is where the :class:`sagemaker.workflow.model_step.ModelStep` comes in.

Follow the example below to create a SageMaker Model and register it to SageMaker Model Registry using :class:`sagemaker.workflow.model_step.ModelStep`.

.. code-block:: python

    step_train = TrainingStep(...)
    model = Model(
        image_uri=pytorch_estimator.training_image_uri(),
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=pipeline_session,
        role=role,
    )

    # we might also want to create a SageMaker Model
    step_model_create = ModelStep(
       name="MyModelCreationStep",
       step_args=model.create(instance_type="ml.m5.xlarge"),
    )

    # in addition, we might also want to register a model to SageMaker Model Registry
    register_model_step_args = model.register(
        content_types=["*"],
        response_types=["application/json"],
        inference_instances=["ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        description="MyModelPackage",
    )

    step_model_registration = ModelStep(
       name="MyModelRegistration",
       step_args=register_model_step_args,
    )
    ...

When model repacking is needed, :class:`sagemaker.workflow.model_step.ModelStep`. is a collection of steps. Model repacking unpacks model data, creates a new model tarball file that includes any custom inference scripts, and uploads this tarball file to Amazon S3. Once a model is repacked, it is ready to deploy to an endpoint or be registered as a model package.

:class:`sagemaker.workflow.model_step.ModelStep` uses the provided inputs to automatically detect if a repack is needed. If a repack is needed, :class:`sagemaker.workflow.steps.TrainingStep` is added to the step collection for that repack. Then, either :class:`sagemaker.workflow.steps.CreateModelStep` or :class:`sagemaker.workflow.step_collections.RegisterModelStep` will be chained after it.

MonitorBatchTransform Step
===========================

MonitorBatchTransformStep is a new step type that allows customers to use SageMaker Model Monitor with batch transform jobs that are a part of their pipeline. Using this step, customers can set up the following monitors for their batch transform job: data quality, model quality, model bias, and feature attribution.


When configuring this step, customers have the flexibility to run the monitoring job before or after the transform job executes. There is an additional flag called :code:`fail_on_violation` which will fail the step if set to true and there is a monitoring violation, or will continue to execute the step if set to false.

Here is an example showing you how to configure a :class:`sagemaker.workflow.monitor_batch_transform_step.MonitorBatchTransformStep` with a Data Quality monitor.

.. code-block:: python

    from sagemaker.workflow.pipeline_context import PipelineSession

    from sagemaker.transformer import Transformer
    from sagemaker.model_monitor import DefaultModelMonitor
    from sagemaker.model_monitor.dataset_format import DatasetFormat
    from sagemaker.workflow.check_job_config import CheckJobConfig
    from sagemaker.workflow.quality_check_step import DataQualityCheckConfig

    from sagemaker.workflow.parameters import ParameterString

    pipeline_session = PipelineSession()

    transform_input_param = ParameterString(
        name="transform_input",
        default_value=f"s3://my-bucket/my-prefix/my-transform-input",
    )

    # the resource configuration for the monitoring job
    job_config = CheckJobConfig(
        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        ...
    )

The following code sample demonstrates how to set up an on-demand batch transform *data quality* monitor:

.. code-block:: python

    # configure your transformer
    transformer = Transformer(..., sagemaker_session=pipeline_session)
    transform_arg = transformer.transform(
        transform_input_param,
        content_type="text/csv",
        split_type="Line",
        ...
    )

    data_quality_config = DataQualityCheckConfig(
        baseline_dataset=transform_input_param,
        dataset_format=DatasetFormat.csv(header=False),
        output_s3_uri="s3://my-report-path",
    )

    from sagemaker.workflow.monitor_batch_transform_step import MonitorBatchTransformStep

    transform_and_monitor_step = MonitorBatchTransformStep(
        name="MyMonitorBatchTransformStep",
        transform_step_args=transform_arg,
        monitor_configuration=data_quality_config,
        check_job_configuration=job_config,
        # since data quality only looks at the inputs,
        # so there is no need to wait for the transform output.
        monitor_before_transform=True,
        # if violation is detected in the monitoring, and you want to skip it
        # and continue running batch transform, you can set fail_on_violation
        # to false.
        fail_on_violation=False,
        supplied_baseline_statistics="s3://my-baseline-statistics.json",
        supplied_baseline_constraints="s3://my-baseline-constraints.json",
    )
    ...

The same example can be extended for model quality, bias, and feature attribute monitoring.

.. warning::
    Note that to run on-demand model quality, you will need to have the ground truth data ready. When running the transform job, include the ground truth inside your transform input, and join the transform inference input and output. Then you can indicate which attribute or column name/index points to the ground truth when run the monitoring job.

.. code-block:: python

    transformer = Transformer(..., sagemaker_session=pipeline_session)

    transform_arg = transformer.transform(
        transform_input_param,
        content_type="text/csv",
        split_type="Line",
        # Note that we need to join both the inference input and output
        # into transform outputs. The inference input needs to have the ground truth.
        # details can be found here
        # https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform-data-processing.html
        join_source="Input",
        # We need to exclude the ground truth inside the inference input
        # before passing it to the prediction model.
        # Assume the first column of our csv file is the ground truth
        input_filter="$[1:]",
        ...
    )

    model_quality_config = ModelQualityCheckConfig(
        baseline_dataset=transformer.output_path,
        problem_type="BinaryClassification",
        dataset_format=DatasetFormat.csv(header=False),
        output_s3_uri="s3://my-output",
        # assume the model output is at column idx 10
        inference_attribute="_c10",
        # As pointed out previously, the first column is the ground truth.
        ground_truth_attribute="_c0",
    )
    from sagemaker.workflow.monitor_batch_transform_step import MonitorBatchTransformStep

    transform_and_monitor_step = MonitorBatchTransformStep(
        name="MyMonitorBatchTransformStep",
        transform_step_args=transform_arg,
        monitor_configuration=data_quality_config,
        check_job_configuration=job_config,
        # model quality job needs the transform outputs, therefore
        # monitor_before_transform can not be true for model quality
        monitor_before_transform=False,
        fail_on_violation=True,
        supplied_baseline_statistics="s3://my-baseline-statistics.json",
        supplied_baseline_constraints="s3://my-baseline-constraints.json",
    )
    ...

=================
Example Notebooks
=================

Feel free to explore the `Amazon SageMaker Example Notebook`_ to explore and experiment with specific SageMaker use cases. The following Notebooks demonstrate examples related to the SageMaker Model Building Pipeline:

.. _Amazon SageMaker Example Notebook: https://sagemaker-examples.readthedocs.io/en/latest/

- `Orchestrate Jobs to Train and Evaluate Models with Amazon SageMaker Pipelines`_
- `Glue ETL as part of a SageMaker pipeline using Pipeline Callback Step`_
- `SageMaker Pipelines Lambda Step`_
- `SageMaker Pipelines integration with Model Monitor and Clarify`_
- `SageMaker Pipelines Tuning Step`_

.. _Orchestrate Jobs to Train and Evaluate Models with Amazon SageMaker Pipelines: https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker-pipelines/tabular/abalone_build_train_deploy/sagemaker-pipelines-preprocess-train-evaluate-batch-transform.ipynb
.. _Glue ETL as part of a SageMaker pipeline using Pipeline Callback Step: https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker-pipelines/tabular/custom_callback_pipelines_step/sagemaker-pipelines-callback-step.ipynb
.. _SageMaker Pipelines Lambda Step: https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker-pipelines/tabular/lambda-step/sagemaker-pipelines-lambda-step.ipynb
.. _SageMaker Pipelines integration with Model Monitor and Clarify: https://github.com/aws/amazon-sagemaker-examples/tree/main/sagemaker-pipelines/tabular/model-monitor-clarify-pipelines
.. _SageMaker Pipelines Tuning Step: https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker-pipelines/tabular/tuning-step/sagemaker-pipelines-tuning-step.ipynb

===================
Limitations
===================

The SageMaker Model Building Pipeline Python SDK offers abstractions to help you construct a pipeline definition at ease. However, there are certain limitations. Read on for information about known issues. If you discover additional limitations, open an issue in the `sagemaker-python-sdk`_ repository.

.. _sagemaker-python-sdk: https://github.com/aws/sagemaker-python-sdk/issues


Parameterization
================

Parameterization is crucial for pipeline composition, it allows you to assign values to properties that are not available until runtime. However, there are limitations.

Incompatibility with other SageMaker Python SDK modules
---------------------------------------------------------

Pipeline parameterization includes pipeline parameters like :class:`sagemaker.workflow.parameters.ParameterString` and :class:`sagemaker.workflow.parameters.ParameterInteger`, property reference, functions like :class:`sagemaker.workflow.functions.Join` and :class:`sagemaker.workflow.functions.JsonGet`, and pipeline execution variables. Pipeline parameterization might not be supported with 100% compatibility when used with other SageMaker Python SDK modules.

For example, when running a training job in script mode, you cannot parameterize the :code:`entry_point` value for estimators inherited from :class:`sagemaker.estimator.EstimatorBase` because a SageMaker EstimatorBase expects an :code:`entry_point` to point to a local Python source file.

.. code-block:: python

    # An example of what not to do
    script_path = ParameterString(name="MyScript", default="s3://my-bucket/my-script.py")
    xgb_script_mode_estimator = XGBoost(
            entry_point=script_path,
            framework_version="1.5-1",
            role=role,
            ...
    )

Not all arguments can be parameterized
---------------------------------------

Many arguments for class constructors or methods from other modules can be parameterized, but not all of them. For example, Inputs or outputs can be parameterized when calling :code:`processor.run`.

.. code-block:: python

    instance_count = ParameterInteger(name="InstanceCount", default_value=2)
    process_s3_input_url = ParameterString(name="ProcessingInputUrl")

    processor = Processor(
       instance_type=instance_count,
       instance_count="ml.m5.xlarge",
       ...
    )
    processor.run(inputs=ProcessingInput(source=process_s3_input_url), ...)

However, you cannot parameterize :code:`git_config` when calling :code:`processor.run`. This is because the source code needs to be downloaded, packaged, and uploaded S3 at compile time and parameterization can only be evaluated at run time.

Not all built-in Python operations can be applied to parameters
-----------------------------------------------------------------

Another limitation of parameterization is that not all built-in Python operations can be applied to a pipeline parameter.  For example, You cannot concatenate the pipeline variables using Python primitives:

.. code-block:: python

    # An example of what not to do
    my_string = "s3://{}/training".format(ParameterString(name="MyBucket", default_value=""))

    # Another example of what not to do
    int_param = str(ParameterInteger(name="MyBucket", default_value=1))

    # Instead, if you want to convert the parameter to string type, do
    int_param.to_string()

The concatenation example above will not work, as the :class:`sagemaker.workflow.parameters.ParameterString` can only be evaluated at run time. Instead, you can concatenate parameters using :class:`sagemaker.workflow.functions.Join`:


This concatenation of :code:`my_string` will not work, as the parameter :code:`MyBucket` can only be evaluated at run time. Instead, the same concatenation can be achieved using function :class:`sagemaker.workflow.functions.Join`:

.. code-block:: python

    my_string = Join(on="", values=[
        "s3://",
        ParameterString(name="MyBucket", default_value=""),
        "/training"]
    )

Pipeline parameters can only be evaluated at run time. If a pipeline parameter needs to be evaluated at compile time, then it will throw an exception.

====================================
Pipeline Composition Alternative
====================================

The SageMaker Python SDK provides you with tools for pipeline composition. Under the hood, it produces a pipeline definition JSON file. If you want to author the pipeline definition by hand, you can follow the `SageMaker Pipeline Definition JSON Schema`_

.. _SageMaker Pipeline Definition JSON Schema: https://aws-sagemaker-mlops.github.io/sagemaker-model-building-pipeline-definition-JSON-schema/index.html
