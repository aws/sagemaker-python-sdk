
##############################
Amazon SageMaker Model Monitor
##############################


Amazon SageMaker Model Monitor allows you to create a set of baseline statistics and constraints using the data with which your model was trained, then set up a schedule to monitor the predictions made on your endpoint.

.. contents::

Background
==========

Amazon SageMaker provides every developer and data scientist with the ability to build, train, and deploy machine learning models quickly. Amazon SageMaker is a fully-managed service that encompasses the entire machine learning workflow. You can label and prepare your data, choose an algorithm, train a model, and then tune and optimize it for deployment. You can deploy your models to production with Amazon SageMaker to make predictions at lower costs than was previously possible.

Amazon SageMaker Model Monitor enables you to capture the input, output and metadata for the invocations of the models that you deploy. It also enables you to analyze the data and monitor its quality. In this notebook, you learn how Amazon SageMaker enables these capabilities.

Setup
=====

To get started, you must satisfy the following prerequisites:

* Specify an AWS Region to host your model.
* Create an IAM role ARN that is used to give Amazon SageMaker access to your data in Amazon Simple Storage Service (Amazon S3). See the `AWS IAM documentation <https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles.html>`__ for how to fine tune the permissions needed.
* Create an S3 bucket used to store the data used to train your model, any additional model data, and the data captured from model invocations. You can use the same bucket for these, or use separate buckets (e.g. if you want different security policies).

Capture real-time inference data from Amazon SageMaker endpoints
================================================================

To enable data capture for monitoring the model data quality, specify the new capture option called ``DataCaptureConfig`` when deploying to an endpoint. You can choose to capture the request payload, the response payload or both with this configuration. The capture config applies to all variants. For more about the ``DataCaptureConfig`` object, see the `API documentation <https://sagemaker.readthedocs.io/en/stable/model_monitor.html#sagemaker.model_monitor.data_capture_config.DataCaptureConfig>`__.

.. code:: python

    from sagemaker.model_monitor import DataCaptureConfig

    data_capture_config = DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=100,
        destination_s3_uri='s3://path/for/data/capture'
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='ml.m4.xlarge',
        data_capture_config=data_capture_config
    )

When you invoke the endpoint, the request and response payload, along with some additional metadata, is saved in the Amazon S3 location that you have specified in the ``DataCaptureConfig``. You should expect to see different files from different time periods, organized based on the hour in which the invocation occurred. The format of the Amazon S3 path is:

.. code::

    s3://{destination-bucket-prefix}/{endpoint-name}/{variant-name}/yyyy/mm/dd/hh/filename.jsonl

You can use the ``S3Downloader`` utility to view and download the captured data in Amazon S3:

.. code:: python

    from sagemaker.s3 import S3Downloader

    # Invoke the endpoint
    predictor.predict(data)

    # Get a list of S3 URIs
    S3Downloader.list('s3://path/for/data/capture')

    # Read a specific file
    S3Downloader.read_file('s3://path/for/data/capture/endpoint-name/variant-name/2020/01/01/00/filename.jsonl')

The contents of the single captured file should be all the data captured in an Amazon SageMaker-specific JSON-line formatted file. Each inference request is captured in a single line in the jsonl file. The line contains both the input and output merged together.

Baselining and continuous monitoring
====================================

In addition to collecting the data, Amazon SageMaker provides the capability for you to monitor and evaluate the data observed by the endpoints. Two tasks are needed for this:

* Create a baseline with which you compare the realtime traffic.
* Setup a schedule to continuously evaluate and compare against the baseline after it has been created.

Constraint suggestion with baseline/training dataset
----------------------------------------------------

You can ask Amazon SageMaker to suggest a set of baseline constraints and generate descriptive statistics that characterize the data in a training dataset stored in Amazon S3. ``DefaultModelMonitor.suggest_baseline()`` starts a Processing Job using a Model Monitor container provided by Amazon SageMaker to generate the constraints. You can read more about ``suggest_baseline()`` in the `API documentation <https://sagemaker.readthedocs.io/en/stable/model_monitor.html#sagemaker.model_monitor.model_monitoring.DefaultModelMonitor.suggest_baseline>`__.

.. code:: python

    from sagemaker.model_monitor import DefaultModelMonitor
    from sagemaker.model_monitor.dataset_format import DatasetFormat

    my_monitor = DefaultModelMonitor(
        role=role,
        instance_count=1,
        instance_type='ml.m5.xlarge',
        volume_size_in_gb=20,
        max_runtime_in_seconds=3600,
    )

    my_monitor.suggest_baseline(
        baseline_dataset='s3://path/to/training-dataset-with-header.csv',
        dataset_format=DatasetFormat.csv(header=True),
    )

With the monitor object, you can also explore the generated constraints and statistics:

.. code:: python

    import pandas as pd

    baseline_job = my_monitor.latest_baselining_job
    schema_df = pd.io.json.json_normalize(baseline_job.baseline_statistics().body_dict["features"])
    schema_df.head(10)

    constraints_df = pd.io.json.json_normalize(baseline_job.suggested_constraints().body_dict["features"])
    constraints_df.head(10)

Analyze the data collected for data quality issues
--------------------------------------------------

You can also analyze and monitor the data with Monitoring Schedules.

Using ``DefaultMonitor.create_monitoring_schedule()``, you can create a model monitoring schedule for an endpoint that compares the baseline resources (constraints and statistics) against the realtime traffic. For more about this method, see the `API documentation <https://sagemaker.readthedocs.io/en/stable/model_monitor.html#sagemaker.model_monitor.model_monitoring.DefaultModelMonitor.create_monitoring_schedule>`__.

.. code:: python

    from sagemaker.model_monitor import CronExpressionGenerator

    my_monitor.create_monitoring_schedule(
        monitor_schedule_name='my-monitoring-schedule',
        endpoint_input=predictor.endpoint_name,
        statistics=my_monitor.baseline_statistics(),
        constraints=my_monitor.suggested_constraints(),
        schedule_cron_expression=CronExpressionGenerator.hourly(),
    )

The schedule starts jobs at the specified interval.

.. note::

    Even for an hourly schedule, Amazon SageMaker has a buffer period of 20 minutes to schedule your execution. This is expected and done for load balancing on the backend.

Once the executions have started, you can use ``list_executions()`` to view them:

.. code:: python

    executions = my_monitor.list_executions()

You can also view the status of a specific execution:

.. code:: python

    latest_execution = executions[-1]

    latest_execution.describe()['ProcessingJobStatus']
    latest_execution.describe()['ExitMessage']

Here are the possible terminal states and what each of them means:

* ``Completed`` - This means the monitoring execution completed and no issues were found in the violations report.
* ``CompletedWithViolations`` - This means the execution completed, but constraint violations were detected.
* ``Failed`` - The monitoring execution failed, maybe due to client error (perhaps incorrect role premissions) or infrastructure issues. Further examination of the FailureReason and ExitMessage is necessary to identify what exactly happened.
* ``Stopped`` - job exceeded the max runtime or was manually stopped.

You can also get the S3 URI for the output with ``latest_execution.output.destination`` and analyze the results.

Visualize results
=================

You can use the monitor object to gather reports for visualization:

.. code:: python

    suggested_constraints = my_monitor.suggested_constraints()
    baseline_statistics = my_monitor.baseline_statistics()

    latest_monitoring_violations = my_monitor.latest_monitoring_constraint_violations()
    latest_monitoring_statistics = my_monitor.latest_monitoring_statistics()

For a tutorial on how to visualize the results, see `SageMaker Model Monitor - visualizing monitoring results <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker_model_monitor/visualization/SageMaker-Model-Monitor-Visualize.ipynb>`__.

Delete the resources
====================

When deleting an endpoint, you need to first delete the monitoring schedule:

.. code:: python

    my_monitor.delete_monitoring_schedule()

    predictor.delete_endpoint()
    predictor.delete_model()

Learn More
==========

Further documentation
---------------------

* API documentation: https://sagemaker.readthedocs.io/en/stable/model_monitor.html
* AWS documentation: https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html
* ``S3Downloader``: https://sagemaker.readthedocs.io/en/stable/s3.html#sagemaker.s3.S3Downloader

Notebook examples
-----------------

Consult our notebook examples for in-depth tutorials: https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker_model_monitor
