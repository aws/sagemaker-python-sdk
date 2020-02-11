##############################
Amazon SageMaker Processing
##############################


Amazon SageMaker Processing allows you to run steps for data pre- or post-processing, feature engineering, data validation, or model evaluation workloads on Amazon SageMaker.

.. contents::

Background
==========

Amazon SageMaker lets developers and data scientists train and deploy machine learning models. With Amazon SageMaker Processing, you can run processing jobs on for data processing steps in your machine learning pipeline, which accept data from Amazon S3 as input, and put data into Amazon S3 as output.

.. image:: ./amazon_sagemaker_processing_image1.png

Setup
=====

The fastest way to run get started with Amazon SageMaker Processing is by running a Jupyter notebook. You can follow the `Getting Started with Amazon SageMaker`_ guide to start running notebooks on Amazon SageMaker.

.. _Getting Started with Amazon SageMaker: https://docs.aws.amazon.com/sagemaker/latest/dg/gs.html

You can run notebooks on Amazon SageMaker that demonstrate end-to-end examples of using processing jobs to perform data pre-processing, feature engineering and model evaluation steps. See `Learn More`_ at the bottom of this page for more in-depth information.


Data Pre-Processing and Model Evaluation with Scikit-Learn
==================================================================

You can run a Scikit-Learn script to do data processing on SageMaker using the `SKLearnProcessor`_ class.

.. _SKLearnProcessor: https://sagemaker.readthedocs.io/en/stable/sagemaker.sklearn.html#sagemaker.sklearn.processing.SKLearnProcessor

You first create a ``SKLearnProcessor``

.. code:: python

    from sagemaker.sklearn.processing import SKLearnProcessor

    sklearn_processor = SKLearnProcessor(framework_version='0.20.0',
                                     role='[Your SageMaker-compatible IAM role]',
                                     instance_type='ml.m5.xlarge',
                                     instance_count=1)

Then you can run a Scikit-Learn script ``preprocessing.py`` in a processing job. In this example, our script takes one input from S3 and one command-line argument, processes the data, then splits the data into two datasets for output. When the job is finished, we can retrive the output from S3.

.. code:: python

    from sagemaker.processing import ProcessingInput, ProcessingOutput

    sklearn_processor.run(code='preprocessing.py',
                      inputs=[ProcessingInput(
                        source='s3://your-bucket/path/to/your/data,
                        destination='/opt/ml/processing/input')],
                      outputs=[ProcessingOutput(output_name='train_data',
                                                source='/opt/ml/processing/train'),
                               ProcessingOutput(output_name='test_data',
                                                source='/opt/ml/processing/test')],
                      arguments=['--train-test-split-ratio', '0.2']
                     )

    preprocessing_job_description = sklearn_processor.jobs[-1].describe()

For an in-depth look, please see the `Scikit-Learn Data Processing and Model Evaluation`_ example notebook.

.. _Scikit-Learn Data Processing and Model Evaluation: https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker_processing/scikit_learn_data_processing_and_model_evaluation/scikit_learn_data_processing_and_model_evaluation.ipynb


Data Pre-Processing with Spark
==============================

You can use the `ScriptProcessor`_ class to run a script in a processing container, including your own container.

.. _ScriptProcessor: https://sagemaker.readthedocs.io/en/stable/processing.html#sagemaker.processing.ScriptProcessor

This example shows how you can run a processing job inside of a container that can run a Spark script called ``preprocess.py`` by invoking a command ``/opt/program/submit`` inside the container.

.. code:: python

    from sagemaker.processing import ScriptProcessor, ProcessingInput

    spark_processor = ScriptProcessor(base_job_name='spark-preprocessor',
                                  image_uri='<ECR repository URI to your Spark processing image>',
                                  command=['/opt/program/submit'],
                                  role=role,
                                  instance_count=2,
                                  instance_type='ml.r5.xlarge',
                                  max_runtime_in_seconds=1200,
                                  env={'mode': 'python'})

    spark_processor.run(code='preprocess.py',
                    arguments=['s3_input_bucket', bucket,
                               's3_input_key_prefix', input_prefix,
                               's3_output_bucket', bucket,
                               's3_output_key_prefix', input_preprocessed_prefix],
                    logs=False)

For an in-depth look, please see the `Feature Transformation with Spark`_ example notebook.

.. _Feature Transformation with Spark: https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker_processing/feature_transformation_with_sagemaker_processing/feature_transformation_with_sagemaker_processing.ipynb


Learn More
==========

Processing class documentation
------------------------------

- ``Processor``: https://sagemaker.readthedocs.io/en/stable/processing.html#sagemaker.processing.Processor
- ``ScriptProcessor``: https://sagemaker.readthedocs.io/en/stable/processing.html#sagemaker.processing.ScriptProcessor
- ``SKLearnProcessor``: https://sagemaker.readthedocs.io/en/stable/sagemaker.sklearn.html#sagemaker.sklearn.processing.SKLearnProcessor
- ``ProcessingInput``: https://sagemaker.readthedocs.io/en/stable/processing.html#sagemaker.processing.ProcessingInput
- ``ProcessingOutput``: https://sagemaker.readthedocs.io/en/stable/processing.html#sagemaker.processing.ProcessingOutput
- ``ProcessingJob``: https://sagemaker.readthedocs.io/en/stable/processing.html#sagemaker.processing.ProcessingJob


Further documentation
---------------------

- Processing class documentation: https://sagemaker.readthedocs.io/en/stable/processing.html
- ​​AWS Documentation: https://docs.aws.amazon.com/sagemaker/latest/dg/processing-job.html
- AWS Notebook examples: https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker_processing
- Processing API documentation: https://docs.aws.amazon.com/sagemaker/latest/dg/API_CreateProcessingJob.html
- Processing container specification: https://docs.aws.amazon.com/sagemaker/latest/dg/build-your-own-processing-container.html
