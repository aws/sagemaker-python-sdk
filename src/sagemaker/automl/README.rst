SageMaker Autopilot
===================

Amazon SageMaker Autopilot is an automated machine learning solution (commonly referred to as "AutoML") for tabular
datasets. It automatically trains and tunes the best machine learning models for classification or regression based
on your data, and hosts a series of models on an Inference Pipeline.

SageMaker AutoML Class
~~~~~~~~~~~~~~~~~~~~~~

The SageMaker ``AutoML`` class is similar to a SageMaker ``Estimator`` where you define the attributes of an AutoML
job and feed input data to start the job.

Here's a simple example of using the ``AutoML`` object:

.. code:: python

    from sagemaker import AutoML

    auto_ml = AutoML(
        role="sagemaker-execution-role",
        target_attribute_name="y",
        sagemaker_session=sagemaker_session,
    )
    auto_ml.fit(inputs=inputs)


The above code starts an AutoML job (data processing, training, tuning) and outputs a maximum of 500 candidates by
default. You can modify the number of output candidates by specifying ``max_candidates`` in the constructor. The AutoML
job will figure out the problem type (BinaryClassification, MulticlassClassification, Regression), but you can also
specify the problem type by setting ``problem_type`` in the constructor. Other configurable settings include security
settings, time limits, job objectives, tags, etc.

After an AutoML job is done, there are a few things that you can do with the result.

#. Describe the AutoML job: ``describe_auto_ml_job()`` will give you an overview of the AutoML job, information
includes job name, best candidate, input/output locations, problem type, objective metrics, etc.

#. Get the best candidate: ``best_candidate()`` allows you to get the best candidate of an AutoML job. You can view the
best candidate's step jobs, inference containers and other information like objective metrics.

#. List all the candidates: ``list_candidates()`` gives you all the candidates (up to the maximum number) of an AutoML
job. By calling this method, you can view and compare the candidates.

#. Deploy the best candidate (or any given candidate): ``deploy()`` by default will deploy the best candidate to an
inference pipeline. But you can also specify a candidate to deploy through ``candidate`` parameter.

For more information about ``AutoML`` parameters, please refer to: https://sagemaker.readthedocs.io/en/stable/automl.html

SageMaker CandidateEstimator Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SageMaker ``CandidateEstimator`` class converts a dictionary with AutoML candidate information to an object that
allows you to re-run the candidate's step jobs.

The simplest re-run is to feed a new dataset but reuse all other configurations from the candidate:

.. code:: python

    candidate_estimator = CandidateEstimator(candidate_dict)
    inputs = new_inputs
    candidate_estimator.fit(inputs=inputs)

If you want to have more control over the step jobs of the candidate, you can call ``get_steps()`` and construct
training/tuning jobs by yourself.

For more information about ``CandidateEstimator`` parameters, please refer to: https://sagemaker.readthedocs.io/en/stable/automl.html#sagemaker.automl.candidate_estimator.CandidateEstimator
