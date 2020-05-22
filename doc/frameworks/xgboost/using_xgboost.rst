#########################################
Use XGBoost with the SageMaker Python SDK
#########################################

.. contents::

eXtreme Gradient Boosting (XGBoost) is a popular and efficient machine learning algorithm used for regression and classification tasks on tabular datasets.
It implements a technique known as gradient boosting on trees, which performs remarkably well in machine learning competitions.

Amazon SageMaker supports two ways to use the XGBoost algorithm:

 * XGBoost built-in algorithm
 * XGBoost open source algorithm

The XGBoost open source algorithm provides the following benefits over the built-in algorithm:

* Latest version - The open source XGBoost algorithm typically supports a more recent version of XGBoost.
  To see the XGBoost version that is currently supported,
  see `XGBoost SageMaker Estimators and Models <https://github.com/aws/sagemaker-python-sdk/tree/master/src/sagemaker/xgboost#xgboost-sagemaker-estimators-and-models>`__.
* Flexibility - Take advantage of the full range of XGBoost functionality, such as cross-validation support.
  You can add custom pre- and post-processing logic and run additional code after training.
* Scalability - The XGBoost open source algorithm has a more efficient implementation of distributed training,
  which enables it to scale out to more instances and reduce out-of-memory errors.
* Extensibility - Because the open source XGBoost container is open source,
  you can extend the container to install additional libraries and change the version of XGBoost that the container uses.
  For an example notebook that shows how to extend SageMaker containers, see `Extending our PyTorch containers <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/pytorch_extending_our_containers/pytorch_extending_our_containers.ipynb>`__.


***********************************
Use XGBoost as a Built-in Algortihm
***********************************

Amazon SageMaker provides XGBoost as a built-in algorithm that you can use like other built-in algorithms.
Using the built-in algorithm version of XGBoost is simpler than using the open source version, because you don't have to write a training script.
If you don't need the features and flexibility of open source XGBoost, consider using the built-in version.
For information about using the Amazon SageMaker XGBoost built-in algorithm, see `XGBoost Algorithm <https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html>`__
in the *Amazon SageMaker Developer Guide*.

*************************************
Use the Open Source XGBoost Algorithm
*************************************

If you want the flexibility and additional features that it provides, use the SageMaker open source XGBoost algorithm.

For which XGBoost versions are supported, see `the AWS documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html>`_.
We recommend that you use the latest supported version because that's where we focus most of our development efforts.

For a complete example of using the open source XGBoost algorithm, see the sample notebook at
https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/xgboost_abalone/xgboost_abalone_dist_script_mode.ipynb.

For more information about XGBoost, see `the XGBoost documentation <https://xgboost.readthedocs.io/en/latest>`_.

Train a Model with Open Source XGBoost
======================================

To train a model by using the Amazon SageMaker open source XGBoost algorithm:

.. |create xgboost estimator| replace:: Create a ``sagemaker.xgboost.XGBoost estimator``
.. _create xgboost estimator: #create-an-estimator

.. |call fit| replace:: Call the estimator's ``fit`` method
.. _call fit: #call-the-fit-method

1. `Prepare a training script <#prepare-a-training-script>`_
2. |create xgboost estimator|_
3. |call fit|_

Prepare a Training Script
-------------------------

A typical training script loads data from the input channels, configures training with hyperparameters, trains a model,
and saves a model to ``model_dir`` so that it can be hosted later.
Hyperparameters are passed to your script as arguments and can be retrieved with an ``argparse.ArgumentParser`` instance.
For information about ``argparse.ArgumentParser``, see `argparse <https://docs.python.org/3/library/argparse.html>`__ in the Python documentation.


For a complete example of an XGBoost training script, see https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/xgboost_abalone/abalone.py.

The training script is very similar to a training script you might run outside of Amazon SageMaker,
but you can access useful properties about the training environment through various environment variables, including the following:

* ``SM_MODEL_DIR``: A string that represents the path where the training job writes the model artifacts to.
  After training, artifacts in this directory are uploaded to Amazon S3 for model hosting.
* ``SM_NUM_GPUS``: An integer representing the number of GPUs available to the host.
* ``SM_CHANNEL_XXXX``: A string that represents the path to the directory that contains the input data for the specified channel.
  For example, if you specify two input channels in the MXNet estimator's ``fit`` call, named 'train' and 'test', the environment variables ``SM_CHANNEL_TRAIN`` and ``SM_CHANNEL_TEST`` are set.
* ``SM_HPS``: A JSON dump of the hyperparameters preserving JSON types (boolean, integer, etc.)

For the exhaustive list of available environment variables, see the `SageMaker Containers documentation <https://github.com/aws/sagemaker-containers#list-of-provided-environment-variables-by-sagemaker-containers>`__.

Let's look at the main elements of the script. Starting with the ``__main__`` guard,
use a parser to read the hyperparameters passed to the estimator when creating the training job.
These hyperparameters are made available as arguments to our input script.
We also parse a number of Amazon SageMaker-specific environment variables to get information about the training environment,
such as the location of input data and location where we want to save the model.

.. code:: python

    if __name__ == '__main__':
        parser = argparse.ArgumentParser()

        # Hyperparameters are described here
        parser.add_argument('--num_round', type=int)
        parser.add_argument('--max_depth', type=int, default=5)
        parser.add_argument('--eta', type=float, default=0.2)
        parser.add_argument('--objective', type=str, default='reg:squarederror')

        # SageMaker specific arguments. Defaults are set in the environment variables.
        parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
        parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
        parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])

        args = parser.parse_args()

        train_hp = {
            'max_depth': args.max_depth,
            'eta': args.eta,
            'gamma': args.gamma,
            'min_child_weight': args.min_child_weight,
            'subsample': args.subsample,
            'silent': args.silent,
            'objective': args.objective
        }

        dtrain = xgb.DMatrix(args.train)
        dval = xgb.DMatrix(args.validation)
        watchlist = [(dtrain, 'train'), (dval, 'validation')] if dval is not None else [(dtrain, 'train')]

        callbacks = []
        prev_checkpoint, n_iterations_prev_run = add_checkpointing(callbacks)
        # If checkpoint is found then we reduce num_boost_round by previously run number of iterations

        bst = xgb.train(
            params=train_hp,
            dtrain=dtrain,
            evals=watchlist,
            num_boost_round=(args.num_round - n_iterations_prev_run),
            xgb_model=prev_checkpoint,
            callbacks=callbacks
        )

        # Save the model to the location specified by ``model_dir``
        model_location = args.model_dir + '/xgboost-model'
        pkl.dump(bst, open(model_location, 'wb'))
        logging.info("Stored trained model at {}".format(model_location))

Create an Estimator
-------------------
After you create your training script, create an instance of the :class:`sagemaker.xgboost.estimator.XGBoost` estimator.
Pass an IAM role that has the permissions necessary to run an Amazon SageMaker training job,
the type and number of instances to use for the training job,
and a dictionary of the hyperparameters to pass to the training script.

.. code::

    from sagemaker.xgboost.estimator import XGBoost

    xgb_estimator = XGBoost(
        entry_point="abalone.py",
        hyperparameters=hyperparameters,
        role=role,
        train_instance_count=1,
        train_instance_type="ml.m5.2xlarge",
        framework_version="0.90-1",
    )


Call the fit Method
-------------------

After you create an estimator, call the ``fit`` method to run the training job.

.. code::

    xgb_script_mode_estimator.fit({"train": train_input})



Deploy Open Source XGBoost Models
=================================

After the training job finishes, call the ``deploy`` method of the estimator to create a predictor that you can use to get inferences from your trained model.

.. code::

    predictor = xgb_script_mode_estimator.deploy(initial_instance_count=1, instance_type="ml.m5.xlarge")
    test_data = xgboost.DMatrix('/path/to/data')
    predictor.predict(test_data)

Customize inference
-------------------

In your inference script, which can be either in the same file as your training script or in a separate file,
you can customize the inference behavior by implementing the following functions:
* ``input_fn`` - how input data is handled
* ``predict_fn`` - how the model is invoked
* ``output_fn`` - How the response data is handled

These functions are optional. If you want to use the default implementations, do not implement them in your training script.


*************************
SageMaker XGBoost Classes
*************************

For information about the SageMaker Python SDK XGBoost classes, see the following topics:

* :class:`sagemaker.xgboost.estimator.XGBoost`
* :class:`sagemaker.xgboost.model.XGBoostModel`
* :class:`sagemaker.xgboost.model.XGBoostPredictor`

***********************************
SageMaker XGBoost Docker Containers
***********************************

For information about SageMaker XGBoost Docker container and its dependencies, see `SageMaker XGBoost Container <https://github.com/aws/sagemaker-xgboost-container>`_.
