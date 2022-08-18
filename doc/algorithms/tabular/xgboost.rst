############
XGBoost
############

The `XGBoost <https://github.com/dmlc/xgboost>`__ (eXtreme Gradient Boosting) is a popular and efficient open-source implementation of the gradient boosted trees algorithm. Gradient boosting is a supervised learning algorithm that attempts to accurately predict a target variable
by combining an ensemble of estimates from a set of simpler and weaker models. The XGBoost algorithm performs well in machine learning competitions because of its robust handling of a variety of data types, relationships, distributions, and the variety of hyperparameters that you can
fine-tune. You can use XGBoost for regression, classification (binary and multiclass), and ranking problems.

You can use the new release of the XGBoost algorithm either as a Amazon SageMaker built-in algorithm or as a framework to run training scripts in your local environments. This implementation has a smaller memory footprint, better logging, improved hyperparameter validation, and
an expanded set of metrics than the original versions. It provides an XGBoost estimator that executes a training script in a managed XGBoost environment. The current release of SageMaker XGBoost is based on the original XGBoost versions 1.0, 1.2, 1.3, and 1.5.

The following table outlines a variety of sample notebooks that address different use cases of Amazon SageMaker XGBoost algorithm.

.. list-table::
   :widths: 25 25
   :header-rows: 1

   * - Notebook Title
     - Description
   * - `How to Create a Custom XGBoost container? <https://sagemaker-examples.readthedocs.io/en/latest/aws_sagemaker_studio/sagemaker_studio_image_build/xgboost_bring_your_own/Batch_Transform_BYO_XGB.html>`__
     - This notebook shows you how to build a custom XGBoost Container with Amazon SageMaker Batch Transform.
   * - `Regression with XGBoost using Parquet <https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_amazon_algorithms/xgboost_abalone/xgboost_parquet_input_training.html>`__
     - This notebook shows you how to use the Abalone dataset in Parquet to train a XGBoost model.
   * - `How to Train and Host a Multiclass Classification Model? <https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_amazon_algorithms/xgboost_mnist/xgboost_mnist.html>`__
     - This notebook shows how to use the MNIST dataset to train and host a multiclass classification model.
   * - `How to train a Model for Customer Churn Prediction? <https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_applying_machine_learning/xgboost_customer_churn/xgboost_customer_churn.html>`__
     - This notebook shows you how to train a model to Predict Mobile Customer Departure in an effort to identify unhappy customers.
   * - `An Introduction to Amazon SageMaker Managed Spot infrastructure for XGBoost Training <https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_amazon_algorithms/xgboost_abalone/xgboost_managed_spot_training.html>`__
     - This notebook shows you how to use Spot Instances for training with a XGBoost Container.
   * - `How to use Amazon SageMaker Debugger to debug XGBoost Training Jobs? <https://sagemaker-examples.readthedocs.io/en/latest/sagemaker-debugger/xgboost_builtin_rules/xgboost-regression-debugger-rules.html>`__
     - This notebook shows you how to use Amazon SageMaker Debugger to monitor training jobs to detect inconsistencies.
   * - `How to use Amazon SageMaker Debugger to debug XGBoost Training Jobs in Real-Time? <https://sagemaker-examples.readthedocs.io/en/latest/sagemaker-debugger/xgboost_realtime_analysis/xgboost-realtime-analysis.html>`__
     - This notebook shows you how to use the MNIST dataset and Amazon SageMaker Debugger to perform real-time analysis of XGBoost training jobs while training jobs are running.

For instructions on how to create and access Jupyter notebook instances that you can use to run the example in SageMaker, see
`Use Amazon SageMaker Notebook Instances <https://docs.aws.amazon.com/sagemaker/latest/dg/nbi.html>`__. After you have created a notebook
instance and opened it, choose the SageMaker Examples tab to see a list of all of the SageMaker samples. To open a notebook, choose its
Use tab and choose Create copy.

For detailed documentation, please refer to the `Sagemaker XGBoost Algorithm <https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html>`__.
