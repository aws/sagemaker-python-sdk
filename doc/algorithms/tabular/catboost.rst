############
CatBoost
############


`CatBoost <https://catboost.ai/>`__ is a popular and high-performance open-source implementation of the Gradient Boosting Decision Tree (GBDT)
algorithm. GBDT is a supervised learning algorithm that attempts to accurately predict a target variable by combining an ensemble of
estimates from a set of simpler and weaker models.

CatBoost introduces two critical algorithmic advances to GBDT:

* The implementation of ordered boosting, a permutation-driven alternative to the classic algorithm

* An innovative algorithm for processing categorical features

Both techniques were created to fight a prediction shift caused by a special kind of target leakage present in all currently existing
implementations of gradient boosting algorithms.

The following table outlines a variety of sample notebooks that address different use cases of Amazon SageMaker CatBoost algorithm.

.. list-table::
   :widths: 25 25
   :header-rows: 1

   * - Notebook Title
     - Description
   * - `Tabular classification with Amazon SageMaker LightGBM and CatBoost algorithm <https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/lightgbm_catboost_tabular/Amazon_Tabular_Classification_LightGBM_CatBoost.ipynb>`__
     - This notebook demonstrates the use of the Amazon SageMaker CatBoost algorithm to train and host a tabular classification model.
   * - `Tabular regression with Amazon SageMaker LightGBM and CatBoost algorithm <https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/lightgbm_catboost_tabular/Amazon_Tabular_Regression_LightGBM_CatBoost.ipynb>`__
     - This notebook demonstrates the use of the Amazon SageMaker CatBoost algorithm to train and host a tabular regression model.

For instructions on how to create and access Jupyter notebook instances that you can use to run the example in SageMaker, see
`Use Amazon SageMaker Notebook Instances <https://docs.aws.amazon.com/sagemaker/latest/dg/nbi.html>`__. After you have created a notebook
instance and opened it, choose the SageMaker Examples tab to see a list of all of the SageMaker samples. To open a notebook, choose its
Use tab and choose Create copy.

For detailed documentation, please refer to the `Sagemaker CatBoost Algorithm <https://docs.aws.amazon.com/sagemaker/latest/dg/catboost.html>`__.
