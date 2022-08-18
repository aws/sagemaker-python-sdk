###############
TabTransformer
###############

`TabTransformer <https://arxiv.org/abs/2012.06678>`__ is a novel deep tabular data modeling architecture for supervised learning. The TabTransformer architecture is built on self-attention-based Transformers.
The Transformer layers transform the embeddings of categorical features into robust contextual embeddings to achieve higher prediction accuracy. Furthermore, the contextual embeddings learned from TabTransformer
are highly robust against both missing and noisy data features, and provide better interpretability.


The following table outlines a variety of sample notebooks that address different use cases of Amazon SageMaker TabTransformer algorithm.

.. list-table::
   :widths: 25 25
   :header-rows: 1

   * - Notebook Title
     - Description
   * - `Tabular classification with Amazon SageMaker TabTransformer algorithm <https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/tabtransformer_tabular/Amazon_Tabular_Classification_TabTransformer.ipynb>`__
     - This notebook demonstrates the use of the Amazon SageMaker TabTransformer algorithm to train and host a tabular classification model.
   * - `Tabular regression with Amazon SageMaker TabTransformer algorithm <https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/tabtransformer_tabular/Amazon_Tabular_Regression_TabTransformer.ipynb>`__
     - This notebook demonstrates the use of the Amazon SageMaker TabTransformer algorithm to train and host a tabular regression model.

For instructions on how to create and access Jupyter notebook instances that you can use to run the example in SageMaker, see
`Use Amazon SageMaker Notebook Instances <https://docs.aws.amazon.com/sagemaker/latest/dg/nbi.html>`__. After you have created a notebook
instance and opened it, choose the SageMaker Examples tab to see a list of all of the SageMaker samples. To open a notebook, choose its
Use tab and choose Create copy.

For detailed documentation, please refer to the `Sagemaker TabTransformer Algorithm <https://docs.aws.amazon.com/sagemaker/latest/dg/tabtransformer.html>`__.
