####################################
Text Embedding - TensorFlow, MxNet
####################################

This is a supervised text embedding algorithm which supports many pre-trained models available in MXNet and Tensorflow Hub. The following
`sample notebook <https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/jumpstart_text_embedding/Amazon_JumpStart_Text_Embedding.ipynb>`__
demonstrates how to use the Sagemaker Python SDK for Text Embedding for using these algorithms.

For detailed documentation please refer :ref:`Use Built-in Algorithms with Pre-trained Models in SageMaker Python SDK <built-in-algos>`

.. list-table:: Available Models
   :widths: 50 20 20 20 20
   :header-rows: 1
   :class: datatable

   * - Model ID
     - Fine Tunable?
     - Latest Version
     - Min SDK Version
     - Source
   * - mxnet-tcembedding-robertafin-base-uncased
     - False
     - 2.0.0
     - 2.189.0
     - `GluonCV <https://nlp.gluon.ai/master/_modules/gluonnlp/models/roberta.html>`__
   * - mxnet-tcembedding-robertafin-base-wiki-uncased
     - False
     - 2.0.0
     - 2.189.0
     - `GluonCV <https://nlp.gluon.ai/master/_modules/gluonnlp/models/roberta.html>`__
   * - mxnet-tcembedding-robertafin-large-uncased
     - False
     - 2.0.0
     - 2.189.0
     - `GluonCV <https://nlp.gluon.ai/master/_modules/gluonnlp/models/roberta.html>`__
   * - mxnet-tcembedding-robertafin-large-wiki-uncased
     - False
     - 2.0.0
     - 2.189.0
     - `GluonCV <https://nlp.gluon.ai/master/_modules/gluonnlp/models/roberta.html>`__
