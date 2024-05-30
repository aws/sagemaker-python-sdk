##################################
Text Classification - TensorFlow
##################################

This is a supervised text classification algorithm which supports fine-tuning of many pre-trained models available in Tensorflow Hub. The following
`sample notebook <https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/jumpstart_text_classification/Amazon_JumpStart_Text_Classification.ipynb>`__
demonstrates how to use the Sagemaker Python SDK for Text Classification for using these algorithms.

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
   * - tensorflow-tc-albert-en-base
     - True
     - 2.0.0
     - 2.189.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/albert_en_base/2>`__
   * - tensorflow-tc-bert-en-cased-L-12-H-768-A-12-2
     - True
     - 3.0.0
     - 2.189.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3>`__
