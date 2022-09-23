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
   * - tensorflow-tc-bert-en-cased-L-12-H-768-A-12-2
     - True
     - 1.1.3
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/2>`__
   * - tensorflow-tc-bert-en-cased-L-24-H-1024-A-16-2
     - True
     - 1.1.3
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_cased_L-24_H-1024_A-16/2>`__
   * - tensorflow-tc-bert-en-uncased-L-12-H-768-A-12-2
     - True
     - 1.1.3
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2>`__
   * - tensorflow-tc-bert-en-wwm-cased-L-24-H-1024-A-16-2
     - True
     - 1.1.3
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_wwm_cased_L-24_H-1024_A-16/2>`__
   * - tensorflow-tc-bert-en-wwm-uncased-L-24-H-1024-A-16-2
     - True
     - 1.1.3
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_wwm_uncased_L-24_H-1024_A-16/2>`__
   * - tensorflow-tc-bert-multi-cased-L-12-H-768-A-12-2
     - True
     - 1.1.3
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/2>`__
   * - tensorflow-tc-electra-base-1
     - True
     - 1.1.3
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/google/electra_base/1>`__
   * - tensorflow-tc-electra-small-1
     - True
     - 1.1.3
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/google/electra_small/1>`__
   * - tensorflow-tc-experts-bert-pubmed-1
     - True
     - 1.1.3
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/google/experts/bert/pubmed/1>`__
   * - tensorflow-tc-experts-bert-wiki-books-1
     - True
     - 1.1.3
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/google/experts/bert/wiki_books/1>`__
