############################################
Sentence Pair Classification - TensorFlow
############################################

This is a supervised sentence pair classification algorithm which supports fine-tuning of many pre-trained models available in Tensorflow Hub. The following
`sample notebook <https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/jumpstart_sentence_pair_classification/Amazon_JumpStart_Sentence_Pair_Classification.ipynb>`__
demonstrates how to use the Sagemaker Python SDK for Sentence Pair Classification for using these algorithms.

For detailed documentation please refer `Use Built-in Algorithms with Pre-trained Models in SageMaker Python SDK <https://sagemaker.readthedocs.io/en/stable/overview.html#use-built-in-algorithms-with-pre-trained-models-in-sagemaker-python-sdk>`__

.. list-table:: Available Models
   :widths: 50 20 20 20 20
   :header-rows: 1
   :class: datatable

   * - Model ID
     - Fine Tunable?
     - Latest Version
     - Min SDK Version
     - Source
   * - tensorflow-spc-bert-en-cased-L-12-H-768-A-12-2
     - True
     - 2.0.0
     - 2.189.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/2>`__
   * - tensorflow-spc-bert-en-uncased-L-12-H-768-A-12-2
     - True
     - 2.0.0
     - 2.189.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2>`__
   * - tensorflow-spc-bert-en-uncased-L-24-H-1024-A-16-2
     - True
     - 2.0.0
     - 2.189.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/2>`__
   * - tensorflow-spc-bert-en-wwm-cased-L-24-H-1024-A-16-2
     - True
     - 2.0.0
     - 2.189.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_wwm_cased_L-24_H-1024_A-16/2>`__
   * - tensorflow-spc-bert-en-wwm-uncased-L-24-H-1024-A-16-2
     - True
     - 2.0.0
     - 2.189.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_wwm_uncased_L-24_H-1024_A-16/2>`__
   * - tensorflow-spc-bert-multi-cased-L-12-H-768-A-12-2
     - True
     - 2.0.0
     - 2.189.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/2>`__
   * - tensorflow-spc-electra-base-1
     - True
     - 2.0.0
     - 2.189.0
     - `Tensorflow Hub <https://tfhub.dev/google/electra_base/1>`__
   * - tensorflow-spc-electra-small-1
     - True
     - 2.0.0
     - 2.189.0
     - `Tensorflow Hub <https://tfhub.dev/google/electra_small/1>`__
   * - tensorflow-spc-experts-bert-pubmed-1
     - True
     - 2.0.0
     - 2.189.0
     - `Tensorflow Hub <https://tfhub.dev/google/experts/bert/pubmed/1>`__
   * - tensorflow-spc-experts-bert-wiki-books-1
     - True
     - 2.0.0
     - 2.189.0
     - `Tensorflow Hub <https://tfhub.dev/google/experts/bert/wiki_books/1>`__
