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
     - 1.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/albert_en_base/2>`__
   * - tensorflow-tc-bert-en-cased-L-12-H-768-A-12-2
     - True
     - 2.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3>`__
   * - tensorflow-tc-bert-en-cased-L-24-H-1024-A-16-2
     - True
     - 2.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_cased_L-24_H-1024_A-16/3>`__
   * - tensorflow-tc-bert-en-uncased-L-12-H-768-A-12-2
     - True
     - 2.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3>`__
   * - tensorflow-tc-bert-en-uncased-L-24-H-1024-A-16-2
     - True
     - 1.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/3>`__
   * - tensorflow-tc-bert-en-wwm-cased-L-24-H-1024-A-16-2
     - True
     - 2.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_wwm_cased_L-24_H-1024_A-16/3>`__
   * - tensorflow-tc-bert-en-wwm-uncased-L-24-H-1024-A-16-2
     - True
     - 2.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_wwm_uncased_L-24_H-1024_A-16/3>`__
   * - tensorflow-tc-bert-multi-cased-L-12-H-768-A-12-2
     - True
     - 2.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3>`__
   * - tensorflow-tc-electra-base-1
     - True
     - 2.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/electra_base/2>`__
   * - tensorflow-tc-electra-small-1
     - True
     - 2.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/electra_small/2>`__
   * - tensorflow-tc-experts-bert-pubmed-1
     - True
     - 2.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/experts/bert/pubmed/2>`__
   * - tensorflow-tc-experts-bert-wiki-books-1
     - True
     - 2.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/experts/bert/wiki_books/2>`__
   * - tensorflow-tc-small-bert-bert-en-uncased-L-10-H-128-A-2
     - True
     - 1.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1>`__
   * - tensorflow-tc-small-bert-bert-en-uncased-L-10-H-256-A-4
     - True
     - 1.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1>`__
   * - tensorflow-tc-small-bert-bert-en-uncased-L-10-H-512-A-8
     - True
     - 1.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1>`__
   * - tensorflow-tc-small-bert-bert-en-uncased-L-10-H-768-A-12
     - True
     - 1.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1>`__
   * - tensorflow-tc-small-bert-bert-en-uncased-L-12-H-128-A-2
     - True
     - 1.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1>`__
   * - tensorflow-tc-small-bert-bert-en-uncased-L-12-H-256-A-4
     - True
     - 1.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1>`__
   * - tensorflow-tc-small-bert-bert-en-uncased-L-12-H-512-A-8
     - True
     - 1.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1>`__
   * - tensorflow-tc-small-bert-bert-en-uncased-L-12-H-768-A-12
     - True
     - 1.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1>`__
   * - tensorflow-tc-small-bert-bert-en-uncased-L-2-H-128-A-2
     - True
     - 1.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1>`__
   * - tensorflow-tc-small-bert-bert-en-uncased-L-2-H-256-A-4
     - True
     - 1.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1>`__
   * - tensorflow-tc-small-bert-bert-en-uncased-L-2-H-512-A-8
     - True
     - 1.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1>`__
   * - tensorflow-tc-small-bert-bert-en-uncased-L-2-H-768-A-12
     - True
     - 1.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1>`__
   * - tensorflow-tc-small-bert-bert-en-uncased-L-4-H-128-A-2
     - True
     - 1.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1>`__
   * - tensorflow-tc-small-bert-bert-en-uncased-L-4-H-256-A-4
     - True
     - 1.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1>`__
   * - tensorflow-tc-small-bert-bert-en-uncased-L-4-H-512-A-8
     - True
     - 1.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1>`__
   * - tensorflow-tc-small-bert-bert-en-uncased-L-4-H-768-A-12
     - True
     - 1.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1>`__
   * - tensorflow-tc-small-bert-bert-en-uncased-L-6-H-128-A-2
     - True
     - 1.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1>`__
   * - tensorflow-tc-small-bert-bert-en-uncased-L-6-H-256-A-4
     - True
     - 1.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1>`__
   * - tensorflow-tc-small-bert-bert-en-uncased-L-6-H-512-A-8
     - True
     - 1.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1>`__
   * - tensorflow-tc-small-bert-bert-en-uncased-L-6-H-768-A-12
     - True
     - 1.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1>`__
   * - tensorflow-tc-small-bert-bert-en-uncased-L-8-H-128-A-2
     - True
     - 1.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1>`__
   * - tensorflow-tc-small-bert-bert-en-uncased-L-8-H-256-A-4
     - True
     - 1.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1>`__
   * - tensorflow-tc-small-bert-bert-en-uncased-L-8-H-512-A-8
     - True
     - 1.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1>`__
   * - tensorflow-tc-small-bert-bert-en-uncased-L-8-H-768-A-12
     - True
     - 1.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1>`__
   * - tensorflow-tc-talking-heads-base
     - True
     - 1.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1>`__
   * - tensorflow-tc-talking-heads-large
     - True
     - 1.0.1
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_large/1>`__
