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
     - 1.2.1
     - 2.100.0
     - `GluonCV <https://nlp.gluon.ai/master/_modules/gluonnlp/models/roberta.html>`__
   * - mxnet-tcembedding-robertafin-base-wiki-uncased
     - False
     - 1.2.1
     - 2.100.0
     - `GluonCV <https://nlp.gluon.ai/master/_modules/gluonnlp/models/roberta.html>`__
   * - mxnet-tcembedding-robertafin-large-uncased
     - False
     - 1.2.1
     - 2.100.0
     - `GluonCV <https://nlp.gluon.ai/master/_modules/gluonnlp/models/roberta.html>`__
   * - mxnet-tcembedding-robertafin-large-wiki-uncased
     - False
     - 1.2.1
     - 2.100.0
     - `GluonCV <https://nlp.gluon.ai/master/_modules/gluonnlp/models/roberta.html>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-10-H-128-A-2-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-10-H-256-A-4-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-10-H-512-A-8-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-10-H-768-A-12-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-12-H-128-A-2-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-12-H-256-A-4
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-12-H-512-A-8-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-12-H-768-A-12-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-12-H-768-A-12-4
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-2-H-128-A-2-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-2-H-256-A-4
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-2-H-512-A-8-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-2-H-768-A-12-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-4-H-128-A-2-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-4-H-256-A-4-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-4-H-512-A-8-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-4-H-768-A-12-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-6-H-128-A-2-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-6-H-256-A-4
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-6-H-512-A-8-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-6-H-768-A-12-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-8-H-256-A-4-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-8-H-512-A-8-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-8-H-768-A-12-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/2>`__
   * - tensorflow-tcembedding-bert-wiki-books-mnli-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/google/experts/bert/wiki_books/mnli/2>`__
   * - tensorflow-tcembedding-bert-wiki-books-sst2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/google/experts/bert/wiki_books/sst2/2>`__
   * - tensorflow-tcembedding-talkheads-ggelu-bert-en-base-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/2>`__
   * - tensorflow-tcembedding-talkheads-ggelu-bert-en-large-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_large/2>`__
   * - tensorflow-tcembedding-universal-sentence-encoder-cmlm-en-base-1
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/1>`__
   * - tensorflow-tcembedding-universal-sentence-encoder-cmlm-en-large-1
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-large/1>`__

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
     - 1.2.1
     - 2.100.0
     - `GluonCV <https://nlp.gluon.ai/master/_modules/gluonnlp/models/roberta.html>`__
   * - mxnet-tcembedding-robertafin-base-wiki-uncased
     - False
     - 1.2.1
     - 2.100.0
     - `GluonCV <https://nlp.gluon.ai/master/_modules/gluonnlp/models/roberta.html>`__
   * - mxnet-tcembedding-robertafin-large-uncased
     - False
     - 1.2.1
     - 2.100.0
     - `GluonCV <https://nlp.gluon.ai/master/_modules/gluonnlp/models/roberta.html>`__
   * - mxnet-tcembedding-robertafin-large-wiki-uncased
     - False
     - 1.2.1
     - 2.100.0
     - `GluonCV <https://nlp.gluon.ai/master/_modules/gluonnlp/models/roberta.html>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-10-H-128-A-2-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-10-H-256-A-4-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-10-H-512-A-8-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-10-H-768-A-12-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-12-H-128-A-2-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-12-H-256-A-4
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-12-H-512-A-8-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-12-H-768-A-12-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-12-H-768-A-12-4
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-2-H-128-A-2-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-2-H-256-A-4
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-2-H-512-A-8-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-2-H-768-A-12-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-4-H-128-A-2-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-4-H-256-A-4-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-4-H-512-A-8-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-4-H-768-A-12-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-6-H-128-A-2-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-6-H-256-A-4
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-6-H-512-A-8-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-6-H-768-A-12-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-8-H-256-A-4-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-8-H-512-A-8-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/2>`__
   * - tensorflow-tcembedding-bert-en-uncased-L-8-H-768-A-12-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/2>`__
   * - tensorflow-tcembedding-bert-wiki-books-mnli-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/google/experts/bert/wiki_books/mnli/2>`__
   * - tensorflow-tcembedding-bert-wiki-books-sst2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/google/experts/bert/wiki_books/sst2/2>`__
   * - tensorflow-tcembedding-talkheads-ggelu-bert-en-base-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/2>`__
   * - tensorflow-tcembedding-talkheads-ggelu-bert-en-large-2
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_large/2>`__
   * - tensorflow-tcembedding-universal-sentence-encoder-cmlm-en-base-1
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/1>`__
   * - tensorflow-tcembedding-universal-sentence-encoder-cmlm-en-large-1
     - False
     - 1.1.1
     - 2.75.0
     - `Tensorflow Hub <https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-large/1>`__
