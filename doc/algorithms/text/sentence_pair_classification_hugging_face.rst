############################################
Sentence Pair Classification - HuggingFace
############################################

This is a supervised sentence pair classification algorithm which supports fine-tuning of many pre-trained models available in Hugging Face. The following
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
   * - huggingface-spc-bert-base-cased
     - True
     - 2.0.0
     - 2.189.0
     - `HuggingFace <https://huggingface.co/bert-base-cased>`__
   * - huggingface-spc-bert-base-multilingual-cased
     - True
     - 2.0.0
     - 2.189.0
     - `HuggingFace <https://huggingface.co/bert-base-multilingual-cased>`__
   * - huggingface-spc-bert-base-multilingual-uncased
     - True
     - 2.0.0
     - 2.189.0
     - `HuggingFace <https://huggingface.co/bert-base-multilingual-uncased>`__
   * - huggingface-spc-bert-base-uncased
     - True
     - 2.0.0
     - 2.189.0
     - `HuggingFace <https://huggingface.co/bert-base-uncased>`__
   * - huggingface-spc-bert-large-cased
     - True
     - 2.0.0
     - 2.189.0
     - `HuggingFace <https://huggingface.co/bert-large-cased>`__
   * - huggingface-spc-bert-large-cased-whole-word-masking
     - True
     - 2.0.0
     - 2.189.0
     - `HuggingFace <https://huggingface.co/bert-large-cased-whole-word-masking>`__
   * - huggingface-spc-bert-large-uncased
     - True
     - 2.0.0
     - 2.189.0
     - `HuggingFace <https://huggingface.co/bert-large-uncased>`__
   * - huggingface-spc-bert-large-uncased-whole-word-masking
     - True
     - 2.0.0
     - 2.189.0
     - `HuggingFace <https://huggingface.co/bert-large-uncased-whole-word-masking>`__
   * - huggingface-spc-distilbert-base-cased
     - True
     - 2.0.0
     - 2.189.0
     - `HuggingFace <https://huggingface.co/distilbert-base-cased>`__
   * - huggingface-spc-distilbert-base-multilingual-cased
     - True
     - 2.0.0
     - 2.189.0
     - `HuggingFace <https://huggingface.co/distilbert-base-multilingual-cased>`__
   * - huggingface-spc-distilbert-base-uncased
     - True
     - 2.0.0
     - 2.189.0
     - `HuggingFace <https://huggingface.co/distilbert-base-uncased>`__
   * - huggingface-spc-distilroberta-base
     - True
     - 2.0.0
     - 2.189.0
     - `HuggingFace <https://huggingface.co/distilroberta-base>`__
   * - huggingface-spc-roberta-base
     - True
     - 2.0.0
     - 2.189.0
     - `HuggingFace <https://huggingface.co/roberta-base>`__
   * - huggingface-spc-roberta-base-openai-detector
     - True
     - 2.0.0
     - 2.189.0
     - `HuggingFace <https://huggingface.co/roberta-base-openai-detector>`__
   * - huggingface-spc-roberta-large
     - True
     - 2.0.0
     - 2.189.0
     - `HuggingFace <https://huggingface.co/roberta-large>`__
   * - huggingface-spc-roberta-large-openai-detector
     - True
     - 2.0.0
     - 2.189.0
     - `HuggingFace <https://huggingface.co/roberta-large-openai-detector>`__
   * - huggingface-spc-xlm-clm-ende-1024
     - True
     - 2.0.0
     - 2.189.0
     - `HuggingFace <https://huggingface.co/xlm-clm-ende-1024>`__
   * - huggingface-spc-xlm-mlm-ende-1024
     - True
     - 2.0.0
     - 2.189.0
     - `HuggingFace <https://huggingface.co/xlm-mlm-ende-1024>`__
   * - huggingface-spc-xlm-mlm-enro-1024
     - True
     - 2.0.0
     - 2.189.0
     - `HuggingFace <https://huggingface.co/xlm-mlm-enro-1024>`__
   * - huggingface-spc-xlm-mlm-tlm-xnli15-1024
     - True
     - 2.0.0
     - 2.189.0
     - `HuggingFace <https://huggingface.co/xlm-mlm-tlm-xnli15-1024>`__
   * - huggingface-spc-xlm-mlm-xnli15-1024
     - True
     - 2.0.0
     - 2.189.0
     - `HuggingFace <https://huggingface.co/xlm-mlm-xnli15-1024>`__
