#####################################
Machine Translation - HuggingFace
#####################################


This is a supervised machine translation algorithm which supports many pre-trained models available in Hugging Face. The following
`sample notebook <https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/jumpstart_machine_translation/Amazon_JumpStart_Machine_Translation.ipynb>`__
demonstrates how to use the Sagemaker Python SDK for Machine Translation for using these algorithms.

For detailed documentation please refer :ref:`Use Built-in Algorithms with Pre-trained Models in SageMaker Python SDK <built-in-algos>`.

.. list-table:: Available Models
   :widths: 50 20 20 20 20
   :header-rows: 1
   :class: datatable

   * - Model ID
     - Fine Tunable?
     - Latest Version
     - Min SDK Version
     - Source
   * - huggingface-translation-opus-mt-en-es
     - False
     - 1.1.0
     - 2.75.0
     - `HuggingFace <https://huggingface.co/Helsinki-NLP/opus-mt-en-es>`__
   * - huggingface-translation-opus-mt-en-vi
     - False
     - 1.1.0
     - 2.75.0
     - `HuggingFace <https://huggingface.co/Helsinki-NLP/opus-mt-en-vi>`__
   * - huggingface-translation-t5-base
     - False
     - 1.1.0
     - 2.75.0
     - `HuggingFace <https://huggingface.co/t5-base>`__
   * - huggingface-translation-t5-large
     - False
     - 1.1.0
     - 2.75.0
     - `HuggingFace <https://huggingface.co/t5-large>`__
   * - huggingface-translation-t5-small
     - False
     - 1.1.0
     - 2.75.0
     - `HuggingFace <https://huggingface.co/t5-small>`__
