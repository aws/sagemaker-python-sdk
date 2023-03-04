############################################
Text Generation - HuggingFace
############################################

This is a supervised text generation algorithm which supports many pre-trained models available in Hugging Face. The following
`sample notebook <https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/jumpstart_text_generation/Amazon_JumpStart_Text_Generation.ipynb>`__
demonstrates how to use the Sagemaker Python SDK for Text Generation for using these algorithms.

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
   * - huggingface-textgeneration-bloom-1b1
     - False
     - 1.0.1
     - 2.75.0
     - `HuggingFace <https://huggingface.co/bigscience/bloom-1b1>`__
   * - huggingface-textgeneration-bloom-1b7
     - False
     - 1.0.1
     - 2.75.0
     - `HuggingFace <https://huggingface.co/bigscience/bloom-1b7>`__
   * - huggingface-textgeneration-bloom-560m
     - False
     - 1.0.1
     - 2.75.0
     - `HuggingFace <https://huggingface.co/bigscience/bloom-560m>`__
   * - huggingface-textgeneration-distilgpt2
     - False
     - 1.2.1
     - 2.75.0
     - `HuggingFace <https://huggingface.co/distilgpt2>`__
   * - huggingface-textgeneration-gpt2
     - False
     - 1.2.1
     - 2.75.0
     - `HuggingFace <https://huggingface.co/gpt2>`__
