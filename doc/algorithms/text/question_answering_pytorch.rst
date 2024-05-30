#####################################
Question Answering - PyTorch
#####################################

This is a supervised question answering algorithm which supports fine-tuning of many pre-trained models available in Hugging Face. The following
`sample notebook <https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/jumpstart_question_answering/Amazon_JumpStart_Question_Answering.ipynb>`__
demonstrates how to use the Sagemaker Python SDK for Question Answering for using these algorithms.

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
   * - pytorch-eqa-bert-base-cased
     - True
     - 1.2.1
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__
   * - pytorch-eqa-bert-base-multilingual-cased
     - True
     - 1.2.1
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__
   * - pytorch-eqa-bert-base-multilingual-uncased
     - True
     - 1.2.1
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__
   * - pytorch-eqa-bert-base-uncased
     - True
     - 1.2.1
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__
   * - pytorch-eqa-bert-large-cased
     - True
     - 1.2.1
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__
   * - pytorch-eqa-bert-large-cased-whole-word-masking
     - True
     - 1.2.1
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__
   * - pytorch-eqa-bert-large-cased-whole-word-masking-finetuned-squad
     - True
     - 1.2.1
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__
   * - pytorch-eqa-bert-large-uncased
     - True
     - 1.2.1
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__
   * - pytorch-eqa-bert-large-uncased-whole-word-masking
     - True
     - 1.2.1
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__
   * - pytorch-eqa-bert-large-uncased-whole-word-masking-finetuned-squad
     - True
     - 1.2.1
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__
   * - pytorch-eqa-distilbert-base-cased
     - True
     - 1.2.1
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__
   * - pytorch-eqa-distilbert-base-multilingual-cased
     - True
     - 1.2.1
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__
   * - pytorch-eqa-distilbert-base-uncased
     - True
     - 1.2.1
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__
   * - pytorch-eqa-distilroberta-base
     - True
     - 1.2.1
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__
   * - pytorch-eqa-roberta-base
     - True
     - 1.2.1
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__
   * - pytorch-eqa-roberta-base-openai-detector
     - True
     - 1.2.1
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__
   * - pytorch-eqa-roberta-large
     - True
     - 1.2.1
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__
   * - pytorch-eqa-roberta-large-openai-detector
     - True
     - 1.2.1
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__
