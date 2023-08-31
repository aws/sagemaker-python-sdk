############################################
Text Summarization - HuggingFace
############################################

This is a supervised text summarization algorithm which supports many pre-trained models available in Hugging Face. The following
`sample notebook <https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/jumpstart_text_summarization/Amazon_JumpStart_Text_Summarization.ipynb>`__
demonstrates how to use the Sagemaker Python SDK for Text Summarization for using these algorithms.

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
   * - huggingface-summarization-bart-large-cnn-samsum
     - False
     - 1.2.0
     - 2.144.0
     - `HuggingFace <https://huggingface.co/philschmid/bart-large-cnn-samsum>`__
   * - huggingface-summarization-bert-small2bert-small-finetuned-cnn-daily-mail-summarization
     - False
     - 1.2.0
     - 2.144.0
     - `HuggingFace <https://huggingface.co/mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization>`__
   * - huggingface-summarization-bigbird-pegasus-large-arxiv
     - False
     - 1.2.0
     - 2.144.0
     - `HuggingFace <https://huggingface.co/google/bigbird-pegasus-large-arxiv>`__
   * - huggingface-summarization-bigbird-pegasus-large-pubmed
     - False
     - 1.2.0
     - 2.144.0
     - `HuggingFace <https://huggingface.co/google/bigbird-pegasus-large-pubmed>`__
   * - huggingface-summarization-distilbart-cnn-12-6
     - False
     - 1.2.0
     - 2.144.0
     - `HuggingFace <https://huggingface.co/sshleifer/distilbart-cnn-12-6>`__
   * - huggingface-summarization-distilbart-cnn-6-6
     - False
     - 1.2.0
     - 2.144.0
     - `HuggingFace <https://huggingface.co/sshleifer/distilbart-cnn-6-6>`__
   * - huggingface-summarization-distilbart-xsum-1-1
     - False
     - 1.2.0
     - 2.144.0
     - `HuggingFace <https://huggingface.co/sshleifer/distilbart-xsum-1-1>`__
   * - huggingface-summarization-distilbart-xsum-12-3
     - False
     - 1.2.0
     - 2.144.0
     - `HuggingFace <https://huggingface.co/sshleifer/distilbart-xsum-12-3>`__
