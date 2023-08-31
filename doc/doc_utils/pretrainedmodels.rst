.. _all-pretrained-models:

.. |external-link| raw:: html

   <i class="fa fa-external-link"></i>

================================================
Built-in Algorithms with pre-trained Model Table
================================================

    The SageMaker Python SDK uses model IDs and model versions to access the necessary
    utilities for pre-trained models. This table serves to provide the core material plus
    some extra information that can be useful in selecting the correct model ID and
    corresponding parameters.

    If you want to automatically use the latest version of the model, use "*" for the `model_version` attribute.
    We highly suggest pinning an exact model version however.

    These models are also available through the
    `JumpStart UI in SageMaker Studio <https://docs.aws.amazon.com/sagemaker/latest/dg/studio-jumpstart.html>`__

.. list-table:: Available Models
   :widths: 50 20 20 20 30 20
   :header-rows: 1
   :class: datatable

   * - Model ID
     - Fine Tunable?
     - Latest Version
     - Min SDK Version
     - Problem Type
     - Source
   * - autogluon-classification-ensemble
     - True
     - 1.1.1
     - 2.103.0
     - Classification
     - `GluonCV <https://auto.gluon.ai/stable/index.html>`__ |external-link|
   * - autogluon-regression-ensemble
     - True
     - 1.1.1
     - 2.103.0
     - Regression
     - `GluonCV <https://auto.gluon.ai/stable/index.html>`__ |external-link|
   * - catboost-classification-model
     - True
     - 1.2.7
     - 2.75.0
     - Classification
     - `Catboost <https://catboost.ai/>`__ |external-link|
   * - catboost-regression-model
     - True
     - 1.2.7
     - 2.75.0
     - Regression
     - `Catboost <https://catboost.ai/>`__ |external-link|
   * - huggingface-asr-whisper-base
     - False
     - 1.0.1
     - 2.163.0
     - Source
     - `HuggingFace <https://huggingface.co/openai/whisper-base>`__ |external-link|
   * - huggingface-asr-whisper-large
     - False
     - 1.0.1
     - 2.163.0
     - Source
     - `HuggingFace <https://huggingface.co/openai/whisper-large>`__ |external-link|
   * - huggingface-asr-whisper-large-v2
     - False
     - 1.0.1
     - 2.163.0
     - Source
     - `HuggingFace <https://huggingface.co/openai/whisper-large-v2>`__ |external-link|
   * - huggingface-asr-whisper-medium
     - False
     - 1.0.1
     - 2.163.0
     - Source
     - `HuggingFace <https://huggingface.co/openai/whisper-medium>`__ |external-link|
   * - huggingface-asr-whisper-small
     - False
     - 1.0.1
     - 2.163.0
     - Source
     - `HuggingFace <https://huggingface.co/openai/whisper-small>`__ |external-link|
   * - huggingface-asr-whisper-tiny
     - False
     - 1.0.1
     - 2.163.0
     - Source
     - `HuggingFace <https://huggingface.co/openai/whisper-tiny>`__ |external-link|
   * - huggingface-eqa-bert-base-cased
     - True
     - 1.0.2
     - 2.75.0
     - Question Answering
     - `HuggingFace <https://huggingface.co/bert-base-cased>`__ |external-link|
   * - huggingface-eqa-bert-base-multilingual-cased
     - True
     - 1.0.2
     - 2.75.0
     - Question Answering
     - `HuggingFace <https://huggingface.co/bert-base-multilingual-cased>`__ |external-link|
   * - huggingface-eqa-bert-base-multilingual-uncased
     - True
     - 1.0.2
     - 2.75.0
     - Question Answering
     - `HuggingFace <https://huggingface.co/bert-base-multilingual-uncased>`__ |external-link|
   * - huggingface-eqa-bert-base-uncased
     - True
     - 1.0.2
     - 2.75.0
     - Question Answering
     - `HuggingFace <https://huggingface.co/bert-base-uncased>`__ |external-link|
   * - huggingface-eqa-bert-large-cased
     - True
     - 1.0.2
     - 2.75.0
     - Question Answering
     - `HuggingFace <https://huggingface.co/bert-large-cased>`__ |external-link|
   * - huggingface-eqa-bert-large-cased-whole-word-masking
     - True
     - 1.0.2
     - 2.75.0
     - Question Answering
     - `HuggingFace <https://huggingface.co/bert-large-cased-whole-word-masking>`__ |external-link|
   * - huggingface-eqa-bert-large-uncased
     - True
     - 1.0.2
     - 2.75.0
     - Question Answering
     - `HuggingFace <https://huggingface.co/bert-large-uncased>`__ |external-link|
   * - huggingface-eqa-bert-large-uncased-whole-word-masking
     - True
     - 1.0.2
     - 2.75.0
     - Question Answering
     - `HuggingFace <https://huggingface.co/bert-large-uncased-whole-word-masking>`__ |external-link|
   * - huggingface-eqa-distilbert-base-cased
     - True
     - 1.0.2
     - 2.75.0
     - Question Answering
     - `HuggingFace <https://huggingface.co/distilbert-base-cased>`__ |external-link|
   * - huggingface-eqa-distilbert-base-multilingual-cased
     - True
     - 1.0.2
     - 2.75.0
     - Question Answering
     - `HuggingFace <https://huggingface.co/distilbert-base-multilingual-cased>`__ |external-link|
   * - huggingface-eqa-distilbert-base-uncased
     - True
     - 1.0.2
     - 2.75.0
     - Question Answering
     - `HuggingFace <https://huggingface.co/distilbert-base-uncased>`__ |external-link|
   * - huggingface-eqa-distilroberta-base
     - True
     - 1.0.2
     - 2.75.0
     - Question Answering
     - `HuggingFace <https://huggingface.co/distilroberta-base>`__ |external-link|
   * - huggingface-eqa-roberta-base
     - True
     - 1.0.2
     - 2.75.0
     - Question Answering
     - `HuggingFace <https://huggingface.co/roberta-base>`__ |external-link|
   * - huggingface-eqa-roberta-base-openai-detector
     - True
     - 1.0.2
     - 2.75.0
     - Question Answering
     - `HuggingFace <https://huggingface.co/roberta-base-openai-detector>`__ |external-link|
   * - huggingface-eqa-roberta-large
     - True
     - 1.0.2
     - 2.75.0
     - Question Answering
     - `HuggingFace <https://huggingface.co/roberta-large>`__ |external-link|
   * - huggingface-fillmask-bert-base-uncased
     - True
     - 1.0.0
     - 2.81.0
     - Source
     - `HuggingFace <https://huggingface.co/bert-base-uncased>`__ |external-link|
   * - huggingface-llm-falcon-180b-bf16
     - False
     - 1.0.0
     - 2.175.0
     - Source
     - `HuggingFace <https://huggingface.co/tiiuae/falcon-180B>`__ |external-link|
   * - huggingface-llm-falcon-180b-chat-bf16
     - False
     - 1.0.0
     - 2.175.0
     - Source
     - `HuggingFace <https://huggingface.co/tiiuae/falcon-180B-chat>`__ |external-link|
   * - huggingface-llm-falcon-40b-bf16
     - False
     - 1.1.0
     - 2.175.0
     - Source
     - `HuggingFace <https://huggingface.co/tiiuae/falcon-40b>`__ |external-link|
   * - huggingface-llm-falcon-40b-instruct-bf16
     - False
     - 1.1.0
     - 2.175.0
     - Source
     - `HuggingFace <https://huggingface.co/tiiuae/falcon-40b-instruct>`__ |external-link|
   * - huggingface-llm-falcon-7b-bf16
     - True
     - 1.2.0
     - 2.175.0
     - Source
     - `HuggingFace <https://huggingface.co/tiiuae/falcon-7b>`__ |external-link|
   * - huggingface-llm-falcon-7b-instruct-bf16
     - True
     - 1.2.0
     - 2.175.0
     - Source
     - `HuggingFace <https://huggingface.co/tiiuae/falcon-7b-instruct>`__ |external-link|
   * - huggingface-llm-rinna-3-6b-instruction-ppo-bf16
     - False
     - 1.1.1
     - 2.175.0
     - Source
     - `HuggingFace <https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-ppo>`__ |external-link|
   * - huggingface-ner-distilbert-base-cased-finetuned-conll03-english
     - False
     - 1.1.0
     - 2.75.0
     - Named Entity Recognition
     - `HuggingFace <https://huggingface.co/elastic/distilbert-base-cased-finetuned-conll03-english>`__ |external-link|
   * - huggingface-ner-distilbert-base-uncased-finetuned-conll03-english
     - False
     - 1.1.0
     - 2.75.0
     - Named Entity Recognition
     - `HuggingFace <https://huggingface.co/elastic/distilbert-base-uncased-finetuned-conll03-english>`__ |external-link|
   * - huggingface-sentencesimilarity-all-MiniLM-L6-v2
     - True
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>`__ |external-link|
   * - huggingface-sentencesimilarity-bge-base-en
     - True
     - 1.0.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/BAAI/bge-base-en>`__ |external-link|
   * - huggingface-sentencesimilarity-bge-large-en
     - True
     - 1.0.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/BAAI/bge-large-en>`__ |external-link|
   * - huggingface-sentencesimilarity-bge-small-en
     - True
     - 1.0.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/BAAI/bge-small-en>`__ |external-link|
   * - huggingface-sentencesimilarity-e5-base
     - True
     - 1.0.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/intfloat/e5-base>`__ |external-link|
   * - huggingface-sentencesimilarity-e5-base-v2
     - True
     - 1.0.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/intfloat/e5-base-v2>`__ |external-link|
   * - huggingface-sentencesimilarity-e5-large
     - True
     - 1.0.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/intfloat/e5-large>`__ |external-link|
   * - huggingface-sentencesimilarity-e5-large-v2
     - True
     - 1.0.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/intfloat/e5-large-v2>`__ |external-link|
   * - huggingface-sentencesimilarity-e5-small-v2
     - True
     - 1.0.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/intfloat/e5-small-v2>`__ |external-link|
   * - huggingface-sentencesimilarity-gte-base
     - True
     - 1.0.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/thenlper/gte-base>`__ |external-link|
   * - huggingface-sentencesimilarity-gte-large
     - True
     - 1.0.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/thenlper/gte-large>`__ |external-link|
   * - huggingface-sentencesimilarity-gte-small
     - True
     - 1.0.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/thenlper/gte-small>`__ |external-link|
   * - huggingface-sentencesimilarity-multilingual-e5-base
     - True
     - 1.0.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/intfloat/multilingual-e5-base>`__ |external-link|
   * - huggingface-sentencesimilarity-multilingual-e5-large
     - True
     - 1.0.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/intfloat/multilingual-e5-large>`__ |external-link|
   * - huggingface-spc-bert-base-cased
     - True
     - 1.2.3
     - 2.75.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/bert-base-cased>`__ |external-link|
   * - huggingface-spc-bert-base-multilingual-cased
     - True
     - 1.2.3
     - 2.75.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/bert-base-multilingual-cased>`__ |external-link|
   * - huggingface-spc-bert-base-multilingual-uncased
     - True
     - 1.2.3
     - 2.75.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/bert-base-multilingual-uncased>`__ |external-link|
   * - huggingface-spc-bert-base-uncased
     - True
     - 1.2.3
     - 2.75.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/bert-base-uncased>`__ |external-link|
   * - huggingface-spc-bert-large-cased
     - True
     - 1.2.3
     - 2.75.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/bert-large-cased>`__ |external-link|
   * - huggingface-spc-bert-large-cased-whole-word-masking
     - True
     - 1.2.3
     - 2.75.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/bert-large-cased-whole-word-masking>`__ |external-link|
   * - huggingface-spc-bert-large-uncased
     - True
     - 1.2.3
     - 2.75.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/bert-large-uncased>`__ |external-link|
   * - huggingface-spc-bert-large-uncased-whole-word-masking
     - True
     - 1.2.3
     - 2.75.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/bert-large-uncased-whole-word-masking>`__ |external-link|
   * - huggingface-spc-distilbert-base-cased
     - True
     - 1.2.3
     - 2.75.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/distilbert-base-cased>`__ |external-link|
   * - huggingface-spc-distilbert-base-multilingual-cased
     - True
     - 1.2.3
     - 2.75.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/distilbert-base-multilingual-cased>`__ |external-link|
   * - huggingface-spc-distilbert-base-uncased
     - True
     - 1.2.3
     - 2.75.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/distilbert-base-uncased>`__ |external-link|
   * - huggingface-spc-distilroberta-base
     - True
     - 1.2.3
     - 2.75.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/distilroberta-base>`__ |external-link|
   * - huggingface-spc-roberta-base
     - True
     - 1.2.3
     - 2.75.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/roberta-base>`__ |external-link|
   * - huggingface-spc-roberta-base-openai-detector
     - True
     - 1.2.3
     - 2.75.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/roberta-base-openai-detector>`__ |external-link|
   * - huggingface-spc-roberta-large
     - True
     - 1.2.3
     - 2.75.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/roberta-large>`__ |external-link|
   * - huggingface-spc-roberta-large-openai-detector
     - True
     - 1.2.3
     - 2.75.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/roberta-large-openai-detector>`__ |external-link|
   * - huggingface-spc-xlm-clm-ende-1024
     - True
     - 1.2.3
     - 2.75.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/xlm-clm-ende-1024>`__ |external-link|
   * - huggingface-spc-xlm-mlm-ende-1024
     - True
     - 1.2.3
     - 2.75.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/xlm-mlm-ende-1024>`__ |external-link|
   * - huggingface-spc-xlm-mlm-enro-1024
     - True
     - 1.2.3
     - 2.75.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/xlm-mlm-enro-1024>`__ |external-link|
   * - huggingface-spc-xlm-mlm-tlm-xnli15-1024
     - True
     - 1.2.3
     - 2.75.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/xlm-mlm-tlm-xnli15-1024>`__ |external-link|
   * - huggingface-spc-xlm-mlm-xnli15-1024
     - True
     - 1.2.3
     - 2.75.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/xlm-mlm-xnli15-1024>`__ |external-link|
   * - huggingface-summarization-bart-large-cnn-samsum
     - False
     - 1.2.0
     - 2.144.0
     - Text Summarization
     - `HuggingFace <https://huggingface.co/philschmid/bart-large-cnn-samsum>`__ |external-link|
   * - huggingface-summarization-bert-small2bert-small-finetuned-cnn-daily-mail-summarization
     - False
     - 1.2.0
     - 2.144.0
     - Text Summarization
     - `HuggingFace <https://huggingface.co/mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization>`__ |external-link|
   * - huggingface-summarization-bigbird-pegasus-large-arxiv
     - False
     - 1.2.0
     - 2.144.0
     - Text Summarization
     - `HuggingFace <https://huggingface.co/google/bigbird-pegasus-large-arxiv>`__ |external-link|
   * - huggingface-summarization-bigbird-pegasus-large-pubmed
     - False
     - 1.2.0
     - 2.144.0
     - Text Summarization
     - `HuggingFace <https://huggingface.co/google/bigbird-pegasus-large-pubmed>`__ |external-link|
   * - huggingface-summarization-distilbart-cnn-12-6
     - False
     - 1.2.0
     - 2.144.0
     - Text Summarization
     - `HuggingFace <https://huggingface.co/sshleifer/distilbart-cnn-12-6>`__ |external-link|
   * - huggingface-summarization-distilbart-cnn-6-6
     - False
     - 1.2.0
     - 2.144.0
     - Text Summarization
     - `HuggingFace <https://huggingface.co/sshleifer/distilbart-cnn-6-6>`__ |external-link|
   * - huggingface-summarization-distilbart-xsum-1-1
     - False
     - 1.2.0
     - 2.144.0
     - Text Summarization
     - `HuggingFace <https://huggingface.co/sshleifer/distilbart-xsum-1-1>`__ |external-link|
   * - huggingface-summarization-distilbart-xsum-12-3
     - False
     - 1.2.0
     - 2.144.0
     - Text Summarization
     - `HuggingFace <https://huggingface.co/sshleifer/distilbart-xsum-12-3>`__ |external-link|
   * - huggingface-tc-bert-base-cased
     - True
     - 1.0.2
     - 2.81.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/bert-base-cased>`__ |external-link|
   * - huggingface-tc-bert-base-multilingual-cased
     - True
     - 1.0.2
     - 2.81.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/bert-base-multilingual-cased>`__ |external-link|
   * - huggingface-tc-bert-base-multilingual-uncased
     - True
     - 1.0.2
     - 2.81.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/bert-base-multilingual-uncased>`__ |external-link|
   * - huggingface-tc-bert-base-uncased
     - True
     - 1.0.2
     - 2.81.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/bert-base-uncased>`__ |external-link|
   * - huggingface-tc-bert-large-cased
     - True
     - 1.0.2
     - 2.81.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/bert-large-cased>`__ |external-link|
   * - huggingface-tc-bert-large-cased-whole-word-masking
     - True
     - 1.0.2
     - 2.81.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/bert-large-cased-whole-word-masking>`__ |external-link|
   * - huggingface-tc-bert-large-uncased
     - True
     - 1.0.2
     - 2.81.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/bert-large-uncased>`__ |external-link|
   * - huggingface-tc-bert-large-uncased-whole-word-masking
     - True
     - 1.0.2
     - 2.81.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/bert-large-uncased-whole-word-masking>`__ |external-link|
   * - huggingface-tc-distilbert-base-cased
     - True
     - 1.0.2
     - 2.81.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/distilbert-base-cased>`__ |external-link|
   * - huggingface-tc-distilbert-base-multilingual-cased
     - True
     - 1.0.2
     - 2.81.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/distilbert-base-multilingual-cased>`__ |external-link|
   * - huggingface-tc-distilbert-base-uncased
     - True
     - 1.0.2
     - 2.81.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/distilbert-base-uncased>`__ |external-link|
   * - huggingface-tc-distilroberta-base
     - True
     - 1.0.2
     - 2.81.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/distilroberta-base>`__ |external-link|
   * - huggingface-tc-models
     - True
     - 1.0.2
     - 2.81.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/albert-base-v2>`__ |external-link|
   * - huggingface-tc-roberta-base
     - True
     - 1.0.2
     - 2.81.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/roberta-base>`__ |external-link|
   * - huggingface-tc-roberta-base-openai-detector
     - True
     - 1.0.2
     - 2.81.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/roberta-base-openai-detector>`__ |external-link|
   * - huggingface-tc-roberta-large
     - True
     - 1.0.2
     - 2.81.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/roberta-large>`__ |external-link|
   * - huggingface-tc-roberta-large-openai-detector
     - True
     - 1.0.2
     - 2.81.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/roberta-large-openai-detector>`__ |external-link|
   * - huggingface-tc-xlm-clm-ende-1024
     - True
     - 1.0.2
     - 2.81.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/xlm-clm-ende-1024>`__ |external-link|
   * - huggingface-tc-xlm-mlm-ende-1024
     - True
     - 1.0.2
     - 2.81.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/xlm-mlm-ende-1024>`__ |external-link|
   * - huggingface-tc-xlm-mlm-enro-1024
     - True
     - 1.0.2
     - 2.81.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/xlm-mlm-enro-1024>`__ |external-link|
   * - huggingface-tc-xlm-mlm-tlm-xnli15-1024
     - True
     - 1.0.2
     - 2.81.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/xlm-mlm-tlm-xnli15-1024>`__ |external-link|
   * - huggingface-text2text-bart4csc-base-chinese
     - False
     - 1.2.2
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/shibing624/bart4csc-base-chinese>`__ |external-link|
   * - huggingface-text2text-bigscience-t0pp
     - False
     - 1.1.2
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/bigscience/T0pp>`__ |external-link|
   * - huggingface-text2text-bigscience-t0pp-bnb-int8
     - False
     - 1.1.2
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/bigscience/T0pp>`__ |external-link|
   * - huggingface-text2text-bigscience-t0pp-fp16
     - False
     - 1.1.2
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/bigscience/T0pp>`__ |external-link|
   * - huggingface-text2text-flan-t5-base
     - True
     - 1.3.2
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/google/flan-t5-base>`__ |external-link|
   * - huggingface-text2text-flan-t5-base-samsum
     - False
     - 1.2.2
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/philschmid/flan-t5-base-samsum>`__ |external-link|
   * - huggingface-text2text-flan-t5-large
     - True
     - 1.2.2
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/google/flan-t5-large>`__ |external-link|
   * - huggingface-text2text-flan-t5-small
     - True
     - 1.3.2
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/google/flan-t5-small>`__ |external-link|
   * - huggingface-text2text-flan-t5-xl
     - True
     - 1.2.2
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/google/flan-t5-xl>`__ |external-link|
   * - huggingface-text2text-flan-t5-xxl
     - True
     - 1.1.2
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/google/flan-t5-xxl>`__ |external-link|
   * - huggingface-text2text-flan-t5-xxl-bnb-int8
     - False
     - 1.2.2
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/google/flan-t5-xxl>`__ |external-link|
   * - huggingface-text2text-flan-t5-xxl-fp16
     - True
     - 1.1.2
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/google/flan-t5-xxl>`__ |external-link|
   * - huggingface-text2text-flan-ul2-bf16
     - False
     - 1.1.2
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/google/flan-ul2>`__ |external-link|
   * - huggingface-text2text-pegasus-paraphrase
     - False
     - 1.2.2
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/shibing624/bart4csc-base-chinese>`__ |external-link|
   * - huggingface-text2text-qcpg-sentences
     - False
     - 1.2.2
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/ibm/qcpg-sentences>`__ |external-link|
   * - huggingface-text2text-t5-one-line-summary
     - False
     - 1.2.2
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/snrspeaks/t5-one-line-summary>`__ |external-link|
   * - huggingface-textembedding-all-MiniLM-L6-v2
     - False
     - 1.0.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>`__ |external-link|
   * - huggingface-textembedding-bloom-7b1
     - False
     - 1.0.1
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/bigscience/bloom-7b1>`__ |external-link|
   * - huggingface-textembedding-bloom-7b1-fp16
     - False
     - 1.0.1
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/bigscience/bloom-7b1>`__ |external-link|
   * - huggingface-textembedding-gpt-j-6b
     - False
     - 1.0.1
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/EleutherAI/gpt-j-6B>`__ |external-link|
   * - huggingface-textembedding-gpt-j-6b-fp16
     - False
     - 1.0.1
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/EleutherAI/gpt-j-6B>`__ |external-link|
   * - huggingface-textgeneration-bloom-1b1
     - False
     - 1.3.0
     - 2.144.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/bigscience/bloom-1b1>`__ |external-link|
   * - huggingface-textgeneration-bloom-1b7
     - False
     - 1.3.0
     - 2.144.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/bigscience/bloom-1b7>`__ |external-link|
   * - huggingface-textgeneration-bloom-560m
     - False
     - 1.3.0
     - 2.144.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/bigscience/bloom-560m>`__ |external-link|
   * - huggingface-textgeneration-bloomz-1b1
     - False
     - 1.2.0
     - 2.144.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/bigscience/bloomz-1b1>`__ |external-link|
   * - huggingface-textgeneration-bloomz-1b7
     - False
     - 1.2.0
     - 2.144.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/bigscience/bloomz-1b7>`__ |external-link|
   * - huggingface-textgeneration-bloomz-560m
     - False
     - 1.2.0
     - 2.144.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/bigscience/bloomz-560m>`__ |external-link|
   * - huggingface-textgeneration-distilgpt2
     - False
     - 1.5.0
     - 2.144.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/distilgpt2>`__ |external-link|
   * - huggingface-textgeneration-dolly-v2-12b-bf16
     - False
     - 1.1.0
     - 2.144.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/databricks/dolly-v2-12b>`__ |external-link|
   * - huggingface-textgeneration-dolly-v2-3b-bf16
     - False
     - 1.1.0
     - 2.144.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/databricks/dolly-v2-3b>`__ |external-link|
   * - huggingface-textgeneration-dolly-v2-7b-bf16
     - False
     - 1.1.0
     - 2.144.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/databricks/dolly-v2-7b>`__ |external-link|
   * - huggingface-textgeneration-falcon-40b-bf16
     - False
     - 1.0.0
     - 2.144.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/tiiuae/falcon-40b>`__ |external-link|
   * - huggingface-textgeneration-falcon-40b-instruct-bf16
     - False
     - 1.0.0
     - 2.144.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/tiiuae/falcon-40b-instruct>`__ |external-link|
   * - huggingface-textgeneration-falcon-7b-bf16
     - False
     - 1.0.0
     - 2.144.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/tiiuae/falcon-7b>`__ |external-link|
   * - huggingface-textgeneration-falcon-7b-instruct-bf16
     - False
     - 1.0.0
     - 2.144.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/tiiuae/falcon-7b-instruct>`__ |external-link|
   * - huggingface-textgeneration-gpt2
     - False
     - 1.5.0
     - 2.144.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/gpt2>`__ |external-link|
   * - huggingface-textgeneration-models
     - False
     - 1.3.0
     - 2.144.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads>`__ |external-link|
   * - huggingface-textgeneration-open-llama
     - False
     - 1.2.0
     - 2.144.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/openlm-research>`__ |external-link|
   * - huggingface-textgeneration1-bloom-176b-int8
     - False
     - 1.0.2
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/microsoft/bloom-deepspeed-inference-int8>`__ |external-link|
   * - huggingface-textgeneration1-bloom-3b
     - True
     - 1.3.0
     - 2.165.0
     - Source
     - `HuggingFace <https://huggingface.co/bigscience/bloom-3b>`__ |external-link|
   * - huggingface-textgeneration1-bloom-3b-fp16
     - True
     - 1.3.0
     - 2.165.0
     - Source
     - `HuggingFace <https://huggingface.co/bigscience/bloom-3b>`__ |external-link|
   * - huggingface-textgeneration1-bloom-7b1
     - True
     - 1.3.0
     - 2.165.0
     - Source
     - `HuggingFace <https://huggingface.co/bigscience/bloom-7b1>`__ |external-link|
   * - huggingface-textgeneration1-bloom-7b1-fp16
     - True
     - 1.3.0
     - 2.165.0
     - Source
     - `HuggingFace <https://huggingface.co/bigscience/bloom-7b1>`__ |external-link|
   * - huggingface-textgeneration1-bloomz-176b-fp16
     - False
     - 1.0.2
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/bigscience/bloomz>`__ |external-link|
   * - huggingface-textgeneration1-bloomz-3b-fp16
     - True
     - 1.3.0
     - 2.165.0
     - Source
     - `HuggingFace <https://huggingface.co/bigscience/bloomz-3b>`__ |external-link|
   * - huggingface-textgeneration1-bloomz-7b1-fp16
     - True
     - 1.3.0
     - 2.165.0
     - Source
     - `HuggingFace <https://huggingface.co/bigscience/bloomz-7b1>`__ |external-link|
   * - huggingface-textgeneration1-gpt-2-xl
     - True
     - 1.3.0
     - 2.165.0
     - Source
     - `HuggingFace <https://huggingface.co/gpt2-xl>`__ |external-link|
   * - huggingface-textgeneration1-gpt-2-xl-fp16
     - True
     - 1.3.0
     - 2.165.0
     - Source
     - `HuggingFace <https://huggingface.co/gpt2-xl>`__ |external-link|
   * - huggingface-textgeneration1-gpt-j-6b
     - True
     - 1.3.0
     - 2.165.0
     - Source
     - `HuggingFace <https://huggingface.co/EleutherAI/gpt-j-6B>`__ |external-link|
   * - huggingface-textgeneration1-gpt-j-6b-fp16
     - True
     - 1.3.0
     - 2.165.0
     - Source
     - `HuggingFace <https://huggingface.co/EleutherAI/gpt-j-6B>`__ |external-link|
   * - huggingface-textgeneration1-gpt-neo-1-3b
     - True
     - 1.3.0
     - 2.165.0
     - Source
     - `HuggingFace <https://huggingface.co/EleutherAI/gpt-neo-1.3B>`__ |external-link|
   * - huggingface-textgeneration1-gpt-neo-1-3b-fp16
     - True
     - 1.3.0
     - 2.165.0
     - Source
     - `HuggingFace <https://huggingface.co/EleutherAI/gpt-neo-1.3B>`__ |external-link|
   * - huggingface-textgeneration1-gpt-neo-125m
     - True
     - 1.3.0
     - 2.165.0
     - Source
     - `HuggingFace <https://huggingface.co/EleutherAI/gpt-neo-125M>`__ |external-link|
   * - huggingface-textgeneration1-gpt-neo-125m-fp16
     - True
     - 1.3.0
     - 2.165.0
     - Source
     - `HuggingFace <https://huggingface.co/EleutherAI/gpt-neo-125M>`__ |external-link|
   * - huggingface-textgeneration1-gpt-neo-2-7b
     - True
     - 1.3.0
     - 2.165.0
     - Source
     - `HuggingFace <https://huggingface.co/EleutherAI/gpt-neo-2.7B>`__ |external-link|
   * - huggingface-textgeneration1-gpt-neo-2-7b-fp16
     - True
     - 1.3.0
     - 2.165.0
     - Source
     - `HuggingFace <https://huggingface.co/EleutherAI/gpt-neo-2.7B>`__ |external-link|
   * - huggingface-textgeneration1-lightgpt
     - True
     - 1.1.0
     - 2.165.0
     - Source
     - `HuggingFace <https://huggingface.co/amazon/LightGPT>`__ |external-link|
   * - huggingface-textgeneration1-mpt-7b-bf16
     - False
     - 1.0.0
     - 2.153.0
     - Source
     - `HuggingFace <https://huggingface.co/mosaicml/mpt-7b>`__ |external-link|
   * - huggingface-textgeneration1-mpt-7b-instruct-bf16
     - False
     - 1.0.0
     - 2.153.0
     - Source
     - `HuggingFace <https://huggingface.co/mosaicml/mpt-7b-instruct>`__ |external-link|
   * - huggingface-textgeneration1-mpt-7b-storywriter-bf16
     - False
     - 1.0.0
     - 2.153.0
     - Source
     - `HuggingFace <https://huggingface.co/mosaicml/mpt-7b-storywriter>`__ |external-link|
   * - huggingface-textgeneration1-redpajama-incite-base-3B-v1-fp16
     - True
     - 1.1.0
     - 2.165.0
     - Source
     - `HuggingFace <https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-3B-v1>`__ |external-link|
   * - huggingface-textgeneration1-redpajama-incite-base-7B-v1-fp16
     - True
     - 1.1.0
     - 2.165.0
     - Source
     - `HuggingFace <https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-7B-v0.1>`__ |external-link|
   * - huggingface-textgeneration1-redpajama-incite-chat-3B-v1-fp16
     - True
     - 1.1.0
     - 2.165.0
     - Source
     - `HuggingFace <https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1>`__ |external-link|
   * - huggingface-textgeneration1-redpajama-incite-chat-7B-v1-fp16
     - True
     - 1.1.0
     - 2.165.0
     - Source
     - `HuggingFace <https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-7B-v0.1>`__ |external-link|
   * - huggingface-textgeneration1-redpajama-incite-instruct-3B-v1-fp16
     - True
     - 1.1.0
     - 2.165.0
     - Source
     - `HuggingFace <https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-3B-v1>`__ |external-link|
   * - huggingface-textgeneration1-redpajama-incite-instruct-7B-v1-fp16
     - True
     - 1.1.0
     - 2.165.0
     - Source
     - `HuggingFace <https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-7B-v0.1>`__ |external-link|
   * - huggingface-textgeneration2-gpt-neox-20b-fp16
     - False
     - 1.0.1
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/EleutherAI/gpt-neox-20b>`__ |external-link|
   * - huggingface-textgeneration2-gpt-neoxt-chat-base-20b-fp16
     - False
     - 1.0.1
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/togethercomputer/GPT-NeoXT-Chat-Base-20B>`__ |external-link|
   * - huggingface-translation-opus-mt-en-es
     - False
     - 1.1.0
     - 2.75.0
     - Machine Translation
     - `HuggingFace <https://huggingface.co/Helsinki-NLP/opus-mt-en-es>`__ |external-link|
   * - huggingface-translation-opus-mt-en-vi
     - False
     - 1.1.0
     - 2.75.0
     - Machine Translation
     - `HuggingFace <https://huggingface.co/Helsinki-NLP/opus-mt-en-vi>`__ |external-link|
   * - huggingface-translation-t5-base
     - False
     - 1.1.0
     - 2.75.0
     - Machine Translation
     - `HuggingFace <https://huggingface.co/t5-base>`__ |external-link|
   * - huggingface-translation-t5-large
     - False
     - 1.1.0
     - 2.75.0
     - Machine Translation
     - `HuggingFace <https://huggingface.co/t5-large>`__ |external-link|
   * - huggingface-translation-t5-small
     - False
     - 1.1.0
     - 2.75.0
     - Machine Translation
     - `HuggingFace <https://huggingface.co/t5-small>`__ |external-link|
   * - huggingface-txt2img-22h-vintedois-diffusion-v0-1
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/22h/vintedois-diffusion-v0-1>`__ |external-link|
   * - huggingface-txt2img-akikagura-mkgen-diffusion
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/AkiKagura/mkgen-diffusion>`__ |external-link|
   * - huggingface-txt2img-alxdfy-noggles-fastdb-4800
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/alxdfy/noggles-fastdb-4800>`__ |external-link|
   * - huggingface-txt2img-alxdfy-noggles9000
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/alxdfy/noggles9000>`__ |external-link|
   * - huggingface-txt2img-andite-anything-v4-0
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/andite/anything-v4.0>`__ |external-link|
   * - huggingface-txt2img-astraliteheart-pony-diffusion-v2
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/AstraliteHeart/pony-diffusion-v2>`__ |external-link|
   * - huggingface-txt2img-avrik-abstract-anim-spritesheets
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/Avrik/abstract-anim-spritesheets>`__ |external-link|
   * - huggingface-txt2img-aybeeceedee-knollingcase
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/Aybeeceedee/knollingcase>`__ |external-link|
   * - huggingface-txt2img-bingsu-my-k-anything-v3-0
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/Bingsu/my-k-anything-v3-0>`__ |external-link|
   * - huggingface-txt2img-bingsu-my-korean-stable-diffusion-v1-5
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/Bingsu/my-korean-stable-diffusion-v1-5>`__ |external-link|
   * - huggingface-txt2img-buntopsih-novgoranstefanovski
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/Buntopsih/novgoranstefanovski>`__ |external-link|
   * - huggingface-txt2img-claudfuen-photorealistic-fuen-v1
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/claudfuen/photorealistic-fuen-v1>`__ |external-link|
   * - huggingface-txt2img-coder119-vectorartz-diffusion
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/coder119/Vectorartz_Diffusion>`__ |external-link|
   * - huggingface-txt2img-conflictx-complex-lineart
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/Conflictx/Complex-Lineart>`__ |external-link|
   * - huggingface-txt2img-dallinmackay-cats-musical-diffusion
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/dallinmackay/Cats-Musical-diffusion>`__ |external-link|
   * - huggingface-txt2img-dallinmackay-jwst-deep-space-diffusion
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/dallinmackay/JWST-Deep-Space-diffusion>`__ |external-link|
   * - huggingface-txt2img-dallinmackay-tron-legacy-diffusion
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/dallinmackay/Tron-Legacy-diffusion>`__ |external-link|
   * - huggingface-txt2img-dallinmackay-van-gogh-diffusion
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/dallinmackay/Van-Gogh-diffusion>`__ |external-link|
   * - huggingface-txt2img-dgspitzer-cyberpunk-anime-diffusion
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/DGSpitzer/Cyberpunk-Anime-Diffusion>`__ |external-link|
   * - huggingface-txt2img-dreamlike-art-dreamlike-diffusion-1-0
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/dreamlike-art/dreamlike-diffusion-1.0>`__ |external-link|
   * - huggingface-txt2img-eimiss-eimisanimediffusion-1-0v
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/eimiss/EimisAnimeDiffusion_1.0v>`__ |external-link|
   * - huggingface-txt2img-envvi-inkpunk-diffusion
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/Envvi/Inkpunk-Diffusion>`__ |external-link|
   * - huggingface-txt2img-evel-yoshin
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/Evel/YoShin>`__ |external-link|
   * - huggingface-txt2img-extraphy-mustafa-kemal-ataturk
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/Extraphy/mustafa-kemal-ataturk>`__ |external-link|
   * - huggingface-txt2img-fffiloni-mr-men-and-little-misses
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/fffiloni/mr-men-and-little-misses>`__ |external-link|
   * - huggingface-txt2img-fictiverse-elrisitas
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/Fictiverse/ElRisitas>`__ |external-link|
   * - huggingface-txt2img-fictiverse-stable-diffusion-balloonart-model
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/Fictiverse/Stable_Diffusion_BalloonArt_Model>`__ |external-link|
   * - huggingface-txt2img-fictiverse-stable-diffusion-microscopic-model
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/Fictiverse/Stable_Diffusion_Microscopic_model>`__ |external-link|
   * - huggingface-txt2img-fictiverse-stable-diffusion-papercut-model
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/Fictiverse/Stable_Diffusion_PaperCut_Model>`__ |external-link|
   * - huggingface-txt2img-fictiverse-stable-diffusion-voxelart-model
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/Fictiverse/Stable_Diffusion_VoxelArt_Model>`__ |external-link|
   * - huggingface-txt2img-haor-evt-v3
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/haor/Evt_V3>`__ |external-link|
   * - huggingface-txt2img-hassanblend-hassanblend1-4
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/hassanblend/hassanblend1.4>`__ |external-link|
   * - huggingface-txt2img-idea-ccnl-taiyi-stable-diffusion-1b-chinese-en-v0-1
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1>`__ |external-link|
   * - huggingface-txt2img-idea-ccnl-taiyi-stable-diffusion-1b-chinese-v0-1
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1>`__ |external-link|
   * - huggingface-txt2img-ifansnek-johndiffusion
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/IfanSnek/JohnDiffusion>`__ |external-link|
   * - huggingface-txt2img-jersonm89-avatar
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/Jersonm89/Avatar>`__ |external-link|
   * - huggingface-txt2img-jvkape-iconsmi-appiconsmodelforsd
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/jvkape/IconsMI-AppIconsModelforSD>`__ |external-link|
   * - huggingface-txt2img-katakana-2d-mix
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/katakana/2D-Mix>`__ |external-link|
   * - huggingface-txt2img-lacambre-vulvine-look-v02
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/LaCambre/vulvine-look-v02>`__ |external-link|
   * - huggingface-txt2img-langboat-guohua-diffusion
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/Langboat/Guohua-Diffusion>`__ |external-link|
   * - huggingface-txt2img-linaqruf-anything-v3-0
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/Linaqruf/anything-v3.0>`__ |external-link|
   * - huggingface-txt2img-mikesmodels-waltz-with-bashir-diffusion
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/mikesmodels/Waltz_with_Bashir_Diffusion>`__ |external-link|
   * - huggingface-txt2img-mitchtech-klingon-diffusion
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/mitchtech/klingon-diffusion>`__ |external-link|
   * - huggingface-txt2img-mitchtech-vulcan-diffusion
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/mitchtech/vulcan-diffusion>`__ |external-link|
   * - huggingface-txt2img-mitsua-mitsua-diffusion-cc0
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/Mitsua/mitsua-diffusion-cc0>`__ |external-link|
   * - huggingface-txt2img-naclbit-trinart-stable-diffusion-v2
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/naclbit/trinart_stable_diffusion_v2>`__ |external-link|
   * - huggingface-txt2img-nitrosocke-arcane-diffusion
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/nitrosocke/Arcane-Diffusion>`__ |external-link|
   * - huggingface-txt2img-nitrosocke-archer-diffusion
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/nitrosocke/archer-diffusion>`__ |external-link|
   * - huggingface-txt2img-nitrosocke-classic-anim-diffusion
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/nitrosocke/classic-anim-diffusion>`__ |external-link|
   * - huggingface-txt2img-nitrosocke-elden-ring-diffusion
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/nitrosocke/elden-ring-diffusion>`__ |external-link|
   * - huggingface-txt2img-nitrosocke-future-diffusion
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/nitrosocke/Future-Diffusion>`__ |external-link|
   * - huggingface-txt2img-nitrosocke-ghibli-diffusion
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/nitrosocke/Ghibli-Diffusion>`__ |external-link|
   * - huggingface-txt2img-nitrosocke-mo-di-diffusion
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/nitrosocke/mo-di-diffusion>`__ |external-link|
   * - huggingface-txt2img-nitrosocke-nitro-diffusion
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/nitrosocke/Nitro-Diffusion>`__ |external-link|
   * - huggingface-txt2img-nitrosocke-redshift-diffusion
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/nitrosocke/redshift-diffusion>`__ |external-link|
   * - huggingface-txt2img-nitrosocke-spider-verse-diffusion
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/nitrosocke/spider-verse-diffusion>`__ |external-link|
   * - huggingface-txt2img-nousr-robo-diffusion
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/nousr/robo-diffusion>`__ |external-link|
   * - huggingface-txt2img-ogkalu-comic-diffusion
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/ogkalu/Comic-Diffusion>`__ |external-link|
   * - huggingface-txt2img-openjourney-openjourney
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/openjourney/openjourney>`__ |external-link|
   * - huggingface-txt2img-piesposito-openpotionbottle-v2
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/piEsposito/openpotionbottle-v2>`__ |external-link|
   * - huggingface-txt2img-plasmo-voxel-ish
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/plasmo/voxel-ish>`__ |external-link|
   * - huggingface-txt2img-plasmo-woolitize
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/plasmo/woolitize>`__ |external-link|
   * - huggingface-txt2img-progamergov-min-illust-background-diffusion
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/ProGamerGov/Min-Illust-Background-Diffusion>`__ |external-link|
   * - huggingface-txt2img-prompthero-linkedin-diffusion
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/prompthero/linkedin-diffusion>`__ |external-link|
   * - huggingface-txt2img-prompthero-openjourney
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/prompthero/openjourney>`__ |external-link|
   * - huggingface-txt2img-qilex-magic-diffusion
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/Qilex/magic-diffusion>`__ |external-link|
   * - huggingface-txt2img-rabidgremlin-sd-db-epic-space-machine
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/rabidgremlin/sd-db-epic-space-machine>`__ |external-link|
   * - huggingface-txt2img-rayhell-popupbook-diffusion
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/RayHell/popupBook-diffusion>`__ |external-link|
   * - huggingface-txt2img-runwayml-stable-diffusion-v1-5
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/runwayml/stable-diffusion-v1-5>`__ |external-link|
   * - huggingface-txt2img-s3nh-beksinski-style-stable-diffusion
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/s3nh/beksinski-style-stable-diffusion>`__ |external-link|
   * - huggingface-txt2img-sd-dreambooth-library-original-character-cyclps
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/sd-dreambooth-library/original-character-cyclps>`__ |external-link|
   * - huggingface-txt2img-sd-dreambooth-library-persona-5-shigenori-style
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/sd-dreambooth-library/persona-5-shigenori-style>`__ |external-link|
   * - huggingface-txt2img-sd-dreambooth-library-seraphm
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/sd-dreambooth-library/seraphm>`__ |external-link|
   * - huggingface-txt2img-shirayu-sd-tohoku-v1
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/shirayu/sd-tohoku-v1>`__ |external-link|
   * - huggingface-txt2img-thelastben-hrrzg-style-768px
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/TheLastBen/hrrzg-style-768px>`__ |external-link|
   * - huggingface-txt2img-timothepearce-gina-the-cat
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/timothepearce/gina-the-cat>`__ |external-link|
   * - huggingface-txt2img-trystar-clonediffusion
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/TryStar/CloneDiffusion>`__ |external-link|
   * - huggingface-txt2img-tuwonga-dbluth
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/tuwonga/dbluth>`__ |external-link|
   * - huggingface-txt2img-tuwonga-rotoscopee
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/tuwonga/rotoscopee>`__ |external-link|
   * - huggingface-txt2img-volrath50-fantasy-card-diffusion
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/volrath50/fantasy-card-diffusion>`__ |external-link|
   * - huggingface-txt2img-yayab-sd-onepiece-diffusers4
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/YaYaB/sd-onepiece-diffusers4>`__ |external-link|
   * - huggingface-zstc-cross-encoder-nli-deberta-base
     - False
     - 1.0.0
     - 2.81.0
     - Source
     - `HuggingFace <https://huggingface.co/cross-encoder/nli-deberta-base>`__ |external-link|
   * - huggingface-zstc-cross-encoder-nli-distilroberta-base
     - False
     - 1.0.0
     - 2.81.0
     - Source
     - `HuggingFace <https://huggingface.co/cross-encoder/nli-distilroberta-base>`__ |external-link|
   * - huggingface-zstc-cross-encoder-nli-minilm2-l6-h768
     - False
     - 1.0.0
     - 2.81.0
     - Source
     - `HuggingFace <https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768>`__ |external-link|
   * - huggingface-zstc-cross-encoder-nli-roberta-base
     - False
     - 1.0.0
     - 2.81.0
     - Source
     - `HuggingFace <https://huggingface.co/cross-encoder/nli-roberta-base>`__ |external-link|
   * - huggingface-zstc-digitalepidemiologylab-covid-twitter-bert-v2-mnli
     - False
     - 1.0.0
     - 2.81.0
     - Source
     - `HuggingFace <https://huggingface.co/digitalepidemiologylab/covid-twitter-bert-v2-mnli>`__ |external-link|
   * - huggingface-zstc-eleldar-theme-classification
     - False
     - 1.0.0
     - 2.81.0
     - Source
     - `HuggingFace <https://huggingface.co/eleldar/theme-classification>`__ |external-link|
   * - huggingface-zstc-emrecan-bert-base-multilingual-cased-allnli-tr
     - False
     - 1.0.0
     - 2.81.0
     - Source
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-multilingual-cased-allnli_tr>`__ |external-link|
   * - huggingface-zstc-emrecan-bert-base-multilingual-cased-multinli-tr
     - False
     - 1.0.0
     - 2.81.0
     - Source
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-multilingual-cased-multinli_tr>`__ |external-link|
   * - huggingface-zstc-emrecan-bert-base-multilingual-cased-snli-tr
     - False
     - 1.0.0
     - 2.81.0
     - Source
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-multilingual-cased-snli_tr>`__ |external-link|
   * - huggingface-zstc-emrecan-bert-base-turkish-cased-allnli-tr
     - False
     - 1.0.0
     - 2.81.0
     - Source
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-turkish-cased-allnli_tr>`__ |external-link|
   * - huggingface-zstc-emrecan-bert-base-turkish-cased-multinli-tr
     - False
     - 1.0.0
     - 2.81.0
     - Source
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-turkish-cased-multinli_tr>`__ |external-link|
   * - huggingface-zstc-emrecan-bert-base-turkish-cased-snli-tr
     - False
     - 1.0.0
     - 2.81.0
     - Source
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-turkish-cased-snli_tr>`__ |external-link|
   * - huggingface-zstc-emrecan-convbert-base-turkish-mc4-cased-allnli-tr
     - False
     - 1.0.0
     - 2.81.0
     - Source
     - `HuggingFace <https://huggingface.co/emrecan/convbert-base-turkish-mc4-cased-allnli_tr>`__ |external-link|
   * - huggingface-zstc-emrecan-convbert-base-turkish-mc4-cased-multinli-tr
     - False
     - 1.0.0
     - 2.81.0
     - Source
     - `HuggingFace <https://huggingface.co/emrecan/convbert-base-turkish-mc4-cased-multinli_tr>`__ |external-link|
   * - huggingface-zstc-emrecan-convbert-base-turkish-mc4-cased-snli-tr
     - False
     - 1.0.0
     - 2.81.0
     - Source
     - `HuggingFace <https://huggingface.co/emrecan/convbert-base-turkish-mc4-cased-snli_tr>`__ |external-link|
   * - huggingface-zstc-emrecan-distilbert-base-turkish-cased-allnli-tr
     - False
     - 1.0.0
     - 2.81.0
     - Source
     - `HuggingFace <https://huggingface.co/emrecan/distilbert-base-turkish-cased-allnli_tr>`__ |external-link|
   * - huggingface-zstc-emrecan-distilbert-base-turkish-cased-multinli-tr
     - False
     - 1.0.0
     - 2.81.0
     - Source
     - `HuggingFace <https://huggingface.co/emrecan/distilbert-base-turkish-cased-multinli_tr>`__ |external-link|
   * - huggingface-zstc-emrecan-distilbert-base-turkish-cased-snli-tr
     - False
     - 1.0.0
     - 2.81.0
     - Source
     - `HuggingFace <https://huggingface.co/emrecan/distilbert-base-turkish-cased-snli_tr>`__ |external-link|
   * - huggingface-zstc-facebook-bart-large-mnli
     - False
     - 1.0.0
     - 2.81.0
     - Source
     - `HuggingFace <https://huggingface.co/facebook/bart-large-mnli>`__ |external-link|
   * - huggingface-zstc-jiva-xlm-roberta-large-it-mnli
     - False
     - 1.0.0
     - 2.81.0
     - Source
     - `HuggingFace <https://huggingface.co/Jiva/xlm-roberta-large-it-mnli>`__ |external-link|
   * - huggingface-zstc-lighteternal-nli-xlm-r-greek
     - False
     - 1.0.0
     - 2.81.0
     - Source
     - `HuggingFace <https://huggingface.co/lighteternal/nli-xlm-r-greek>`__ |external-link|
   * - huggingface-zstc-moritzlaurer-deberta-v3-large-mnli-fever-anli-ling-wanli
     - False
     - 1.0.0
     - 2.81.0
     - Source
     - `HuggingFace <https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli>`__ |external-link|
   * - huggingface-zstc-moritzlaurer-mdeberta-v3-base-xnli-multilingual-nli-2mil7
     - False
     - 1.0.0
     - 2.81.0
     - Source
     - `HuggingFace <https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7>`__ |external-link|
   * - huggingface-zstc-narsil-bart-large-mnli-opti
     - False
     - 1.0.0
     - 2.81.0
     - Source
     - `HuggingFace <https://huggingface.co/Narsil/bart-large-mnli-opti>`__ |external-link|
   * - huggingface-zstc-narsil-deberta-large-mnli-zero-cls
     - False
     - 1.0.0
     - 2.81.0
     - Source
     - `HuggingFace <https://huggingface.co/Narsil/deberta-large-mnli-zero-cls>`__ |external-link|
   * - huggingface-zstc-navteca-bart-large-mnli
     - False
     - 1.0.0
     - 2.81.0
     - Source
     - `HuggingFace <https://huggingface.co/navteca/bart-large-mnli>`__ |external-link|
   * - huggingface-zstc-recognai-bert-base-spanish-wwm-cased-xnli
     - False
     - 1.0.0
     - 2.81.0
     - Source
     - `HuggingFace <https://huggingface.co/Recognai/bert-base-spanish-wwm-cased-xnli>`__ |external-link|
   * - huggingface-zstc-recognai-zeroshot-selectra-medium
     - False
     - 1.0.0
     - 2.81.0
     - Source
     - `HuggingFace <https://huggingface.co/Recognai/zeroshot_selectra_medium>`__ |external-link|
   * - huggingface-zstc-recognai-zeroshot-selectra-small
     - False
     - 1.0.0
     - 2.81.0
     - Source
     - `HuggingFace <https://huggingface.co/Recognai/zeroshot_selectra_small>`__ |external-link|
   * - lightgbm-classification-model
     - True
     - 1.5.1
     - 2.75.0
     - Classification
     - `LightGBM <https://lightgbm.readthedocs.io/en/latest/>`__ |external-link|
   * - lightgbm-regression-model
     - True
     - 1.5.1
     - 2.75.0
     - Regression
     - `LightGBM <https://lightgbm.readthedocs.io/en/latest/>`__ |external-link|
   * - meta-textgeneration-llama-2-13b
     - True
     - 2.1.1
     - 2.174.0
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-2-13b-f
     - False
     - 1.2.0
     - 2.174.0
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-2-70b
     - True
     - 2.0.0
     - 2.174.0
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-2-70b-f
     - False
     - 1.2.0
     - 2.174.0
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-2-7b
     - True
     - 2.1.1
     - 2.174.0
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-2-7b-f
     - False
     - 1.2.0
     - 2.174.0
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - model-depth2img-stable-diffusion-2-depth-fp16
     - False
     - 1.0.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/stabilityai/stable-diffusion-2-depth>`__ |external-link|
   * - model-depth2img-stable-diffusion-v1-5-controlnet
     - False
     - 1.0.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/lllyasviel/sd-controlnet-depth>`__ |external-link|
   * - model-depth2img-stable-diffusion-v1-5-controlnet-fp16
     - False
     - 1.0.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/lllyasviel/sd-controlnet-depth>`__ |external-link|
   * - model-depth2img-stable-diffusion-v1-5-controlnet-v1-1
     - False
     - 1.0.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth>`__ |external-link|
   * - model-depth2img-stable-diffusion-v1-5-controlnet-v1-1-fp16
     - False
     - 1.0.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth>`__ |external-link|
   * - model-depth2img-stable-diffusion-v2-1-controlnet
     - False
     - 1.0.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/thibaud/controlnet-sd21-depth-diffusers>`__ |external-link|
   * - model-depth2img-stable-diffusion-v2-1-controlnet-fp16
     - False
     - 1.0.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/thibaud/controlnet-sd21-depth-diffusers>`__ |external-link|
   * - model-inpainting-runwayml-stable-diffusion-inpainting
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/runwayml/stable-diffusion-inpainting>`__ |external-link|
   * - model-inpainting-runwayml-stable-diffusion-inpainting-fp16
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/runwayml/stable-diffusion-inpainting>`__ |external-link|
   * - model-inpainting-stabilityai-stable-diffusion-2-inpainting
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/stabilityai/stable-diffusion-2-inpainting>`__ |external-link|
   * - model-inpainting-stabilityai-stable-diffusion-2-inpainting-fp16
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/stabilityai/stable-diffusion-2-inpainting>`__ |external-link|
   * - model-txt2img-stabilityai-stable-diffusion-v1-4
     - False
     - 1.3.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/CompVis/stable-diffusion-v1-4>`__ |external-link|
   * - model-txt2img-stabilityai-stable-diffusion-v1-4-fp16
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/CompVis/stable-diffusion-v1-4>`__ |external-link|
   * - model-txt2img-stabilityai-stable-diffusion-v2
     - False
     - 1.2.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/stabilityai/stable-diffusion-2>`__ |external-link|
   * - model-txt2img-stabilityai-stable-diffusion-v2-1-base
     - True
     - 1.1.3
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/stabilityai/stable-diffusion-2-1-base>`__ |external-link|
   * - model-txt2img-stabilityai-stable-diffusion-v2-fp16
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/stabilityai/stable-diffusion-2>`__ |external-link|
   * - model-upscaling-stabilityai-stable-diffusion-x4-upscaler-fp16
     - False
     - 1.1.0
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler>`__ |external-link|
   * - mxnet-is-mask-rcnn-fpn-resnet101-v1d-coco
     - False
     - 1.2.1
     - 2.100.0
     - Instance Segmentation
     - `GluonCV <https://cv.gluon.ai/model_zoo/segmentation.html>`__ |external-link|
   * - mxnet-is-mask-rcnn-fpn-resnet18-v1b-coco
     - False
     - 1.2.1
     - 2.100.0
     - Instance Segmentation
     - `GluonCV <https://cv.gluon.ai/model_zoo/segmentation.html>`__ |external-link|
   * - mxnet-is-mask-rcnn-fpn-resnet50-v1b-coco
     - False
     - 1.2.1
     - 2.100.0
     - Instance Segmentation
     - `GluonCV <https://cv.gluon.ai/model_zoo/segmentation.html>`__ |external-link|
   * - mxnet-is-mask-rcnn-resnet18-v1b-coco
     - False
     - 1.2.1
     - 2.100.0
     - Instance Segmentation
     - `GluonCV <https://cv.gluon.ai/model_zoo/segmentation.html>`__ |external-link|
   * - mxnet-od-faster-rcnn-fpn-resnet101-v1d-coco
     - False
     - 1.2.1
     - 2.100.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-faster-rcnn-fpn-resnet50-v1b-coco
     - False
     - 1.2.1
     - 2.100.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-faster-rcnn-resnet101-v1d-coco
     - False
     - 1.2.1
     - 2.100.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-faster-rcnn-resnet50-v1b-coco
     - False
     - 1.2.1
     - 2.100.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-faster-rcnn-resnet50-v1b-voc
     - False
     - 1.2.1
     - 2.100.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-ssd-300-vgg16-atrous-coco
     - True
     - 1.3.2
     - 2.100.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-ssd-300-vgg16-atrous-voc
     - True
     - 1.3.2
     - 2.100.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-ssd-512-mobilenet1-0-coco
     - True
     - 1.3.2
     - 2.100.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-ssd-512-mobilenet1-0-voc
     - True
     - 1.3.2
     - 2.100.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-ssd-512-resnet50-v1-coco
     - True
     - 1.3.2
     - 2.100.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-ssd-512-resnet50-v1-voc
     - True
     - 1.3.2
     - 2.100.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-ssd-512-vgg16-atrous-coco
     - True
     - 1.3.2
     - 2.100.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-ssd-512-vgg16-atrous-voc
     - True
     - 1.3.2
     - 2.100.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-yolo3-darknet53-coco
     - False
     - 1.2.1
     - 2.100.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-yolo3-darknet53-voc
     - False
     - 1.2.1
     - 2.100.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-yolo3-mobilenet1-0-coco
     - False
     - 1.2.1
     - 2.100.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-yolo3-mobilenet1-0-voc
     - False
     - 1.2.1
     - 2.100.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-semseg-fcn-resnet101-ade
     - True
     - 1.4.2
     - 2.100.0
     - Semantic Segmentation
     - `GluonCV <https://cv.gluon.ai/model_zoo/segmentation.html>`__ |external-link|
   * - mxnet-semseg-fcn-resnet101-coco
     - True
     - 1.4.2
     - 2.100.0
     - Semantic Segmentation
     - `GluonCV <https://cv.gluon.ai/model_zoo/segmentation.html>`__ |external-link|
   * - mxnet-semseg-fcn-resnet101-voc
     - True
     - 1.4.2
     - 2.100.0
     - Semantic Segmentation
     - `GluonCV <https://cv.gluon.ai/model_zoo/segmentation.html>`__ |external-link|
   * - mxnet-semseg-fcn-resnet50-ade
     - True
     - 1.4.2
     - 2.100.0
     - Semantic Segmentation
     - `GluonCV <https://cv.gluon.ai/model_zoo/segmentation.html>`__ |external-link|
   * - mxnet-tcembedding-robertafin-base-uncased
     - False
     - 1.2.1
     - 2.100.0
     - Text Embedding
     - `GluonCV <https://nlp.gluon.ai/master/_modules/gluonnlp/models/roberta.html>`__ |external-link|
   * - mxnet-tcembedding-robertafin-base-wiki-uncased
     - False
     - 1.2.1
     - 2.100.0
     - Text Embedding
     - `GluonCV <https://nlp.gluon.ai/master/_modules/gluonnlp/models/roberta.html>`__ |external-link|
   * - mxnet-tcembedding-robertafin-large-uncased
     - False
     - 1.2.1
     - 2.100.0
     - Text Embedding
     - `GluonCV <https://nlp.gluon.ai/master/_modules/gluonnlp/models/roberta.html>`__ |external-link|
   * - mxnet-tcembedding-robertafin-large-wiki-uncased
     - False
     - 1.2.1
     - 2.100.0
     - Text Embedding
     - `GluonCV <https://nlp.gluon.ai/master/_modules/gluonnlp/models/roberta.html>`__ |external-link|
   * - pytorch-eqa-bert-base-cased
     - True
     - 1.2.1
     - 2.75.0
     - Question Answering
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__ |external-link|
   * - pytorch-eqa-bert-base-multilingual-cased
     - True
     - 1.2.1
     - 2.75.0
     - Question Answering
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__ |external-link|
   * - pytorch-eqa-bert-base-multilingual-uncased
     - True
     - 1.2.1
     - 2.75.0
     - Question Answering
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__ |external-link|
   * - pytorch-eqa-bert-base-uncased
     - True
     - 1.2.1
     - 2.75.0
     - Question Answering
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__ |external-link|
   * - pytorch-eqa-bert-large-cased
     - True
     - 1.2.1
     - 2.75.0
     - Question Answering
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__ |external-link|
   * - pytorch-eqa-bert-large-cased-whole-word-masking
     - True
     - 1.2.1
     - 2.75.0
     - Question Answering
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__ |external-link|
   * - pytorch-eqa-bert-large-cased-whole-word-masking-finetuned-squad
     - True
     - 1.2.1
     - 2.75.0
     - Question Answering
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__ |external-link|
   * - pytorch-eqa-bert-large-uncased
     - True
     - 1.2.1
     - 2.75.0
     - Question Answering
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__ |external-link|
   * - pytorch-eqa-bert-large-uncased-whole-word-masking
     - True
     - 1.2.1
     - 2.75.0
     - Question Answering
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__ |external-link|
   * - pytorch-eqa-bert-large-uncased-whole-word-masking-finetuned-squad
     - True
     - 1.2.1
     - 2.75.0
     - Question Answering
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__ |external-link|
   * - pytorch-eqa-distilbert-base-cased
     - True
     - 1.2.1
     - 2.75.0
     - Question Answering
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__ |external-link|
   * - pytorch-eqa-distilbert-base-multilingual-cased
     - True
     - 1.2.1
     - 2.75.0
     - Question Answering
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__ |external-link|
   * - pytorch-eqa-distilbert-base-uncased
     - True
     - 1.2.1
     - 2.75.0
     - Question Answering
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__ |external-link|
   * - pytorch-eqa-distilroberta-base
     - True
     - 1.2.1
     - 2.75.0
     - Question Answering
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__ |external-link|
   * - pytorch-eqa-roberta-base
     - True
     - 1.2.1
     - 2.75.0
     - Question Answering
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__ |external-link|
   * - pytorch-eqa-roberta-base-openai-detector
     - True
     - 1.2.1
     - 2.75.0
     - Question Answering
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__ |external-link|
   * - pytorch-eqa-roberta-large
     - True
     - 1.2.1
     - 2.75.0
     - Question Answering
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__ |external-link|
   * - pytorch-eqa-roberta-large-openai-detector
     - True
     - 1.2.1
     - 2.75.0
     - Question Answering
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__ |external-link|
   * - pytorch-ic-alexnet
     - True
     - 2.2.4
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_alexnet/>`__ |external-link|
   * - pytorch-ic-densenet121
     - True
     - 2.2.4
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_densenet/>`__ |external-link|
   * - pytorch-ic-densenet161
     - True
     - 2.2.4
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_densenet/>`__ |external-link|
   * - pytorch-ic-densenet169
     - True
     - 2.2.4
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_densenet/>`__ |external-link|
   * - pytorch-ic-densenet201
     - True
     - 2.2.4
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_densenet/>`__ |external-link|
   * - pytorch-ic-googlenet
     - True
     - 2.2.4
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_googlenet/>`__ |external-link|
   * - pytorch-ic-mobilenet-v2
     - True
     - 2.2.4
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_mobilenet_v2/>`__ |external-link|
   * - pytorch-ic-resnet101
     - True
     - 2.2.4
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_resnet/>`__ |external-link|
   * - pytorch-ic-resnet152
     - True
     - 2.2.4
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_resnet/>`__ |external-link|
   * - pytorch-ic-resnet18
     - True
     - 2.2.4
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_resnet/>`__ |external-link|
   * - pytorch-ic-resnet34
     - True
     - 2.2.4
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_resnet/>`__ |external-link|
   * - pytorch-ic-resnet50
     - True
     - 2.2.4
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_resnet/>`__ |external-link|
   * - pytorch-ic-resnext101-32x8d
     - True
     - 2.2.4
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_resnext/>`__ |external-link|
   * - pytorch-ic-resnext50-32x4d
     - True
     - 2.2.4
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_resnext/>`__ |external-link|
   * - pytorch-ic-shufflenet-v2-x1-0
     - True
     - 2.2.4
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_shufflenet_v2/>`__ |external-link|
   * - pytorch-ic-squeezenet1-0
     - True
     - 2.2.4
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_squeezenet/>`__ |external-link|
   * - pytorch-ic-squeezenet1-1
     - True
     - 2.2.4
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_squeezenet/>`__ |external-link|
   * - pytorch-ic-vgg11
     - True
     - 2.2.4
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_vgg/>`__ |external-link|
   * - pytorch-ic-vgg11-bn
     - True
     - 2.2.4
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_vgg/>`__ |external-link|
   * - pytorch-ic-vgg13
     - True
     - 2.2.4
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_vgg/>`__ |external-link|
   * - pytorch-ic-vgg13-bn
     - True
     - 2.2.4
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_vgg/>`__ |external-link|
   * - pytorch-ic-vgg16
     - True
     - 2.2.4
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_vgg/>`__ |external-link|
   * - pytorch-ic-vgg16-bn
     - True
     - 2.2.4
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_vgg/>`__ |external-link|
   * - pytorch-ic-vgg19
     - True
     - 2.2.4
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_vgg/>`__ |external-link|
   * - pytorch-ic-vgg19-bn
     - True
     - 2.2.4
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_vgg/>`__ |external-link|
   * - pytorch-ic-wide-resnet101-2
     - True
     - 2.2.4
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_wide_resnet/>`__ |external-link|
   * - pytorch-ic-wide-resnet50-2
     - True
     - 2.2.4
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_wide_resnet/>`__ |external-link|
   * - pytorch-od-nvidia-ssd
     - False
     - 1.0.3
     - 2.75.0
     - Object Detection
     - `Pytorch Hub <https://pytorch.org/hub/nvidia_deeplearningexamples_ssd/>`__ |external-link|
   * - pytorch-od1-fasterrcnn-mobilenet-v3-large-320-fpn
     - False
     - 1.0.0
     - 2.75.0
     - Object Detection
     - `Pytorch Hub <https://pytorch.org/vision/stable/_modules/torchvision/models/detection/faster_rcnn.html>`__ |external-link|
   * - pytorch-od1-fasterrcnn-mobilenet-v3-large-fpn
     - False
     - 1.0.0
     - 2.75.0
     - Object Detection
     - `Pytorch Hub <https://pytorch.org/vision/stable/_modules/torchvision/models/detection/faster_rcnn.html>`__ |external-link|
   * - pytorch-od1-fasterrcnn-resnet50-fpn
     - True
     - 1.3.2
     - 2.75.0
     - Object Detection
     - `Pytorch Hub <https://pytorch.org/vision/stable/_modules/torchvision/models/detection/faster_rcnn.html>`__ |external-link|
   * - pytorch-tabtransformerclassification-model
     - True
     - 1.0.5
     - 2.75.0
     - Source
     - `Source <https://arxiv.org/abs/2012.06678>`__ |external-link|
   * - pytorch-tabtransformerregression-model
     - True
     - 1.0.3
     - 2.75.0
     - Source
     - `Source <https://arxiv.org/abs/2012.06678>`__ |external-link|
   * - pytorch-textgeneration1-alexa20b
     - False
     - 1.0.0
     - 2.144.0
     - Source
     - `Source <https://www.amazon.science/blog/20b-parameter-alexa-model-sets-new-marks-in-few-shot-learning>`__ |external-link|
   * - sklearn-classification-linear
     - True
     - 1.2.0
     - 2.139.0
     - Classification
     - `ScikitLearn <https://scikit-learn.org/stable/>`__ |external-link|
   * - sklearn-classification-snowflake
     - True
     - 1.0.0
     - 2.139.0
     - Classification
     - `ScikitLearn <https://scikit-learn.org/stable/>`__ |external-link|
   * - sklearn-regression-linear
     - True
     - 1.2.0
     - 2.139.0
     - Regression
     - `ScikitLearn <https://scikit-learn.org/stable/>`__ |external-link|
   * - sklearn-regression-snowflake
     - True
     - 1.0.0
     - 2.139.0
     - Regression
     - `ScikitLearn <https://scikit-learn.org/stable/>`__ |external-link|
   * - tensorflow-audioembedding-frill-1
     - False
     - 1.0.1
     - 2.80.0
     - Source
     - `Tensorflow Hub <https://tfhub.dev/google/nonsemantic-speech-benchmark/frill/1>`__ |external-link|
   * - tensorflow-audioembedding-trill-3
     - False
     - 1.0.1
     - 2.80.0
     - Source
     - `Tensorflow Hub <https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/3>`__ |external-link|
   * - tensorflow-audioembedding-trill-distilled-3
     - False
     - 1.0.1
     - 2.80.0
     - Source
     - `Tensorflow Hub <https://tfhub.dev/google/nonsemantic-speech-benchmark/trill-distilled/3>`__ |external-link|
   * - tensorflow-audioembedding-trillsson1-1
     - False
     - 1.0.1
     - 2.80.0
     - Source
     - `Tensorflow Hub <https://tfhub.dev/google/nonsemantic-speech-benchmark/trillsson1/1>`__ |external-link|
   * - tensorflow-audioembedding-trillsson2-1
     - False
     - 1.0.1
     - 2.80.0
     - Source
     - `Tensorflow Hub <https://tfhub.dev/google/nonsemantic-speech-benchmark/trillsson2/1>`__ |external-link|
   * - tensorflow-audioembedding-trillsson3-1
     - False
     - 1.0.1
     - 2.80.0
     - Source
     - `Tensorflow Hub <https://tfhub.dev/google/nonsemantic-speech-benchmark/trillsson3/1>`__ |external-link|
   * - tensorflow-ic-bit-m-r101x1-ilsvrc2012-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r101x1/ilsvrc2012_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-m-r101x1-imagenet21k-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r101x1/imagenet21k_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-m-r101x3-ilsvrc2012-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r101x3/ilsvrc2012_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-m-r101x3-imagenet21k-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r101x3/imagenet21k_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-m-r152x4-ilsvrc2012
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r152x4/ilsvrc2012_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-m-r152x4-imagenet21k
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r152x4/imagenet21k_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-m-r50x1-ilsvrc2012-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r50x1/ilsvrc2012_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-m-r50x1-imagenet21k-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r50x1/imagenet21k_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-m-r50x3-ilsvrc2012-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r50x3/ilsvrc2012_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-m-r50x3-imagenet21k-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r50x3/imagenet21k_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-s-r101x1-ilsvrc2012-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r101x1/ilsvrc2012_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-s-r101x3-ilsvrc2012-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r101x3/ilsvrc2012_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-s-r152x4-ilsvrc2012
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r152x4/ilsvrc2012_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-s-r50x1-ilsvrc2012-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r50x1/ilsvrc2012_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-s-r50x3-ilsvrc2012-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r50x3/ilsvrc2012_classification/1>`__ |external-link|
   * - tensorflow-ic-cait-m36-384
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/cait_m36_384/1>`__ |external-link|
   * - tensorflow-ic-cait-m48-448
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/cait_m48_448/1>`__ |external-link|
   * - tensorflow-ic-cait-s24-224
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/cait_s24_224/1>`__ |external-link|
   * - tensorflow-ic-cait-s24-384
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/cait_s24_384/1>`__ |external-link|
   * - tensorflow-ic-cait-s36-384
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/cait_s36_384/1>`__ |external-link|
   * - tensorflow-ic-cait-xs24-384
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/cait_xs24_384/1>`__ |external-link|
   * - tensorflow-ic-cait-xxs24-224
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/cait_xxs24_224/1>`__ |external-link|
   * - tensorflow-ic-cait-xxs24-384
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/cait_xxs24_384/1>`__ |external-link|
   * - tensorflow-ic-cait-xxs36-224
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/cait_xxs36_224/1>`__ |external-link|
   * - tensorflow-ic-cait-xxs36-384
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/cait_xxs36_384/1>`__ |external-link|
   * - tensorflow-ic-deit-base-distilled-patch16-224
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/deit_base_distilled_patch16_224/1>`__ |external-link|
   * - tensorflow-ic-deit-base-distilled-patch16-384
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/deit_base_distilled_patch16_384/1>`__ |external-link|
   * - tensorflow-ic-deit-base-patch16-224
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/deit_base_patch16_224/1>`__ |external-link|
   * - tensorflow-ic-deit-base-patch16-384
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/deit_base_patch16_384/1>`__ |external-link|
   * - tensorflow-ic-deit-small-distilled-patch16-224
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/deit_small_distilled_patch16_224/1>`__ |external-link|
   * - tensorflow-ic-deit-small-patch16-224
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/deit_small_patch16_224/1>`__ |external-link|
   * - tensorflow-ic-deit-tiny-distilled-patch16-224
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/deit_tiny_distilled_patch16_224/1>`__ |external-link|
   * - tensorflow-ic-deit-tiny-patch16-224
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/deit_tiny_patch16_224/1>`__ |external-link|
   * - tensorflow-ic-efficientnet-b0-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/b0/classification/1>`__ |external-link|
   * - tensorflow-ic-efficientnet-b1-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/b1/classification/1>`__ |external-link|
   * - tensorflow-ic-efficientnet-b2-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/b2/classification/1>`__ |external-link|
   * - tensorflow-ic-efficientnet-b3-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/b3/classification/1>`__ |external-link|
   * - tensorflow-ic-efficientnet-b4-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/b4/classification/1>`__ |external-link|
   * - tensorflow-ic-efficientnet-b5-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/b5/classification/1>`__ |external-link|
   * - tensorflow-ic-efficientnet-b6-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/b6/classification/1>`__ |external-link|
   * - tensorflow-ic-efficientnet-b7-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/b7/classification/1>`__ |external-link|
   * - tensorflow-ic-efficientnet-lite0-classification-2
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite0/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-lite1-classification-2
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite1/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-lite2-classification-2
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite2/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-lite3-classification-2
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite3/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-lite4-classification-2
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite4/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet1k-b0
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet1k-b1
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b1/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet1k-b2
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b2/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet1k-b3
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b3/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet1k-l
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_l/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet1k-m
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_m/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet1k-s
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet21k-b0
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b0/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet21k-b1
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b1/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet21k-b2
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b2/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet21k-b3
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b3/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet21k-ft1k-b0
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b0/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet21k-ft1k-b1
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b1/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet21k-ft1k-b2
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b2/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet21k-ft1k-b3
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b3/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet21k-ft1k-l
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_l/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet21k-ft1k-m
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_m/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet21k-ft1k-s
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_s/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet21k-ft1k-xl
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_xl/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet21k-l
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_l/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet21k-m
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_m/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet21k-s
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_s/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet21k-xl
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_xl/classification/2>`__ |external-link|
   * - tensorflow-ic-imagenet-inception-resnet-v2-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-inception-v1-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/inception_v1/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-inception-v2-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/inception_v2/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-inception-v3-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/inception_v3/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-025-128-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_128/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-025-160-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_160/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-025-192-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_192/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-025-224-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-050-128-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_128/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-050-160-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-050-192-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_192/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-050-224-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_224/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-075-128-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_128/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-075-160-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_160/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-075-192-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_192/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-075-224-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_224/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-100-128-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_128/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-100-160-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_160/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-100-192-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_192/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-100-224-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-035-128
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_035_128/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-035-160
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_035_160/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-035-192
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_035_192/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-035-224-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_035_224/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-035-96
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_035_96/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-050-128
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_050_128/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-050-160
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_050_160/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-050-192
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_050_192/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-050-224-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_050_224/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-050-96
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_050_96/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-075-128
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_075_128/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-075-160
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_075_160/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-075-192
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_075_192/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-075-224-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_075_224/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-075-96
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_075_96/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-100-160
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_100_160/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-100-192
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_100_192/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-100-224-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-100-96
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-130-224-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-140-224-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v3-large-075-224
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v3-large-100-224
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v3-small-075-224
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v3_small_075_224/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v3-small-100-224
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-nasnet-large
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/nasnet_large/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-nasnet-mobile
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/nasnet_mobile/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-pnasnet-large
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/pnasnet_large/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-resnet-v1-101-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v1_101/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-resnet-v1-152-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v1_152/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-resnet-v1-50-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v1_50/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-resnet-v2-101-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v2_101/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-resnet-v2-152-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v2_152/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-resnet-v2-50-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v2_50/classification/5>`__ |external-link|
   * - tensorflow-ic-resnet-50-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/resnet_50/classification/1>`__ |external-link|
   * - tensorflow-ic-swin-base-patch4-window12-384
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/swin_base_patch4_window12_384/1>`__ |external-link|
   * - tensorflow-ic-swin-base-patch4-window7-224
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/swin_base_patch4_window7_224/1>`__ |external-link|
   * - tensorflow-ic-swin-large-patch4-window12-384
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/swin_large_patch4_window12_384/1>`__ |external-link|
   * - tensorflow-ic-swin-large-patch4-window7-224
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/swin_large_patch4_window7_224/1>`__ |external-link|
   * - tensorflow-ic-swin-s3-base-224
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/swin_s3_base_224/1>`__ |external-link|
   * - tensorflow-ic-swin-s3-small-224
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/swin_s3_small_224/1>`__ |external-link|
   * - tensorflow-ic-swin-s3-tiny-224
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/swin_s3_tiny_224/1>`__ |external-link|
   * - tensorflow-ic-swin-small-patch4-window7-224
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/swin_small_patch4_window7_224/1>`__ |external-link|
   * - tensorflow-ic-swin-tiny-patch4-window7-224
     - True
     - 1.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/swin_tiny_patch4_window7_224/1>`__ |external-link|
   * - tensorflow-ic-tf2-preview-inception-v3-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/tf2-preview/inception_v3/classification/4>`__ |external-link|
   * - tensorflow-ic-tf2-preview-mobilenet-v2-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4>`__ |external-link|
   * - tensorflow-icembedding-bit-m-r101x1-ilsvrc2012-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r101x1/1>`__ |external-link|
   * - tensorflow-icembedding-bit-m-r101x3-imagenet21k-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r101x3/1>`__ |external-link|
   * - tensorflow-icembedding-bit-m-r50x1-ilsvrc2012-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r50x1/1>`__ |external-link|
   * - tensorflow-icembedding-bit-m-r50x3-imagenet21k-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r50x3/1>`__ |external-link|
   * - tensorflow-icembedding-bit-s-r101x1-ilsvrc2012-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r101x1/1>`__ |external-link|
   * - tensorflow-icembedding-bit-s-r101x3-ilsvrc2012-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r101x3/1>`__ |external-link|
   * - tensorflow-icembedding-bit-s-r50x1-ilsvrc2012-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r50x1/1>`__ |external-link|
   * - tensorflow-icembedding-bit-s-r50x3-ilsvrc2012-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r50x3/1>`__ |external-link|
   * - tensorflow-icembedding-efficientnet-b0-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b0/feature-vector/1>`__ |external-link|
   * - tensorflow-icembedding-efficientnet-b1-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b1/feature-vector/1>`__ |external-link|
   * - tensorflow-icembedding-efficientnet-b2-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b2/feature-vector/1>`__ |external-link|
   * - tensorflow-icembedding-efficientnet-b3-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b3/feature-vector/1>`__ |external-link|
   * - tensorflow-icembedding-efficientnet-b6-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b6/feature-vector/1>`__ |external-link|
   * - tensorflow-icembedding-efficientnet-lite0-featurevector-2
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2>`__ |external-link|
   * - tensorflow-icembedding-efficientnet-lite1-featurevector-2
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite1/feature-vector/2>`__ |external-link|
   * - tensorflow-icembedding-efficientnet-lite2-featurevector-2
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite2/feature-vector/2>`__ |external-link|
   * - tensorflow-icembedding-efficientnet-lite3-featurevector-2
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite3/feature-vector/2>`__ |external-link|
   * - tensorflow-icembedding-efficientnet-lite4-featurevector-2
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite4/feature-vector/2>`__ |external-link|
   * - tensorflow-icembedding-imagenet-inception-v1-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/inception_v1/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-inception-v2-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/inception_v2/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-inception-v3-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-025-128-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_128/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-025-160-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_160/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-025-192-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_192/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-025-224-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-050-128-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_128/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-050-160-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-050-192-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_192/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-050-224-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-075-128-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_128/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-075-160-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_160/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-075-192-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_192/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-075-224-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-100-128-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_128/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-100-160-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_160/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-100-192-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_192/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-100-224-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v2-035-224-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_035_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v2-050-224-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_050_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v2-075-224-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_075_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v2-100-224-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v2-130-224-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v2-140-224-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-resnet-v1-101-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v1_101/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-resnet-v1-152-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v1_152/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-resnet-v1-50-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-resnet-v2-101-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-resnet-v2-152-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-resnet-v2-50-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-resnet-50-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/resnet_50/feature_vector/1>`__ |external-link|
   * - tensorflow-icembedding-tf2-preview-inception-v3-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-tf2-preview-mobilenet-v2-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4>`__ |external-link|
   * - tensorflow-od-centernet-hourglass-1024x1024-1
     - False
     - 2.0.2
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/hourglass_1024x1024/1>`__ |external-link|
   * - tensorflow-od-centernet-hourglass-1024x1024-kpts-1
     - False
     - 2.0.2
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/hourglass_1024x1024_kpts/1>`__ |external-link|
   * - tensorflow-od-centernet-hourglass-512x512-1
     - False
     - 2.0.2
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1>`__ |external-link|
   * - tensorflow-od-centernet-hourglass-512x512-kpts-1
     - False
     - 2.0.2
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/hourglass_512x512_kpts/1>`__ |external-link|
   * - tensorflow-od-centernet-resnet101v1-fpn-512x512-1
     - False
     - 2.0.2
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/resnet101v1_fpn_512x512/1>`__ |external-link|
   * - tensorflow-od-centernet-resnet50v1-fpn-512x512-1
     - False
     - 2.0.2
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512/1>`__ |external-link|
   * - tensorflow-od-centernet-resnet50v1-fpn-512x512-kpts-1
     - False
     - 2.0.2
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512_kpts/1>`__ |external-link|
   * - tensorflow-od-centernet-resnet50v2-512x512-1
     - False
     - 2.0.2
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/resnet50v2_512x512/1>`__ |external-link|
   * - tensorflow-od-centernet-resnet50v2-512x512-kpts-1
     - False
     - 2.0.2
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/resnet50v2_512x512_kpts/1>`__ |external-link|
   * - tensorflow-od-efficientdet-d0-1
     - False
     - 2.0.2
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientdet/d0/1>`__ |external-link|
   * - tensorflow-od-efficientdet-d1-1
     - False
     - 2.0.2
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientdet/d1/1>`__ |external-link|
   * - tensorflow-od-efficientdet-d2-1
     - False
     - 2.0.2
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientdet/d2/1>`__ |external-link|
   * - tensorflow-od-efficientdet-d3-1
     - False
     - 2.0.2
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientdet/d3/1>`__ |external-link|
   * - tensorflow-od-efficientdet-d4-1
     - False
     - 2.0.2
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientdet/d4/1>`__ |external-link|
   * - tensorflow-od-efficientdet-d5-1
     - False
     - 2.0.2
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientdet/d5/1>`__ |external-link|
   * - tensorflow-od-faster-rcnn-inception-resnet-v2-1024x1024-1
     - False
     - 2.0.2
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_1024x1024/1>`__ |external-link|
   * - tensorflow-od-faster-rcnn-inception-resnet-v2-640x640-1
     - False
     - 2.0.2
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1>`__ |external-link|
   * - tensorflow-od-faster-rcnn-resnet101-v1-1024x1024-1
     - False
     - 2.0.2
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_1024x1024/1>`__ |external-link|
   * - tensorflow-od-faster-rcnn-resnet101-v1-640x640-1
     - False
     - 2.0.2
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_640x640/1>`__ |external-link|
   * - tensorflow-od-faster-rcnn-resnet101-v1-800x1333-1
     - False
     - 2.0.2
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_800x1333/1>`__ |external-link|
   * - tensorflow-od-faster-rcnn-resnet152-v1-1024x1024-1
     - False
     - 2.0.2
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_1024x1024/1>`__ |external-link|
   * - tensorflow-od-faster-rcnn-resnet152-v1-640x640-1
     - False
     - 2.0.2
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_640x640/1>`__ |external-link|
   * - tensorflow-od-faster-rcnn-resnet152-v1-800x1333-1
     - False
     - 2.0.2
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_800x1333/1>`__ |external-link|
   * - tensorflow-od-faster-rcnn-resnet50-v1-1024x1024-1
     - False
     - 2.0.2
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_1024x1024/1>`__ |external-link|
   * - tensorflow-od-faster-rcnn-resnet50-v1-640x640-1
     - False
     - 2.0.2
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1>`__ |external-link|
   * - tensorflow-od-faster-rcnn-resnet50-v1-800x1333-1
     - False
     - 2.0.2
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_800x1333/1>`__ |external-link|
   * - tensorflow-od-retinanet-resnet101-v1-fpn-1024x1024-1
     - False
     - 2.0.2
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/retinanet/resnet101_v1_fpn_1024x1024/1>`__ |external-link|
   * - tensorflow-od-retinanet-resnet101-v1-fpn-640x640-1
     - False
     - 2.0.2
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/retinanet/resnet101_v1_fpn_640x640/1>`__ |external-link|
   * - tensorflow-od-retinanet-resnet152-v1-fpn-1024x1024-1
     - False
     - 2.0.2
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/retinanet/resnet152_v1_fpn_1024x1024/1>`__ |external-link|
   * - tensorflow-od-retinanet-resnet152-v1-fpn-640x640-1
     - False
     - 2.0.2
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/retinanet/resnet152_v1_fpn_640x640/1>`__ |external-link|
   * - tensorflow-od-retinanet-resnet50-v1-fpn-1024x1024-1
     - False
     - 2.0.2
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/retinanet/resnet50_v1_fpn_1024x1024/1>`__ |external-link|
   * - tensorflow-od-retinanet-resnet50-v1-fpn-640x640-1
     - False
     - 2.0.2
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/retinanet/resnet50_v1_fpn_640x640/1>`__ |external-link|
   * - tensorflow-od-ssd-mobilenet-v1-fpn-640x640-1
     - False
     - 2.0.2
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/ssd_mobilenet_v1/fpn_640x640/1>`__ |external-link|
   * - tensorflow-od-ssd-mobilenet-v2-2
     - False
     - 2.0.2
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2>`__ |external-link|
   * - tensorflow-od-ssd-mobilenet-v2-fpnlite-320x320-1
     - False
     - 2.0.2
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1>`__ |external-link|
   * - tensorflow-od-ssd-mobilenet-v2-fpnlite-640x640-1
     - False
     - 2.0.2
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_640x640/1>`__ |external-link|
   * - tensorflow-od1-ssd-efficientdet-d0-512x512-coco17-tpu-8
     - True
     - 1.2.0
     - 2.129.0
     - Object Detection
     - `Source <http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz>`__ |external-link|
   * - tensorflow-od1-ssd-efficientdet-d1-640x640-coco17-tpu-8
     - True
     - 1.2.0
     - 2.129.0
     - Object Detection
     - `Source <http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz>`__ |external-link|
   * - tensorflow-od1-ssd-efficientdet-d2-768x768-coco17-tpu-8
     - True
     - 1.2.0
     - 2.129.0
     - Object Detection
     - `Source <http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d2_coco17_tpu-32.tar.gz>`__ |external-link|
   * - tensorflow-od1-ssd-efficientdet-d3-896x896-coco17-tpu-32
     - True
     - 1.2.0
     - 2.129.0
     - Object Detection
     - `Source <http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d3_coco17_tpu-32.tar.gz>`__ |external-link|
   * - tensorflow-od1-ssd-mobilenet-v1-fpn-640x640-coco17-tpu-8
     - True
     - 1.2.0
     - 2.129.0
     - Object Detection
     - `Source <http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz>`__ |external-link|
   * - tensorflow-od1-ssd-mobilenet-v2-fpnlite-320x320-coco17-tpu-8
     - True
     - 1.2.0
     - 2.129.0
     - Object Detection
     - `Source <http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz>`__ |external-link|
   * - tensorflow-od1-ssd-mobilenet-v2-fpnlite-640x640-coco17-tpu-8
     - True
     - 1.2.0
     - 2.129.0
     - Object Detection
     - `Source <http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz>`__ |external-link|
   * - tensorflow-od1-ssd-resnet101-v1-fpn-1024x1024-coco17-tpu-8
     - True
     - 1.2.0
     - 2.129.0
     - Object Detection
     - `Source <http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8.tar.gz>`__ |external-link|
   * - tensorflow-od1-ssd-resnet101-v1-fpn-640x640-coco17-tpu-8
     - True
     - 1.2.0
     - 2.129.0
     - Object Detection
     - `Source <http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz>`__ |external-link|
   * - tensorflow-od1-ssd-resnet152-v1-fpn-1024x1024-coco17-tpu-8
     - True
     - 1.2.0
     - 2.129.0
     - Object Detection
     - `Source <http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8.tar.gz>`__ |external-link|
   * - tensorflow-od1-ssd-resnet152-v1-fpn-640x640-coco17-tpu-8
     - True
     - 1.2.0
     - 2.129.0
     - Object Detection
     - `Source <http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_640x640_coco17_tpu-8.tar.gz>`__ |external-link|
   * - tensorflow-od1-ssd-resnet50-v1-fpn-1024x1024-coco17-tpu-8
     - True
     - 1.2.0
     - 2.129.0
     - Object Detection
     - `Source <http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8.tar.gz>`__ |external-link|
   * - tensorflow-od1-ssd-resnet50-v1-fpn-640x640-coco17-tpu-8
     - True
     - 1.2.0
     - 2.129.0
     - Object Detection
     - `Source <http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz>`__ |external-link|
   * - tensorflow-spc-bert-en-cased-L-12-H-768-A-12-2
     - True
     - 1.2.3
     - 2.75.0
     - Sentence Pair Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/2>`__ |external-link|
   * - tensorflow-spc-bert-en-uncased-L-12-H-768-A-12-2
     - True
     - 1.2.3
     - 2.75.0
     - Sentence Pair Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2>`__ |external-link|
   * - tensorflow-spc-bert-en-uncased-L-24-H-1024-A-16-2
     - True
     - 1.2.3
     - 2.75.0
     - Sentence Pair Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/2>`__ |external-link|
   * - tensorflow-spc-bert-en-wwm-cased-L-24-H-1024-A-16-2
     - True
     - 1.2.3
     - 2.75.0
     - Sentence Pair Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_wwm_cased_L-24_H-1024_A-16/2>`__ |external-link|
   * - tensorflow-spc-bert-en-wwm-uncased-L-24-H-1024-A-16-2
     - True
     - 1.2.3
     - 2.75.0
     - Sentence Pair Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_wwm_uncased_L-24_H-1024_A-16/2>`__ |external-link|
   * - tensorflow-spc-bert-multi-cased-L-12-H-768-A-12-2
     - True
     - 1.2.3
     - 2.75.0
     - Sentence Pair Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/2>`__ |external-link|
   * - tensorflow-spc-electra-base-1
     - True
     - 1.2.3
     - 2.75.0
     - Sentence Pair Classification
     - `Tensorflow Hub <https://tfhub.dev/google/electra_base/1>`__ |external-link|
   * - tensorflow-spc-electra-small-1
     - True
     - 1.2.3
     - 2.75.0
     - Sentence Pair Classification
     - `Tensorflow Hub <https://tfhub.dev/google/electra_small/1>`__ |external-link|
   * - tensorflow-spc-experts-bert-pubmed-1
     - True
     - 1.2.3
     - 2.75.0
     - Sentence Pair Classification
     - `Tensorflow Hub <https://tfhub.dev/google/experts/bert/pubmed/1>`__ |external-link|
   * - tensorflow-spc-experts-bert-wiki-books-1
     - True
     - 1.2.3
     - 2.75.0
     - Sentence Pair Classification
     - `Tensorflow Hub <https://tfhub.dev/google/experts/bert/wiki_books/1>`__ |external-link|
   * - tensorflow-tc-albert-en-base
     - True
     - 1.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/albert_en_base/2>`__ |external-link|
   * - tensorflow-tc-bert-en-cased-L-12-H-768-A-12-2
     - True
     - 2.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3>`__ |external-link|
   * - tensorflow-tc-bert-en-cased-L-24-H-1024-A-16-2
     - True
     - 2.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_cased_L-24_H-1024_A-16/3>`__ |external-link|
   * - tensorflow-tc-bert-en-uncased-L-12-H-768-A-12-2
     - True
     - 2.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3>`__ |external-link|
   * - tensorflow-tc-bert-en-uncased-L-24-H-1024-A-16-2
     - True
     - 1.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/3>`__ |external-link|
   * - tensorflow-tc-bert-en-wwm-cased-L-24-H-1024-A-16-2
     - True
     - 2.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_wwm_cased_L-24_H-1024_A-16/3>`__ |external-link|
   * - tensorflow-tc-bert-en-wwm-uncased-L-24-H-1024-A-16-2
     - True
     - 2.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_wwm_uncased_L-24_H-1024_A-16/3>`__ |external-link|
   * - tensorflow-tc-bert-multi-cased-L-12-H-768-A-12-2
     - True
     - 2.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3>`__ |external-link|
   * - tensorflow-tc-electra-base-1
     - True
     - 2.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/google/electra_base/2>`__ |external-link|
   * - tensorflow-tc-electra-small-1
     - True
     - 2.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/google/electra_small/2>`__ |external-link|
   * - tensorflow-tc-experts-bert-pubmed-1
     - True
     - 2.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/google/experts/bert/pubmed/2>`__ |external-link|
   * - tensorflow-tc-experts-bert-wiki-books-1
     - True
     - 2.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/google/experts/bert/wiki_books/2>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-10-H-128-A-2
     - True
     - 1.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-10-H-256-A-4
     - True
     - 1.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-10-H-512-A-8
     - True
     - 1.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-10-H-768-A-12
     - True
     - 1.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-12-H-128-A-2
     - True
     - 1.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-12-H-256-A-4
     - True
     - 1.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-12-H-512-A-8
     - True
     - 1.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-12-H-768-A-12
     - True
     - 1.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-2-H-128-A-2
     - True
     - 1.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-2-H-256-A-4
     - True
     - 1.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-2-H-512-A-8
     - True
     - 1.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-2-H-768-A-12
     - True
     - 1.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-4-H-128-A-2
     - True
     - 1.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-4-H-256-A-4
     - True
     - 1.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-4-H-512-A-8
     - True
     - 1.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-4-H-768-A-12
     - True
     - 1.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-6-H-128-A-2
     - True
     - 1.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-6-H-256-A-4
     - True
     - 1.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-6-H-512-A-8
     - True
     - 1.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-6-H-768-A-12
     - True
     - 1.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-8-H-128-A-2
     - True
     - 1.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-8-H-256-A-4
     - True
     - 1.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-8-H-512-A-8
     - True
     - 1.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-8-H-768-A-12
     - True
     - 1.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1>`__ |external-link|
   * - tensorflow-tc-talking-heads-base
     - True
     - 1.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1>`__ |external-link|
   * - tensorflow-tc-talking-heads-large
     - True
     - 1.0.1
     - 2.80.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_large/1>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-10-H-128-A-2-2
     - False
     - 1.1.1
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-10-H-256-A-4-2
     - False
     - 1.1.1
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-10-H-512-A-8-2
     - False
     - 1.1.1
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-10-H-768-A-12-2
     - False
     - 1.1.1
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-12-H-128-A-2-2
     - False
     - 1.1.1
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-12-H-256-A-4
     - False
     - 1.1.1
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-12-H-512-A-8-2
     - False
     - 1.1.1
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-12-H-768-A-12-2
     - False
     - 1.1.1
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-12-H-768-A-12-4
     - False
     - 1.1.1
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-2-H-128-A-2-2
     - False
     - 1.1.1
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-2-H-256-A-4
     - False
     - 1.1.1
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-2-H-512-A-8-2
     - False
     - 1.1.1
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-2-H-768-A-12-2
     - False
     - 1.1.1
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-4-H-128-A-2-2
     - False
     - 1.1.1
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-4-H-256-A-4-2
     - False
     - 1.1.1
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-4-H-512-A-8-2
     - False
     - 1.1.1
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-4-H-768-A-12-2
     - False
     - 1.1.1
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-6-H-128-A-2-2
     - False
     - 1.1.1
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-6-H-256-A-4
     - False
     - 1.1.1
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-6-H-512-A-8-2
     - False
     - 1.1.1
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-6-H-768-A-12-2
     - False
     - 1.1.1
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-8-H-256-A-4-2
     - False
     - 1.1.1
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-8-H-512-A-8-2
     - False
     - 1.1.1
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-8-H-768-A-12-2
     - False
     - 1.1.1
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-wiki-books-mnli-2
     - False
     - 1.1.1
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/experts/bert/wiki_books/mnli/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-wiki-books-sst2
     - False
     - 1.1.1
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/experts/bert/wiki_books/sst2/2>`__ |external-link|
   * - tensorflow-tcembedding-talkheads-ggelu-bert-en-base-2
     - False
     - 1.1.1
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/2>`__ |external-link|
   * - tensorflow-tcembedding-talkheads-ggelu-bert-en-large-2
     - False
     - 1.1.1
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_large/2>`__ |external-link|
   * - tensorflow-tcembedding-universal-sentence-encoder-cmlm-en-base-1
     - False
     - 1.1.1
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/1>`__ |external-link|
   * - tensorflow-tcembedding-universal-sentence-encoder-cmlm-en-large-1
     - False
     - 1.1.1
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-large/1>`__ |external-link|
   * - xgboost-classification-model
     - True
     - 2.0.0
     - 2.139.0
     - Classification
     - `XGBoost <https://xgboost.readthedocs.io/en/release_1.7.0/>`__ |external-link|
   * - xgboost-classification-snowflake
     - True
     - 1.0.0
     - 2.139.0
     - Classification
     - `XGBoost <https://xgboost.readthedocs.io/en/release_1.7.0/>`__ |external-link|
   * - xgboost-regression-model
     - True
     - 2.0.0
     - 2.139.0
     - Regression
     - `XGBoost <https://xgboost.readthedocs.io/en/release_1.7.0/>`__ |external-link|
   * - xgboost-regression-snowflake
     - True
     - 1.0.0
     - 2.139.0
     - Regression
     - `XGBoost <https://xgboost.readthedocs.io/en/release_1.7.0/>`__ |external-link|
