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
     - 2.0.15
     - 2.189.0
     - Classification
     - `GluonCV <https://auto.gluon.ai/stable/index.html>`__ |external-link|
   * - autogluon-forecasting-chronos-bolt-base
     - False
     - 2.0.5
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/amazon/chronos-bolt-base>`__ |external-link|
   * - autogluon-forecasting-chronos-bolt-small
     - False
     - 1.0.8
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/amazon/chronos-bolt-small>`__ |external-link|
   * - autogluon-forecasting-chronos-t5-base
     - False
     - 2.0.8
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/amazon/chronos-t5-base>`__ |external-link|
   * - autogluon-forecasting-chronos-t5-large
     - False
     - 2.0.8
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/amazon/chronos-t5-large>`__ |external-link|
   * - autogluon-forecasting-chronos-t5-small
     - False
     - 2.0.8
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/amazon/chronos-t5-small>`__ |external-link|
   * - autogluon-regression-ensemble
     - True
     - 2.0.15
     - 2.189.0
     - Regression
     - `GluonCV <https://auto.gluon.ai/stable/index.html>`__ |external-link|
   * - catboost-classification-model
     - True
     - 2.1.21
     - 2.189.0
     - Classification
     - `Catboost <https://catboost.ai/>`__ |external-link|
   * - catboost-regression-model
     - True
     - 2.1.21
     - 2.189.0
     - Regression
     - `Catboost <https://catboost.ai/>`__ |external-link|
   * - deepseek-llm-r1
     - False
     - 4.0.7
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/deepseek-ai/DeepSeek-R1>`__ |external-link|
   * - deepseek-llm-r1-0528
     - False
     - 1.0.2
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/deepseek-ai/DeepSeek-R1-0528>`__ |external-link|
   * - deepseek-llm-r1-distill-llama-70b
     - False
     - 2.0.7
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B>`__ |external-link|
   * - deepseek-llm-r1-distill-llama-8b
     - False
     - 2.0.7
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B>`__ |external-link|
   * - deepseek-llm-r1-distill-qwen-1-5b
     - False
     - 2.0.7
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B>`__ |external-link|
   * - deepseek-llm-r1-distill-qwen-14b
     - False
     - 2.0.7
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B>`__ |external-link|
   * - deepseek-llm-r1-distill-qwen-32b
     - False
     - 2.0.7
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B>`__ |external-link|
   * - deepseek-llm-r1-distill-qwen-7b
     - False
     - 2.0.7
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B>`__ |external-link|
   * - huggingface-asr-whisper-base
     - False
     - 3.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/openai/whisper-base>`__ |external-link|
   * - huggingface-asr-whisper-large
     - False
     - 3.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/openai/whisper-large>`__ |external-link|
   * - huggingface-asr-whisper-large-v2
     - False
     - 3.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/openai/whisper-large-v2>`__ |external-link|
   * - huggingface-asr-whisper-large-v3
     - False
     - 1.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/openai/whisper-large-v3>`__ |external-link|
   * - huggingface-asr-whisper-large-v3-turbo
     - False
     - 1.1.10
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/openai/whisper-large-v3-turbo>`__ |external-link|
   * - huggingface-asr-whisper-medium
     - False
     - 3.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/openai/whisper-medium>`__ |external-link|
   * - huggingface-asr-whisper-small
     - False
     - 3.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/openai/whisper-small>`__ |external-link|
   * - huggingface-asr-whisper-tiny
     - False
     - 3.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/openai/whisper-tiny>`__ |external-link|
   * - huggingface-eqa-bert-base-cased
     - True
     - 3.0.6
     - 2.189.0
     - Question Answering
     - `HuggingFace <https://huggingface.co/google-bert/bert-base-cased>`__ |external-link|
   * - huggingface-eqa-bert-base-multilingual-cased
     - True
     - 3.0.6
     - 2.189.0
     - Question Answering
     - `HuggingFace <https://huggingface.co/google-bert/bert-base-multilingual-cased>`__ |external-link|
   * - huggingface-eqa-bert-base-multilingual-uncased
     - True
     - 3.0.6
     - 2.237.1
     - Question Answering
     - `HuggingFace <https://huggingface.co/google-bert/bert-base-multilingual-uncased>`__ |external-link|
   * - huggingface-eqa-bert-base-uncased
     - True
     - 3.0.6
     - 2.189.0
     - Question Answering
     - `HuggingFace <https://huggingface.co/google-bert/bert-base-uncased>`__ |external-link|
   * - huggingface-eqa-bert-large-cased
     - True
     - 3.0.6
     - 2.237.1
     - Question Answering
     - `HuggingFace <https://huggingface.co/google-bert/bert-large-cased>`__ |external-link|
   * - huggingface-eqa-bert-large-cased-whole-word-masking
     - True
     - 3.0.6
     - 2.189.0
     - Question Answering
     - `HuggingFace <https://huggingface.co/google-bert/bert-large-cased-whole-word-masking>`__ |external-link|
   * - huggingface-eqa-bert-large-uncased
     - True
     - 3.0.6
     - 2.189.0
     - Question Answering
     - `HuggingFace <https://huggingface.co/google-bert/bert-large-uncased>`__ |external-link|
   * - huggingface-eqa-bert-large-uncased-whole-word-masking
     - True
     - 3.0.6
     - 2.189.0
     - Question Answering
     - `HuggingFace <https://huggingface.co/google-bert/bert-large-uncased-whole-word-masking>`__ |external-link|
   * - huggingface-eqa-distilbert-base-cased
     - True
     - 3.0.6
     - 2.237.1
     - Question Answering
     - `HuggingFace <https://huggingface.co/distilbert/distilbert-base-cased>`__ |external-link|
   * - huggingface-eqa-distilbert-base-multilingual-cased
     - True
     - 3.0.6
     - 2.237.1
     - Question Answering
     - `HuggingFace <https://huggingface.co/distilbert/distilbert-base-multilingual-cased>`__ |external-link|
   * - huggingface-eqa-distilbert-base-uncased
     - True
     - 3.0.6
     - 2.237.1
     - Question Answering
     - `HuggingFace <https://huggingface.co/distilbert/distilbert-base-uncased>`__ |external-link|
   * - huggingface-eqa-distilroberta-base
     - True
     - 3.0.6
     - 2.237.1
     - Question Answering
     - `HuggingFace <https://huggingface.co/distilbert/distilroberta-base>`__ |external-link|
   * - huggingface-eqa-roberta-base
     - True
     - 3.0.6
     - 2.237.1
     - Question Answering
     - `HuggingFace <https://huggingface.co/FacebookAI/roberta-base>`__ |external-link|
   * - huggingface-eqa-roberta-base-openai-detector
     - True
     - 3.0.6
     - 2.237.1
     - Question Answering
     - `HuggingFace <https://huggingface.co/openai-community/roberta-base-openai-detector>`__ |external-link|
   * - huggingface-eqa-roberta-large
     - True
     - 3.0.6
     - 2.237.1
     - Question Answering
     - `HuggingFace <https://huggingface.co/FacebookAI/roberta-large>`__ |external-link|
   * - huggingface-fillmask-bert-base-uncased
     - True
     - 3.0.6
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/google-bert/bert-base-uncased>`__ |external-link|
   * - huggingface-llm-ahxt-litellama-460m-1t
     - False
     - 1.1.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/ahxt/LiteLlama-460M-1T>`__ |external-link|
   * - huggingface-llm-ai-forever-mgpt
     - False
     - 1.3.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/ai-forever/mGPT>`__ |external-link|
   * - huggingface-llm-alpindale-wizard-lm-2-8-22B
     - False
     - 1.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/alpindale/WizardLM-2-8x22B>`__ |external-link|
   * - huggingface-llm-amazon-falconlite
     - False
     - 2.1.12
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/amazon/FalconLite>`__ |external-link|
   * - huggingface-llm-amazon-falconlite2
     - False
     - 1.2.12
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/amazon/FalconLite2>`__ |external-link|
   * - huggingface-llm-amazon-mistrallite
     - False
     - 1.3.12
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/amazon/MistralLite>`__ |external-link|
   * - huggingface-llm-aya-101
     - False
     - 1.1.12
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/CohereForAI/aya-101>`__ |external-link|
   * - huggingface-llm-berkeley-nest-starling-lm-7b-alpha
     - False
     - 1.2.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/berkeley-nest/Starling-LM-7B-alpha>`__ |external-link|
   * - huggingface-llm-bilingual-rinna-4b-instruction-ppo-bf16
     - False
     - 2.2.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/rinna/bilingual-gpt-neox-4b-instruction-ppo>`__ |external-link|
   * - huggingface-llm-calm2-7b-chat-bf16
     - True
     - 1.4.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/cyberagent/calm2-7b-chat>`__ |external-link|
   * - huggingface-llm-calm3-22b-chat
     - False
     - 2.2.8
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/cyberagent/calm3-22b-chat>`__ |external-link|
   * - huggingface-llm-cognitive-dolphin-29-llama3-8b
     - False
     - 1.1.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/cognitivecomputations/dolphin-2.9-llama3-8b>`__ |external-link|
   * - huggingface-llm-cohereforai-c4ai-command-r-plus
     - False
     - 1.1.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/CohereForAI/c4ai-command-r-plus>`__ |external-link|
   * - huggingface-llm-cultrix-mistraltrix-v1
     - False
     - 1.3.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/CultriX/MistralTrix-v1>`__ |external-link|
   * - huggingface-llm-dbrx-base
     - False
     - 1.5.6
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/databricks/dbrx-base>`__ |external-link|
   * - huggingface-llm-dbrx-instruct
     - False
     - 1.4.6
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/databricks/dbrx-instruct>`__ |external-link|
   * - huggingface-llm-dolphin-2-2-1-mistral-7b
     - False
     - 1.3.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/cognitivecomputations/dolphin-2.2.1-mistral-7b>`__ |external-link|
   * - huggingface-llm-dolphin-2-5-mixtral-8x7b
     - False
     - 1.2.12
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/cognitivecomputations/dolphin-2.5-mixtral-8x7b>`__ |external-link|
   * - huggingface-llm-dolphin-2-7-mixtral-8x7b
     - False
     - 1.1.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/cognitivecomputations/dolphin-2.7-mixtral-8x7b>`__ |external-link|
   * - huggingface-llm-eleutherai-gpt-neo-1-3b
     - False
     - 1.3.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/EleutherAI/gpt-neo-1.3B>`__ |external-link|
   * - huggingface-llm-eleutherai-gpt-neo-2-7b
     - False
     - 1.2.12
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/EleutherAI/gpt-neo-2.7B>`__ |external-link|
   * - huggingface-llm-eleutherai-pythia-160m-deduped
     - False
     - 1.2.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/EleutherAI/pythia-160m-deduped>`__ |external-link|
   * - huggingface-llm-eleutherai-pythia-70m-deduped
     - False
     - 1.2.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/EleutherAI/pythia-70m-deduped>`__ |external-link|
   * - huggingface-llm-elyza-japanese-llama-2-13b-chat
     - True
     - 1.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/elyza/ELYZA-japanese-Llama-2-13b-instruct>`__ |external-link|
   * - huggingface-llm-elyza-japanese-llama-2-13b-fast-chat
     - True
     - 1.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/elyza/ELYZA-japanese-Llama-2-13b-fast-instruct>`__ |external-link|
   * - huggingface-llm-elyza-japanese-llama-2-7b-chat-bf16
     - True
     - 1.1.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/elyza/ELYZA-japanese-Llama-2-7b-instruct>`__ |external-link|
   * - huggingface-llm-elyza-japanese-llama-2-7b-fast-chat-bf16
     - True
     - 1.2.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/elyza/ELYZA-japanese-Llama-2-7b-fast-instruct>`__ |external-link|
   * - huggingface-llm-falcon-180b-bf16
     - False
     - 1.7.7
     - 2.188.0
     - Source
     - `HuggingFace <https://huggingface.co/tiiuae/falcon-180B>`__ |external-link|
   * - huggingface-llm-falcon-180b-chat-bf16
     - False
     - 1.6.7
     - 2.188.0
     - Source
     - `HuggingFace <https://huggingface.co/tiiuae/falcon-180B-chat>`__ |external-link|
   * - huggingface-llm-falcon-3-10B-Instruct
     - False
     - 2.1.5
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/tiiuae/Falcon3-10B-Instruct>`__ |external-link|
   * - huggingface-llm-falcon-3-10B-base
     - False
     - 3.0.6
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/tiiuae/Falcon3-10B-Base>`__ |external-link|
   * - huggingface-llm-falcon-3-1B-Instruct
     - False
     - 2.0.5
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/tiiuae/Falcon3-1B-Instruct>`__ |external-link|
   * - huggingface-llm-falcon-3-3B-Instruct
     - False
     - 2.0.5
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/tiiuae/Falcon3-3B-Instruct>`__ |external-link|
   * - huggingface-llm-falcon-3-3B-base
     - False
     - 3.0.6
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/tiiuae/Falcon3-3B-Base>`__ |external-link|
   * - huggingface-llm-falcon-3-7B-Instruct
     - False
     - 2.0.5
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/tiiuae/Falcon3-7B-Instruct>`__ |external-link|
   * - huggingface-llm-falcon-3-7B-base
     - False
     - 3.0.6
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/tiiuae/Falcon3-7B-Base>`__ |external-link|
   * - huggingface-llm-falcon-40b-bf16
     - True
     - 2.4.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/tiiuae/falcon-40b>`__ |external-link|
   * - huggingface-llm-falcon-40b-instruct-bf16
     - True
     - 2.3.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/tiiuae/falcon-40b-instruct>`__ |external-link|
   * - huggingface-llm-falcon-7b-bf16
     - True
     - 4.1.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/tiiuae/falcon-7b>`__ |external-link|
   * - huggingface-llm-falcon-7b-instruct-bf16
     - True
     - 4.6.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/tiiuae/falcon-7b-instruct>`__ |external-link|
   * - huggingface-llm-falcon2-11b
     - False
     - 1.2.8
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/tiiuae/falcon-11B>`__ |external-link|
   * - huggingface-llm-garage-baind-platypus2-7b
     - False
     - 1.1.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/garage-bAInd/Platypus2-7B>`__ |external-link|
   * - huggingface-llm-gemma-2-27b
     - False
     - 1.0.10
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/google/gemma-2-27b>`__ |external-link|
   * - huggingface-llm-gemma-2-27b-instruct
     - False
     - 1.0.10
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/google/gemma-2-27b>`__ |external-link|
   * - huggingface-llm-gemma-2-2b
     - False
     - 1.0.11
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/google/gemma-2-2b>`__ |external-link|
   * - huggingface-llm-gemma-2-2b-instruct
     - False
     - 1.0.11
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/google/gemma-2-2b>`__ |external-link|
   * - huggingface-llm-gemma-2-9b
     - False
     - 1.1.8
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/google/gemma-2-9b>`__ |external-link|
   * - huggingface-llm-gemma-2-9b-instruct
     - False
     - 1.1.8
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/google/gemma-2-9b>`__ |external-link|
   * - huggingface-llm-gemma-2b
     - True
     - 2.1.14
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/google/gemma-2b>`__ |external-link|
   * - huggingface-llm-gemma-2b-instruct
     - True
     - 1.4.14
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/google/gemma-2b-it>`__ |external-link|
   * - huggingface-llm-gemma-3-1b-instruct
     - False
     - 1.0.1
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/google/gemma-3-1b-it>`__ |external-link|
   * - huggingface-llm-gemma-7b
     - True
     - 1.3.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/google/gemma-7b>`__ |external-link|
   * - huggingface-llm-gemma-7b-instruct
     - True
     - 1.3.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/google/gemma-7b-it>`__ |external-link|
   * - huggingface-llm-gradientai-llama-3-8B-instruct-262k
     - False
     - 1.1.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/gradientai/Llama-3-8B-Instruct-262k>`__ |external-link|
   * - huggingface-llm-huggingfaceh4-mistral-7b-sft-alpha
     - False
     - 1.2.12
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/HuggingFaceH4/mistral-7b-sft-alpha>`__ |external-link|
   * - huggingface-llm-huggingfaceh4-mistral-7b-sft-beta
     - False
     - 1.4.12
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/HuggingFaceH4/mistral-7b-sft-beta>`__ |external-link|
   * - huggingface-llm-huggingfaceh4-starchat-alpha
     - False
     - 1.1.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/HuggingFaceH4/starchat-alpha>`__ |external-link|
   * - huggingface-llm-huggingfaceh4-starchat-beta
     - False
     - 1.2.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/HuggingFaceH4/starchat-beta>`__ |external-link|
   * - huggingface-llm-huggingfaceh4-zephyr-7b-alpha
     - False
     - 1.3.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha>`__ |external-link|
   * - huggingface-llm-huggingfaceh4-zephyr-7b-beta
     - False
     - 1.2.12
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/HuggingFaceH4/zephyr-7b-beta>`__ |external-link|
   * - huggingface-llm-huggingfaceh4-zephyr-orpo-141b-a35b-v01
     - False
     - 1.1.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1>`__ |external-link|
   * - huggingface-llm-llama-3-8b-instruct-gradient
     - False
     - 1.1.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k>`__ |external-link|
   * - huggingface-llm-llama3-8b-sealionv21-instruct
     - False
     - 1.3.6
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/aisingapore/llama3-8b-cpt-sea-lionv2.1-instruct>`__ |external-link|
   * - huggingface-llm-mistral-7b
     - True
     - 2.22.0
     - 2.225.0
     - Source
     - `HuggingFace <https://huggingface.co/mistralai/Mistral-7B-v0.1>`__ |external-link|
   * - huggingface-llm-mistral-7b-instruct
     - False
     - 3.19.0
     - 2.225.0
     - Source
     - `HuggingFace <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2>`__ |external-link|
   * - huggingface-llm-mistral-7b-instruct-v3
     - False
     - 1.1.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3>`__ |external-link|
   * - huggingface-llm-mistral-7b-openorca-gptq
     - False
     - 1.3.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GPTQ>`__ |external-link|
   * - huggingface-llm-mistral-7b-v3
     - False
     - 1.1.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/mistralai/Mistral-7B-v0.3>`__ |external-link|
   * - huggingface-llm-mistral-nemo-base-2407
     - False
     - 1.1.7
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/mistralai/Mistral-Nemo-Base-2407>`__ |external-link|
   * - huggingface-llm-mistral-nemo-instruct-2407
     - False
     - 1.1.7
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407>`__ |external-link|
   * - huggingface-llm-mistral-small-24B-Instruct-2501
     - False
     - 3.0.0
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501>`__ |external-link|
   * - huggingface-llm-mistralai-mixtral-8x22B-instruct-v0-1
     - False
     - 1.16.0
     - 2.225.0
     - Source
     - `HuggingFace <https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1>`__ |external-link|
   * - huggingface-llm-mixtral-8x22B
     - False
     - 1.2.7
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/mistralai/Mixtral-8x22B-v0.1>`__ |external-link|
   * - huggingface-llm-mixtral-8x7b
     - True
     - 1.23.0
     - 2.225.0
     - Source
     - `HuggingFace <https://huggingface.co/mistralai/Mixtral-8x7B-v0.1>`__ |external-link|
   * - huggingface-llm-mixtral-8x7b-instruct
     - True
     - 1.23.0
     - 2.225.0
     - Source
     - `HuggingFace <https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1>`__ |external-link|
   * - huggingface-llm-mixtral-8x7b-instruct-gptq
     - False
     - 1.3.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ>`__ |external-link|
   * - huggingface-llm-nexaaidev-octopus-v2
     - False
     - 1.1.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/NexaAIDev/Octopus-v2>`__ |external-link|
   * - huggingface-llm-nexusflow-starling-lm-7b-beta
     - False
     - 1.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/Nexusflow/Starling-LM-7B-beta>`__ |external-link|
   * - huggingface-llm-nousresearch-hermes-2-pro-llama-3-8B
     - False
     - 1.1.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B>`__ |external-link|
   * - huggingface-llm-nousresearch-nous-hermes-2-solar-10-7b
     - False
     - 1.2.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/NousResearch/Nous-Hermes-2-SOLAR-10.7B>`__ |external-link|
   * - huggingface-llm-nousresearch-nous-hermes-llama-2-7b
     - False
     - 1.1.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/NousResearch/Nous-Hermes-llama-2-7b>`__ |external-link|
   * - huggingface-llm-nousresearch-nous-hermes-llama2-13b
     - False
     - 1.1.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/NousResearch/Nous-Hermes-Llama2-13b>`__ |external-link|
   * - huggingface-llm-nousresearch-yarn-mistral-7b-128k
     - False
     - 1.2.12
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/NousResearch/Yarn-Mistral-7b-128k>`__ |external-link|
   * - huggingface-llm-nvidia-llama3-chatqa-1-5-70B
     - False
     - 1.1.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/nvidia/Llama3-ChatQA-1.5-70B>`__ |external-link|
   * - huggingface-llm-nvidia-llama3-chatqa-1-5-8B
     - False
     - 1.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/nvidia/Llama3-ChatQA-1.5-8B>`__ |external-link|
   * - huggingface-llm-openlm-research-open-llama-7b-v2
     - False
     - 1.2.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/openlm-research/open_llama_7b_v2>`__ |external-link|
   * - huggingface-llm-phi-2
     - False
     - 1.2.7
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/microsoft/phi-2>`__ |external-link|
   * - huggingface-llm-phi-3-5-mini-instruct
     - False
     - 1.1.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/microsoft/Phi-3.5-mini-instruct>`__ |external-link|
   * - huggingface-llm-phi-3-mini-128k-instruct
     - False
     - 1.3.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/microsoft/Phi-3-mini-128k-instruct>`__ |external-link|
   * - huggingface-llm-phi-3-mini-4k-instruct
     - False
     - 1.3.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/microsoft/Phi-3-mini-4k-instruct>`__ |external-link|
   * - huggingface-llm-qwen2-0-5b
     - False
     - 1.2.5
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/Qwen/Qwen2-0.5B>`__ |external-link|
   * - huggingface-llm-qwen2-0-5b-instruct
     - False
     - 1.2.5
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/Qwen/Qwen2-0.5B-Instruct>`__ |external-link|
   * - huggingface-llm-qwen2-1-5b
     - False
     - 1.2.5
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/Qwen/Qwen2-1.5B>`__ |external-link|
   * - huggingface-llm-qwen2-1-5b-instruct
     - False
     - 1.2.5
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/Qwen/Qwen2-1.5B-Instruct>`__ |external-link|
   * - huggingface-llm-qwen2-5-14b-instruct
     - False
     - 1.0.6
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/Qwen/Qwen2.5-14B-Instruct>`__ |external-link|
   * - huggingface-llm-qwen2-5-32b-instruct
     - False
     - 1.0.6
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/Qwen/Qwen2.5-32B-Instruct>`__ |external-link|
   * - huggingface-llm-qwen2-5-72b-instruct
     - False
     - 1.0.6
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/Qwen/Qwen2.5-72B-Instruct>`__ |external-link|
   * - huggingface-llm-qwen2-5-7b-instruct
     - False
     - 1.0.6
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>`__ |external-link|
   * - huggingface-llm-qwen2-5-coder-32b-instruct
     - False
     - 1.0.6
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct>`__ |external-link|
   * - huggingface-llm-qwen2-5-coder-7b-instruct
     - False
     - 1.0.7
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct>`__ |external-link|
   * - huggingface-llm-qwen2-7b
     - False
     - 1.2.5
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/Qwen/Qwen2-7B>`__ |external-link|
   * - huggingface-llm-qwen2-7b-instruct
     - False
     - 1.2.5
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/Qwen/Qwen2-7B-Instruct>`__ |external-link|
   * - huggingface-llm-qwq-32b
     - False
     - 1.0.6
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/Qwen/QwQ-32B>`__ |external-link|
   * - huggingface-llm-rinna-3-6b-instruction-ppo-bf16
     - False
     - 2.1.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-ppo>`__ |external-link|
   * - huggingface-llm-sealion-3b
     - False
     - 1.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/aisingapore/sealion3b>`__ |external-link|
   * - huggingface-llm-sealion-7b
     - False
     - 1.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/aisingapore/sealion7b>`__ |external-link|
   * - huggingface-llm-sealion-7b-instruct
     - False
     - 1.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/aisingapore/sea-lion-7b-instruct>`__ |external-link|
   * - huggingface-llm-shenzhi-wang-llama3-8B-chinese-chat
     - False
     - 1.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/shenzhi-wang/Llama3-8B-Chinese-Chat>`__ |external-link|
   * - huggingface-llm-snowflake-arctic-instruct-vllm
     - False
     - 1.3.6
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/Snowflake/snowflake-arctic-instruct-vllm>`__ |external-link|
   * - huggingface-llm-starcoder
     - False
     - 1.2.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/bigcode/starcoder>`__ |external-link|
   * - huggingface-llm-starcoderbase
     - False
     - 1.2.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/bigcode/starcoderbase>`__ |external-link|
   * - huggingface-llm-teknium-openhermes-2-mistral-7b
     - False
     - 1.3.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/teknium/OpenHermes-2-Mistral-7B>`__ |external-link|
   * - huggingface-llm-thebloke-mistral-7b-openorca-awq
     - False
     - 1.3.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-AWQ>`__ |external-link|
   * - huggingface-llm-tiiuae-falcon-rw-1b
     - False
     - 1.3.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/tiiuae/falcon-rw-1b>`__ |external-link|
   * - huggingface-llm-tinyllama-1-1b-intermediate-step-1431k-3
     - False
     - 1.1.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T>`__ |external-link|
   * - huggingface-llm-tinyllama-tinyllama-1-1b-chat-v0-6
     - False
     - 1.1.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v0.6>`__ |external-link|
   * - huggingface-llm-tinyllama-tinyllama-1-1b-chat-v1-0
     - False
     - 1.1.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0>`__ |external-link|
   * - huggingface-llm-writer-palmyra-small
     - False
     - 1.2.12
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/Writer/palmyra-small>`__ |external-link|
   * - huggingface-llm-yi-1-5-34b
     - False
     - 1.4.5
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/01-ai/Yi-1.5-34B>`__ |external-link|
   * - huggingface-llm-yi-1-5-34b-chat
     - False
     - 1.5.5
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/01-ai/Yi-1.5-34B-Chat>`__ |external-link|
   * - huggingface-llm-yi-1-5-6b
     - False
     - 1.4.5
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/01-ai/Yi-1.5-6B>`__ |external-link|
   * - huggingface-llm-yi-1-5-6b-chat
     - False
     - 1.4.5
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/01-ai/Yi-1.5-6B-Chat>`__ |external-link|
   * - huggingface-llm-yi-1-5-9b
     - False
     - 1.4.5
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/01-ai/Yi-1.5-9B>`__ |external-link|
   * - huggingface-llm-yi-1-5-9b-chat
     - False
     - 1.4.5
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/01-ai/Yi-1.5-9B-Chat>`__ |external-link|
   * - huggingface-llm-zephyr-7b-gemma
     - False
     - 1.2.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/HuggingFaceH4/zephyr-7b-gemma-v0.1>`__ |external-link|
   * - huggingface-llmneuron-mistral-7b
     - False
     - 1.0.11
     - 2.198.0
     - Source
     - `HuggingFace <https://huggingface.co/mistralai/Mistral-7B-v0.1>`__ |external-link|
   * - huggingface-llmneuron-mistral-7b-instruct
     - False
     - 1.0.11
     - 2.198.0
     - Source
     - `HuggingFace <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1>`__ |external-link|
   * - huggingface-ner-distilbert-base-cased-finetuned-conll03-eng
     - False
     - 1.1.7
     - 2.189.0
     - Named Entity Recognition
     - `HuggingFace <https://huggingface.co/elastic/distilbert-base-cased-finetuned-conll03-english>`__ |external-link|
   * - huggingface-ner-distilbert-base-cased-finetuned-conll03-english
     - False
     - 2.0.14
     - 2.189.0
     - Named Entity Recognition
     - `HuggingFace <https://huggingface.co/elastic/distilbert-base-cased-finetuned-conll03-english>`__ |external-link|
   * - huggingface-ner-distilbert-base-uncased-finetuned-conll03-eng
     - False
     - 1.1.7
     - 2.189.0
     - Named Entity Recognition
     - `HuggingFace <https://huggingface.co/elastic/distilbert-base-uncased-finetuned-conll03-english>`__ |external-link|
   * - huggingface-ner-distilbert-base-uncased-finetuned-conll03-english
     - False
     - 2.0.14
     - 2.189.0
     - Named Entity Recognition
     - `HuggingFace <https://huggingface.co/elastic/distilbert-base-uncased-finetuned-conll03-english>`__ |external-link|
   * - huggingface-reasoning-qwen3-06b
     - False
     - 1.0.1
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/Qwen/Qwen3-0.6B>`__ |external-link|
   * - huggingface-reasoning-qwen3-32b
     - False
     - 1.0.1
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/Qwen/Qwen3-32B>`__ |external-link|
   * - huggingface-reasoning-qwen3-4b
     - False
     - 1.0.1
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/Qwen/Qwen3-4B>`__ |external-link|
   * - huggingface-reasoning-qwen3-8b
     - False
     - 1.0.1
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/Qwen/Qwen3-8B>`__ |external-link|
   * - huggingface-sentencesimilarity-all-MiniLM-L6-v2
     - True
     - 2.1.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>`__ |external-link|
   * - huggingface-sentencesimilarity-bge-base-en
     - True
     - 2.1.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/BAAI/bge-base-en>`__ |external-link|
   * - huggingface-sentencesimilarity-bge-base-en-v1-5
     - True
     - 1.1.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/BAAI/bge-base-en-v1.5>`__ |external-link|
   * - huggingface-sentencesimilarity-bge-large-en
     - True
     - 2.1.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/BAAI/bge-large-en>`__ |external-link|
   * - huggingface-sentencesimilarity-bge-large-en-v1-5
     - True
     - 1.1.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/BAAI/bge-large-en-v1.5>`__ |external-link|
   * - huggingface-sentencesimilarity-bge-large-zh-v1-5
     - True
     - 1.1.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/BAAI/bge-large-zh-v1.5>`__ |external-link|
   * - huggingface-sentencesimilarity-bge-m3
     - True
     - 1.1.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/BAAI/bge-m3>`__ |external-link|
   * - huggingface-sentencesimilarity-bge-small-en
     - True
     - 2.1.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/BAAI/bge-small-en>`__ |external-link|
   * - huggingface-sentencesimilarity-bge-small-en-v1-5
     - True
     - 1.1.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/BAAI/bge-small-en-v1.5>`__ |external-link|
   * - huggingface-sentencesimilarity-e5-base
     - True
     - 2.1.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/intfloat/e5-base>`__ |external-link|
   * - huggingface-sentencesimilarity-e5-base-v2
     - True
     - 2.1.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/intfloat/e5-base-v2>`__ |external-link|
   * - huggingface-sentencesimilarity-e5-large
     - True
     - 2.1.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/intfloat/e5-large>`__ |external-link|
   * - huggingface-sentencesimilarity-e5-large-v2
     - True
     - 2.1.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/intfloat/e5-large-v2>`__ |external-link|
   * - huggingface-sentencesimilarity-e5-small-v2
     - True
     - 2.1.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/intfloat/e5-small-v2>`__ |external-link|
   * - huggingface-sentencesimilarity-gte-base
     - True
     - 2.1.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/thenlper/gte-base>`__ |external-link|
   * - huggingface-sentencesimilarity-gte-large
     - True
     - 3.1.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/thenlper/gte-large>`__ |external-link|
   * - huggingface-sentencesimilarity-gte-small
     - True
     - 2.1.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/thenlper/gte-small>`__ |external-link|
   * - huggingface-sentencesimilarity-multilingual-e5-base
     - True
     - 2.1.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/intfloat/multilingual-e5-base>`__ |external-link|
   * - huggingface-sentencesimilarity-multilingual-e5-large
     - True
     - 2.1.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/intfloat/multilingual-e5-large>`__ |external-link|
   * - huggingface-spc-bert-base-cased
     - True
     - 3.0.6
     - 2.189.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/google-bert/bert-base-cased>`__ |external-link|
   * - huggingface-spc-bert-base-multilingual-cased
     - True
     - 3.0.6
     - 2.189.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/google-bert/bert-base-multilingual-cased>`__ |external-link|
   * - huggingface-spc-bert-base-multilingual-uncased
     - True
     - 3.0.6
     - 2.189.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/google-bert/bert-base-multilingual-uncased>`__ |external-link|
   * - huggingface-spc-bert-base-uncased
     - True
     - 3.0.6
     - 2.189.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/google-bert/bert-base-uncased>`__ |external-link|
   * - huggingface-spc-bert-large-cased
     - True
     - 3.0.6
     - 2.189.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/google-bert/bert-large-cased>`__ |external-link|
   * - huggingface-spc-bert-large-cased-whole-word-masking
     - True
     - 3.0.6
     - 2.189.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/google-bert/bert-large-cased-whole-word-masking>`__ |external-link|
   * - huggingface-spc-bert-large-uncased
     - True
     - 3.0.6
     - 2.189.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/google-bert/bert-large-uncased>`__ |external-link|
   * - huggingface-spc-bert-large-uncased-whole-word-masking
     - True
     - 3.0.6
     - 2.189.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/google-bert/bert-large-uncased-whole-word-masking>`__ |external-link|
   * - huggingface-spc-distilbert-base-cased
     - True
     - 3.0.6
     - 2.189.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/distilbert/distilbert-base-cased>`__ |external-link|
   * - huggingface-spc-distilbert-base-multilingual-cased
     - True
     - 3.0.6
     - 2.189.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/distilbert/distilbert-base-multilingual-cased>`__ |external-link|
   * - huggingface-spc-distilbert-base-uncased
     - True
     - 3.0.6
     - 2.189.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/distilbert/distilbert-base-uncased>`__ |external-link|
   * - huggingface-spc-distilroberta-base
     - True
     - 3.0.6
     - 2.189.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/distilbert/distilroberta-base>`__ |external-link|
   * - huggingface-spc-roberta-base
     - True
     - 3.0.6
     - 2.189.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/FacebookAI/roberta-base>`__ |external-link|
   * - huggingface-spc-roberta-base-openai-detector
     - True
     - 3.0.6
     - 2.189.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/openai-community/roberta-base-openai-detector>`__ |external-link|
   * - huggingface-spc-roberta-large
     - True
     - 3.0.6
     - 2.189.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/FacebookAI/roberta-large>`__ |external-link|
   * - huggingface-spc-roberta-large-openai-detector
     - True
     - 3.0.6
     - 2.189.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/openai-community/roberta-large-openai-detector>`__ |external-link|
   * - huggingface-spc-xlm-clm-ende-1024
     - True
     - 3.0.6
     - 2.189.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/FacebookAI/xlm-clm-ende-1024>`__ |external-link|
   * - huggingface-spc-xlm-mlm-ende-1024
     - True
     - 3.0.6
     - 2.189.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/FacebookAI/xlm-mlm-ende-1024>`__ |external-link|
   * - huggingface-spc-xlm-mlm-enro-1024
     - True
     - 3.0.6
     - 2.189.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/FacebookAI/xlm-mlm-enro-1024>`__ |external-link|
   * - huggingface-spc-xlm-mlm-tlm-xnli15-1024
     - True
     - 3.0.6
     - 2.189.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/FacebookAI/xlm-mlm-tlm-xnli15-1024>`__ |external-link|
   * - huggingface-spc-xlm-mlm-xnli15-1024
     - True
     - 3.0.6
     - 2.189.0
     - Sentence Pair Classification
     - `HuggingFace <https://huggingface.co/FacebookAI/xlm-mlm-xnli15-1024>`__ |external-link|
   * - huggingface-summarization-bart-large-cnn-samsum
     - False
     - 2.2.7
     - 2.189.0
     - Text Summarization
     - `HuggingFace <https://huggingface.co/philschmid/bart-large-cnn-samsum>`__ |external-link|
   * - huggingface-summarization-bert-small2bert-cnn-dailymail-summ
     - False
     - 1.1.7
     - 2.189.0
     - Text Summarization
     - `HuggingFace <https://huggingface.co/mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization>`__ |external-link|
   * - huggingface-summarization-bert-small2bert-small-finetuned-cnn-daily-mail-summarization
     - False
     - 2.1.7
     - 2.189.0
     - Text Summarization
     - `HuggingFace <https://huggingface.co/mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization>`__ |external-link|
   * - huggingface-summarization-bigbird-pegasus-large-arxiv
     - False
     - 2.1.7
     - 2.189.0
     - Text Summarization
     - `HuggingFace <https://huggingface.co/google/bigbird-pegasus-large-arxiv>`__ |external-link|
   * - huggingface-summarization-bigbird-pegasus-large-pubmed
     - False
     - 2.1.7
     - 2.189.0
     - Text Summarization
     - `HuggingFace <https://huggingface.co/google/bigbird-pegasus-large-pubmed>`__ |external-link|
   * - huggingface-summarization-distilbart-cnn-12-6
     - False
     - 2.2.7
     - 2.189.0
     - Text Summarization
     - `HuggingFace <https://huggingface.co/sshleifer/distilbart-cnn-12-6>`__ |external-link|
   * - huggingface-summarization-distilbart-cnn-6-6
     - False
     - 2.1.13
     - 2.189.0
     - Text Summarization
     - `HuggingFace <https://huggingface.co/sshleifer/distilbart-cnn-6-6>`__ |external-link|
   * - huggingface-summarization-distilbart-xsum-1-1
     - False
     - 2.2.7
     - 2.189.0
     - Text Summarization
     - `HuggingFace <https://huggingface.co/sshleifer/distilbart-xsum-1-1>`__ |external-link|
   * - huggingface-summarization-distilbart-xsum-12-3
     - False
     - 2.1.13
     - 2.189.0
     - Text Summarization
     - `HuggingFace <https://huggingface.co/sshleifer/distilbart-xsum-12-3>`__ |external-link|
   * - huggingface-tc-bert-base-cased
     - True
     - 3.0.6
     - 2.189.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/google-bert/bert-base-cased>`__ |external-link|
   * - huggingface-tc-bert-base-multilingual-cased
     - True
     - 3.0.6
     - 2.189.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/google-bert/bert-base-multilingual-cased>`__ |external-link|
   * - huggingface-tc-bert-base-multilingual-uncased
     - True
     - 3.0.6
     - 2.189.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/google-bert/bert-base-multilingual-uncased>`__ |external-link|
   * - huggingface-tc-bert-base-uncased
     - True
     - 3.0.6
     - 2.189.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/google-bert/bert-base-uncased>`__ |external-link|
   * - huggingface-tc-bert-large-cased
     - True
     - 3.0.6
     - 2.189.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/google-bert/bert-large-cased>`__ |external-link|
   * - huggingface-tc-bert-large-cased-whole-word-masking
     - True
     - 3.0.6
     - 2.189.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/google-bert/bert-large-cased-whole-word-masking>`__ |external-link|
   * - huggingface-tc-bert-large-uncased
     - True
     - 3.0.6
     - 2.189.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/google-bert/bert-large-uncased>`__ |external-link|
   * - huggingface-tc-bert-large-uncased-whole-word-masking
     - True
     - 3.0.6
     - 2.189.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/google-bert/bert-large-uncased-whole-word-masking>`__ |external-link|
   * - huggingface-tc-distilbert-base-cased
     - True
     - 3.0.6
     - 2.189.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/distilbert/distilbert-base-cased>`__ |external-link|
   * - huggingface-tc-distilbert-base-multilingual-cased
     - True
     - 3.0.6
     - 2.189.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/distilbert/distilbert-base-multilingual-cased>`__ |external-link|
   * - huggingface-tc-distilbert-base-uncased
     - True
     - 3.0.6
     - 2.189.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/distilbert/distilbert-base-uncased>`__ |external-link|
   * - huggingface-tc-distilroberta-base
     - True
     - 3.0.6
     - 2.189.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/distilbert/distilroberta-base>`__ |external-link|
   * - huggingface-tc-models
     - True
     - 3.0.6
     - 2.189.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/albert/albert-base-v2>`__ |external-link|
   * - huggingface-tc-roberta-base
     - True
     - 3.0.6
     - 2.189.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/FacebookAI/roberta-base>`__ |external-link|
   * - huggingface-tc-roberta-base-openai-detector
     - True
     - 3.0.6
     - 2.189.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/openai-community/roberta-base-openai-detector>`__ |external-link|
   * - huggingface-tc-roberta-large
     - True
     - 3.0.6
     - 2.189.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/FacebookAI/roberta-large>`__ |external-link|
   * - huggingface-tc-roberta-large-openai-detector
     - True
     - 3.0.6
     - 2.189.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/openai-community/roberta-large-openai-detector>`__ |external-link|
   * - huggingface-tc-xlm-clm-ende-1024
     - True
     - 3.0.6
     - 2.189.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/FacebookAI/xlm-clm-ende-1024>`__ |external-link|
   * - huggingface-tc-xlm-mlm-ende-1024
     - True
     - 3.0.6
     - 2.189.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/FacebookAI/xlm-mlm-ende-1024>`__ |external-link|
   * - huggingface-tc-xlm-mlm-enro-1024
     - True
     - 3.0.6
     - 2.189.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/FacebookAI/xlm-mlm-enro-1024>`__ |external-link|
   * - huggingface-tc-xlm-mlm-tlm-xnli15-1024
     - True
     - 3.0.6
     - 2.189.0
     - Text Classification
     - `HuggingFace <https://huggingface.co/FacebookAI/xlm-mlm-tlm-xnli15-1024>`__ |external-link|
   * - huggingface-text2text-bart4csc-base-chinese
     - False
     - 1.3.5
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/shibing624/bart4csc-base-chinese>`__ |external-link|
   * - huggingface-text2text-bigscience-t0pp
     - False
     - 2.2.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/bigscience/T0pp>`__ |external-link|
   * - huggingface-text2text-bigscience-t0pp-bnb-int8
     - False
     - 1.2.5
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/bigscience/T0pp>`__ |external-link|
   * - huggingface-text2text-bigscience-t0pp-fp16
     - False
     - 1.2.5
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/bigscience/T0pp>`__ |external-link|
   * - huggingface-text2text-flan-t5-base
     - True
     - 2.3.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/google/flan-t5-base>`__ |external-link|
   * - huggingface-text2text-flan-t5-base-samsum
     - False
     - 2.3.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/philschmid/flan-t5-base-samsum>`__ |external-link|
   * - huggingface-text2text-flan-t5-large
     - True
     - 2.3.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/google/flan-t5-large>`__ |external-link|
   * - huggingface-text2text-flan-t5-small
     - True
     - 2.3.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/google/flan-t5-small>`__ |external-link|
   * - huggingface-text2text-flan-t5-xl
     - True
     - 2.2.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/google/flan-t5-xl>`__ |external-link|
   * - huggingface-text2text-flan-t5-xxl
     - True
     - 2.2.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/google/flan-t5-xxl>`__ |external-link|
   * - huggingface-text2text-flan-t5-xxl-bnb-int8
     - False
     - 1.3.5
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/google/flan-t5-xxl>`__ |external-link|
   * - huggingface-text2text-flan-t5-xxl-fp16
     - True
     - 1.2.5
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/google/flan-t5-xxl>`__ |external-link|
   * - huggingface-text2text-flan-ul2-bf16
     - False
     - 2.2.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/google/flan-ul2>`__ |external-link|
   * - huggingface-text2text-pegasus-paraphrase
     - False
     - 1.3.5
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/shibing624/bart4csc-base-chinese>`__ |external-link|
   * - huggingface-text2text-qcpg-sentences
     - False
     - 2.2.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/ibm/qcpg-sentences>`__ |external-link|
   * - huggingface-text2text-t5-one-line-summary
     - False
     - 2.2.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/snrspeaks/t5-one-line-summary>`__ |external-link|
   * - huggingface-textembedding-all-MiniLM-L6-v2
     - False
     - 2.0.11
     - 2.225.0
     - Source
     - `HuggingFace <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>`__ |external-link|
   * - huggingface-textembedding-bge-base-en-v1-5
     - False
     - 1.0.11
     - 2.225.0
     - Source
     - `HuggingFace <https://huggingface.co/BAAI/bge-base-en-v1.5>`__ |external-link|
   * - huggingface-textembedding-bloom-7b1
     - False
     - 1.1.17
     - 2.225.0
     - Source
     - `HuggingFace <https://huggingface.co/bigscience/bloom-7b1>`__ |external-link|
   * - huggingface-textembedding-bloom-7b1-fp16
     - False
     - 1.1.17
     - 2.225.0
     - Source
     - `HuggingFace <https://huggingface.co/bigscience/bloom-7b1>`__ |external-link|
   * - huggingface-textembedding-gpt-j-6b
     - False
     - 1.1.17
     - 2.225.0
     - Source
     - `HuggingFace <https://huggingface.co/EleutherAI/gpt-j-6B>`__ |external-link|
   * - huggingface-textembedding-gpt-j-6b-fp16
     - False
     - 1.1.17
     - 2.225.0
     - Source
     - `HuggingFace <https://huggingface.co/EleutherAI/gpt-j-6B>`__ |external-link|
   * - huggingface-textembedding-gte-qwen2-7b-instruct
     - False
     - 1.0.11
     - 2.225.0
     - Source
     - `HuggingFace <https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct>`__ |external-link|
   * - huggingface-textembedding-sfr-embedding-2-r
     - False
     - 1.0.11
     - 2.225.0
     - Source
     - `HuggingFace <https://huggingface.co/Salesforce/SFR-Embedding-2_R>`__ |external-link|
   * - huggingface-textembedding-sfr-embedding-mistral
     - False
     - 1.0.11
     - 2.225.0
     - Source
     - `HuggingFace <https://huggingface.co/Salesforce/SFR-Embedding-Mistral>`__ |external-link|
   * - huggingface-textgeneration-bloom-1b1
     - False
     - 2.3.7
     - 2.189.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/bigscience/bloom-1b1>`__ |external-link|
   * - huggingface-textgeneration-bloom-1b7
     - False
     - 2.3.7
     - 2.189.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/bigscience/bloom-1b7>`__ |external-link|
   * - huggingface-textgeneration-bloom-560m
     - False
     - 2.3.7
     - 2.189.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/bigscience/bloom-560m>`__ |external-link|
   * - huggingface-textgeneration-bloomz-1b1
     - False
     - 2.3.7
     - 2.189.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/bigscience/bloomz-1b1>`__ |external-link|
   * - huggingface-textgeneration-bloomz-1b7
     - False
     - 2.3.7
     - 2.189.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/bigscience/bloomz-1b7>`__ |external-link|
   * - huggingface-textgeneration-bloomz-560m
     - False
     - 2.2.7
     - 2.189.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/bigscience/bloomz-560m>`__ |external-link|
   * - huggingface-textgeneration-distilgpt2
     - False
     - 3.0.6
     - 2.189.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/distilbert/distilgpt2>`__ |external-link|
   * - huggingface-textgeneration-dolly-v2-12b-bf16
     - False
     - 2.2.12
     - 2.189.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/databricks/dolly-v2-12b>`__ |external-link|
   * - huggingface-textgeneration-dolly-v2-3b-bf16
     - False
     - 2.3.7
     - 2.189.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/databricks/dolly-v2-3b>`__ |external-link|
   * - huggingface-textgeneration-dolly-v2-7b-bf16
     - False
     - 2.2.12
     - 2.189.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/databricks/dolly-v2-7b>`__ |external-link|
   * - huggingface-textgeneration-falcon-40b-bf16
     - False
     - 1.1.5
     - 2.189.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/tiiuae/falcon-40b>`__ |external-link|
   * - huggingface-textgeneration-falcon-40b-instruct-bf16
     - False
     - 1.1.5
     - 2.189.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/tiiuae/falcon-40b-instruct>`__ |external-link|
   * - huggingface-textgeneration-falcon-7b-bf16
     - False
     - 1.1.5
     - 2.189.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/tiiuae/falcon-7b>`__ |external-link|
   * - huggingface-textgeneration-falcon-7b-instruct-bf16
     - False
     - 1.1.5
     - 2.189.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/tiiuae/falcon-7b-instruct>`__ |external-link|
   * - huggingface-textgeneration-gpt2
     - False
     - 3.0.6
     - 2.189.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/openai-community/gpt2>`__ |external-link|
   * - huggingface-textgeneration-gpt2-large
     - False
     - 1.0.6
     - 2.189.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/openai-community/gpt2-large>`__ |external-link|
   * - huggingface-textgeneration-gpt2-medium
     - False
     - 1.0.6
     - 2.189.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/openai-community/gpt2-medium>`__ |external-link|
   * - huggingface-textgeneration-models
     - False
     - 1.4.5
     - 2.189.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads>`__ |external-link|
   * - huggingface-textgeneration-open-llama
     - False
     - 3.2.7
     - 2.189.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/openlm-research/open_llama_7b>`__ |external-link|
   * - huggingface-textgeneration-openai-gpt
     - False
     - 1.0.6
     - 2.189.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/openai-community/openai-gpt>`__ |external-link|
   * - huggingface-textgeneration1-bloom-176b-int8
     - False
     - 1.1.4
     - 2.144.0
     - Source
     - `HuggingFace <https://huggingface.co/microsoft/bloom-deepspeed-inference-int8>`__ |external-link|
   * - huggingface-textgeneration1-bloom-3b
     - True
     - 3.3.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/bigscience/bloom-3b>`__ |external-link|
   * - huggingface-textgeneration1-bloom-3b-fp16
     - True
     - 2.1.4
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/bigscience/bloom-3b>`__ |external-link|
   * - huggingface-textgeneration1-bloom-7b1
     - True
     - 3.2.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/bigscience/bloom-7b1>`__ |external-link|
   * - huggingface-textgeneration1-bloom-7b1-fp16
     - True
     - 3.0.4
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/bigscience/bloom-7b1>`__ |external-link|
   * - huggingface-textgeneration1-bloomz-176b-fp16
     - False
     - 2.2.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/bigscience/bloomz>`__ |external-link|
   * - huggingface-textgeneration1-bloomz-3b-fp16
     - True
     - 3.2.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/bigscience/bloomz-3b>`__ |external-link|
   * - huggingface-textgeneration1-bloomz-7b1-fp16
     - True
     - 3.3.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/bigscience/bloomz-7b1>`__ |external-link|
   * - huggingface-textgeneration1-gpt-2-xl
     - True
     - 4.0.6
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/openai-community/gpt2-xl>`__ |external-link|
   * - huggingface-textgeneration1-gpt-2-xl-fp16
     - True
     - 3.0.4
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/openai-community/gpt2-xl>`__ |external-link|
   * - huggingface-textgeneration1-gpt-j-6b
     - True
     - 3.2.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/EleutherAI/gpt-j-6B>`__ |external-link|
   * - huggingface-textgeneration1-gpt-j-6b-fp16
     - True
     - 2.1.4
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/EleutherAI/gpt-j-6B>`__ |external-link|
   * - huggingface-textgeneration1-gpt-neo-1-3b
     - True
     - 3.2.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/EleutherAI/gpt-neo-1.3B>`__ |external-link|
   * - huggingface-textgeneration1-gpt-neo-1-3b-fp16
     - True
     - 3.0.4
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/EleutherAI/gpt-neo-1.3B>`__ |external-link|
   * - huggingface-textgeneration1-gpt-neo-125m
     - True
     - 3.2.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/EleutherAI/gpt-neo-125M>`__ |external-link|
   * - huggingface-textgeneration1-gpt-neo-125m-fp16
     - True
     - 2.1.4
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/EleutherAI/gpt-neo-125M>`__ |external-link|
   * - huggingface-textgeneration1-gpt-neo-2-7b
     - True
     - 3.2.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/EleutherAI/gpt-neo-2.7B>`__ |external-link|
   * - huggingface-textgeneration1-gpt-neo-2-7b-fp16
     - True
     - 2.1.4
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/EleutherAI/gpt-neo-2.7B>`__ |external-link|
   * - huggingface-textgeneration1-lightgpt
     - True
     - 3.1.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/amazon/LightGPT>`__ |external-link|
   * - huggingface-textgeneration1-mpt-7b-bf16
     - False
     - 3.2.12
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/mosaicml/mpt-7b>`__ |external-link|
   * - huggingface-textgeneration1-mpt-7b-instruct-bf16
     - False
     - 3.2.12
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/mosaicml/mpt-7b-instruct>`__ |external-link|
   * - huggingface-textgeneration1-mpt-7b-storywriter-bf16
     - False
     - 3.2.12
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/mosaicml/mpt-7b-storywriter>`__ |external-link|
   * - huggingface-textgeneration1-redpajama-incite-base-3B-v1-fp16
     - True
     - 3.1.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-3B-v1>`__ |external-link|
   * - huggingface-textgeneration1-redpajama-incite-base-7B-v1-fp16
     - True
     - 3.1.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-7B-v0.1>`__ |external-link|
   * - huggingface-textgeneration1-redpajama-incite-chat-3B-v1-fp16
     - True
     - 3.1.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1>`__ |external-link|
   * - huggingface-textgeneration1-redpajama-incite-chat-7B-v1-fp16
     - True
     - 3.1.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-7B-v0.1>`__ |external-link|
   * - huggingface-textgeneration1-redpajama-incite-instruct-3B-v1-fp16
     - True
     - 3.1.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-3B-v1>`__ |external-link|
   * - huggingface-textgeneration1-redpajama-incite-instruct-3Bv1fp16
     - True
     - 1.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-3B-v1>`__ |external-link|
   * - huggingface-textgeneration1-redpajama-incite-instruct-7B-v1-fp16
     - True
     - 3.1.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-7B-v0.1>`__ |external-link|
   * - huggingface-textgeneration1-redpajama-incite-instruct-7B1fp16
     - True
     - 1.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-7B-v0.1>`__ |external-link|
   * - huggingface-textgeneration2-gpt-neox-20b-fp16
     - False
     - 3.3.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/EleutherAI/gpt-neox-20b>`__ |external-link|
   * - huggingface-textgeneration2-gpt-neoxt-chat-base-20b-fp16
     - False
     - 3.2.7
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/togethercomputer/GPT-NeoXT-Chat-Base-20B>`__ |external-link|
   * - huggingface-translation-opus-mt-en-es
     - False
     - 2.1.7
     - 2.189.0
     - Machine Translation
     - `HuggingFace <https://huggingface.co/Helsinki-NLP/opus-mt-en-es>`__ |external-link|
   * - huggingface-translation-opus-mt-en-vi
     - False
     - 2.1.7
     - 2.189.0
     - Machine Translation
     - `HuggingFace <https://huggingface.co/Helsinki-NLP/opus-mt-en-vi>`__ |external-link|
   * - huggingface-translation-opus-mt-mul-en
     - False
     - 1.0.14
     - 2.189.0
     - Machine Translation
     - `HuggingFace <https://huggingface.co/Helsinki-NLP/opus-mt-mul-en>`__ |external-link|
   * - huggingface-translation-t5-base
     - False
     - 3.0.6
     - 2.189.0
     - Machine Translation
     - `HuggingFace <https://huggingface.co/google-t5/t5-base>`__ |external-link|
   * - huggingface-translation-t5-large
     - False
     - 3.0.6
     - 2.189.0
     - Machine Translation
     - `HuggingFace <https://huggingface.co/google-t5/t5-large>`__ |external-link|
   * - huggingface-translation-t5-small
     - False
     - 3.0.6
     - 2.189.0
     - Machine Translation
     - `HuggingFace <https://huggingface.co/google-t5/t5-small>`__ |external-link|
   * - huggingface-txt2img-22h-vintedois-diffusion-v0-1
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/22h/vintedois-diffusion-v0-1>`__ |external-link|
   * - huggingface-txt2img-akikagura-mkgen-diffusion
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/AkiKagura/mkgen-diffusion>`__ |external-link|
   * - huggingface-txt2img-alxdfy-noggles-fastdb-4800
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/alxdfy/noggles-fastdb-4800>`__ |external-link|
   * - huggingface-txt2img-alxdfy-noggles9000
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/alxdfy/noggles9000>`__ |external-link|
   * - huggingface-txt2img-andite-anything-v4-0
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/andite/anything-v4.0>`__ |external-link|
   * - huggingface-txt2img-astraliteheart-pony-diffusion-v2
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/AstraliteHeart/pony-diffusion-v2>`__ |external-link|
   * - huggingface-txt2img-avrik-abstract-anim-spritesheets
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/Avrik/abstract-anim-spritesheets>`__ |external-link|
   * - huggingface-txt2img-aybeeceedee-knollingcase
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/Aybeeceedee/knollingcase>`__ |external-link|
   * - huggingface-txt2img-bingsu-my-k-anything-v3-0
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/Bingsu/my-k-anything-v3-0>`__ |external-link|
   * - huggingface-txt2img-bingsu-my-korean-stable-diffusion-v1-5
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/Bingsu/my-korean-stable-diffusion-v1-5>`__ |external-link|
   * - huggingface-txt2img-black-forest-labs-flux-1-schnell
     - False
     - 3.0.2
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/black-forest-labs/FLUX.1-schnell>`__ |external-link|
   * - huggingface-txt2img-buntopsih-novgoranstefanovski
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/Buntopsih/novgoranstefanovski>`__ |external-link|
   * - huggingface-txt2img-claudfuen-photorealistic-fuen-v1
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/claudfuen/photorealistic-fuen-v1>`__ |external-link|
   * - huggingface-txt2img-coder119-vectorartz-diffusion
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/coder119/Vectorartz_Diffusion>`__ |external-link|
   * - huggingface-txt2img-conflictx-complex-lineart
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/Conflictx/Complex-Lineart>`__ |external-link|
   * - huggingface-txt2img-dallinmackay-cats-musical-diffusion
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/dallinmackay/Cats-Musical-diffusion>`__ |external-link|
   * - huggingface-txt2img-dallinmackay-jwst-deep-space-diffusion
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/dallinmackay/JWST-Deep-Space-diffusion>`__ |external-link|
   * - huggingface-txt2img-dallinmackay-tron-legacy-diffusion
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/dallinmackay/Tron-Legacy-diffusion>`__ |external-link|
   * - huggingface-txt2img-dallinmackay-van-gogh-diffusion
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/dallinmackay/Van-Gogh-diffusion>`__ |external-link|
   * - huggingface-txt2img-dgspitzer-cyberpunk-anime-diffusion
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/DGSpitzer/Cyberpunk-Anime-Diffusion>`__ |external-link|
   * - huggingface-txt2img-dreamlike-art-dreamlike-diffusion-1-0
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/dreamlike-art/dreamlike-diffusion-1.0>`__ |external-link|
   * - huggingface-txt2img-eimiss-eimisanimediffusion-1-0v
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/eimiss/EimisAnimeDiffusion_1.0v>`__ |external-link|
   * - huggingface-txt2img-envvi-inkpunk-diffusion
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/Envvi/Inkpunk-Diffusion>`__ |external-link|
   * - huggingface-txt2img-evel-yoshin
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/Evel/YoShin>`__ |external-link|
   * - huggingface-txt2img-extraphy-mustafa-kemal-ataturk
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/Extraphy/mustafa-kemal-ataturk>`__ |external-link|
   * - huggingface-txt2img-fffiloni-mr-men-and-little-misses
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/fffiloni/mr-men-and-little-misses>`__ |external-link|
   * - huggingface-txt2img-fictiverse-elrisitas
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/Fictiverse/ElRisitas>`__ |external-link|
   * - huggingface-txt2img-fictiverse-stable-diffusion-balloonart
     - False
     - 1.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/Fictiverse/Stable_Diffusion_BalloonArt_Model>`__ |external-link|
   * - huggingface-txt2img-fictiverse-stable-diffusion-balloonart-model
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/Fictiverse/Stable_Diffusion_BalloonArt_Model>`__ |external-link|
   * - huggingface-txt2img-fictiverse-stable-diffusion-micro-model
     - False
     - 1.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/Fictiverse/Stable_Diffusion_Microscopic_model>`__ |external-link|
   * - huggingface-txt2img-fictiverse-stable-diffusion-microscopic-model
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/Fictiverse/Stable_Diffusion_Microscopic_model>`__ |external-link|
   * - huggingface-txt2img-fictiverse-stable-diffusion-papercut-model
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/Fictiverse/Stable_Diffusion_PaperCut_Model>`__ |external-link|
   * - huggingface-txt2img-fictiverse-stable-diffusion-voxelart-model
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/Fictiverse/Stable_Diffusion_VoxelArt_Model>`__ |external-link|
   * - huggingface-txt2img-haor-evt-v3
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/haor/Evt_V3>`__ |external-link|
   * - huggingface-txt2img-hassanblend-hassanblend1-4
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/hassanblend/hassanblend1.4>`__ |external-link|
   * - huggingface-txt2img-idea-ccnl-taiyi-1b-chinese-en-v01
     - False
     - 1.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1>`__ |external-link|
   * - huggingface-txt2img-idea-ccnl-taiyi-1b-chinese-v0-1
     - False
     - 1.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1>`__ |external-link|
   * - huggingface-txt2img-idea-ccnl-taiyi-stable-diffusion-1b-chinese-en-v0-1
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1>`__ |external-link|
   * - huggingface-txt2img-idea-ccnl-taiyi-stable-diffusion-1b-chinese-v0-1
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1>`__ |external-link|
   * - huggingface-txt2img-ifansnek-johndiffusion
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/IfanSnek/JohnDiffusion>`__ |external-link|
   * - huggingface-txt2img-jersonm89-avatar
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/Jersonm89/Avatar>`__ |external-link|
   * - huggingface-txt2img-jvkape-iconsmi-appiconsmodelforsd
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/jvkape/IconsMI-AppIconsModelforSD>`__ |external-link|
   * - huggingface-txt2img-katakana-2d-mix
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/katakana/2D-Mix>`__ |external-link|
   * - huggingface-txt2img-lacambre-vulvine-look-v02
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/LaCambre/vulvine-look-v02>`__ |external-link|
   * - huggingface-txt2img-langboat-guohua-diffusion
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/Langboat/Guohua-Diffusion>`__ |external-link|
   * - huggingface-txt2img-linaqruf-anything-v3-0
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/Linaqruf/anything-v3.0>`__ |external-link|
   * - huggingface-txt2img-mikesmodels-waltz-with-bashir-diffusion
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/mikesmodels/Waltz_with_Bashir_Diffusion>`__ |external-link|
   * - huggingface-txt2img-mitchtech-klingon-diffusion
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/mitchtech/klingon-diffusion>`__ |external-link|
   * - huggingface-txt2img-mitchtech-vulcan-diffusion
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/mitchtech/vulcan-diffusion>`__ |external-link|
   * - huggingface-txt2img-mitsua-mitsua-diffusion-cc0
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/Mitsua/mitsua-diffusion-cc0>`__ |external-link|
   * - huggingface-txt2img-naclbit-trinart-stable-diffusion-v2
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/naclbit/trinart_stable_diffusion_v2>`__ |external-link|
   * - huggingface-txt2img-nitrosocke-arcane-diffusion
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/nitrosocke/Arcane-Diffusion>`__ |external-link|
   * - huggingface-txt2img-nitrosocke-archer-diffusion
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/nitrosocke/archer-diffusion>`__ |external-link|
   * - huggingface-txt2img-nitrosocke-classic-anim-diffusion
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/nitrosocke/classic-anim-diffusion>`__ |external-link|
   * - huggingface-txt2img-nitrosocke-elden-ring-diffusion
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/nitrosocke/elden-ring-diffusion>`__ |external-link|
   * - huggingface-txt2img-nitrosocke-future-diffusion
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/nitrosocke/Future-Diffusion>`__ |external-link|
   * - huggingface-txt2img-nitrosocke-ghibli-diffusion
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/nitrosocke/Ghibli-Diffusion>`__ |external-link|
   * - huggingface-txt2img-nitrosocke-mo-di-diffusion
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/nitrosocke/mo-di-diffusion>`__ |external-link|
   * - huggingface-txt2img-nitrosocke-nitro-diffusion
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/nitrosocke/Nitro-Diffusion>`__ |external-link|
   * - huggingface-txt2img-nitrosocke-redshift-diffusion
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/nitrosocke/redshift-diffusion>`__ |external-link|
   * - huggingface-txt2img-nitrosocke-spider-verse-diffusion
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/nitrosocke/spider-verse-diffusion>`__ |external-link|
   * - huggingface-txt2img-nousr-robo-diffusion
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/nousr/robo-diffusion>`__ |external-link|
   * - huggingface-txt2img-ogkalu-comic-diffusion
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/ogkalu/Comic-Diffusion>`__ |external-link|
   * - huggingface-txt2img-openjourney-openjourney
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/openjourney/openjourney>`__ |external-link|
   * - huggingface-txt2img-piesposito-openpotionbottle-v2
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/piEsposito/openpotionbottle-v2>`__ |external-link|
   * - huggingface-txt2img-plasmo-voxel-ish
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/plasmo/voxel-ish>`__ |external-link|
   * - huggingface-txt2img-plasmo-woolitize
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/plasmo/woolitize>`__ |external-link|
   * - huggingface-txt2img-progamergov-min-illust-background-diffusion
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/ProGamerGov/Min-Illust-Background-Diffusion>`__ |external-link|
   * - huggingface-txt2img-progamergov-min-illust-backgrounddiffusion
     - False
     - 1.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/ProGamerGov/Min-Illust-Background-Diffusion>`__ |external-link|
   * - huggingface-txt2img-prompthero-linkedin-diffusion
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/prompthero/linkedin-diffusion>`__ |external-link|
   * - huggingface-txt2img-prompthero-openjourney
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/prompthero/openjourney>`__ |external-link|
   * - huggingface-txt2img-qilex-magic-diffusion
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/Qilex/magic-diffusion>`__ |external-link|
   * - huggingface-txt2img-rabidgremlin-sd-db-epic-space-machine
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/rabidgremlin/sd-db-epic-space-machine>`__ |external-link|
   * - huggingface-txt2img-rayhell-popupbook-diffusion
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/RayHell/popupBook-diffusion>`__ |external-link|
   * - huggingface-txt2img-runwayml-stable-diffusion-v1-5
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/runwayml/stable-diffusion-v1-5>`__ |external-link|
   * - huggingface-txt2img-s3nh-beksinski-style-stable-diffusion
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/s3nh/beksinski-style-stable-diffusion>`__ |external-link|
   * - huggingface-txt2img-sd-dreambooth-library-original-char-cyclps
     - False
     - 1.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/sd-dreambooth-library/original-character-cyclps>`__ |external-link|
   * - huggingface-txt2img-sd-dreambooth-library-original-character-cyclps
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/sd-dreambooth-library/original-character-cyclps>`__ |external-link|
   * - huggingface-txt2img-sd-dreambooth-library-persona-5-shigenori
     - False
     - 1.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/sd-dreambooth-library/persona-5-shigenori-style>`__ |external-link|
   * - huggingface-txt2img-sd-dreambooth-library-persona-5-shigenori-style
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/sd-dreambooth-library/persona-5-shigenori-style>`__ |external-link|
   * - huggingface-txt2img-sd-dreambooth-library-seraphm
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/sd-dreambooth-library/seraphm>`__ |external-link|
   * - huggingface-txt2img-shirayu-sd-tohoku-v1
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/shirayu/sd-tohoku-v1>`__ |external-link|
   * - huggingface-txt2img-thelastben-hrrzg-style-768px
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/TheLastBen/hrrzg-style-768px>`__ |external-link|
   * - huggingface-txt2img-timothepearce-gina-the-cat
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/timothepearce/gina-the-cat>`__ |external-link|
   * - huggingface-txt2img-trystar-clonediffusion
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/TryStar/CloneDiffusion>`__ |external-link|
   * - huggingface-txt2img-tuwonga-dbluth
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/tuwonga/dbluth>`__ |external-link|
   * - huggingface-txt2img-tuwonga-rotoscopee
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/tuwonga/rotoscopee>`__ |external-link|
   * - huggingface-txt2img-volrath50-fantasy-card-diffusion
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/volrath50/fantasy-card-diffusion>`__ |external-link|
   * - huggingface-txt2img-yayab-sd-onepiece-diffusers4
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/YaYaB/sd-onepiece-diffusers4>`__ |external-link|
   * - huggingface-txt2imgneuron-stabilityai-stable-diffusion-v2-1
     - False
     - 1.1.9
     - 2.198.0
     - Source
     - `HuggingFace <https://huggingface.co/stabilityai/stable-diffusion-2-1>`__ |external-link|
   * - huggingface-txt2imgneuron-stabilityai-stable-diffusion-xl-base-1-0
     - False
     - 1.1.9
     - 2.198.0
     - Source
     - `HuggingFace <https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0>`__ |external-link|
   * - huggingface-txt2imgneuron-stabilityai-stable-diffusion-xlbase1
     - False
     - 1.1.9
     - 2.198.0
     - Source
     - `HuggingFace <https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0>`__ |external-link|
   * - huggingface-vlm-gemma-3-27b-instruct
     - False
     - 2.0.3
     - 2.225.0
     - Source
     - `HuggingFace <https://huggingface.co/google/gemma-3-27b-it>`__ |external-link|
   * - huggingface-vlm-gemma-3-4b-instruct
     - False
     - 1.0.2
     - 2.225.0
     - Source
     - `HuggingFace <https://huggingface.co/google/gemma-3-4b-it>`__ |external-link|
   * - huggingface-vlm-mistral-pixtral-12b-2409
     - False
     - 3.1.6
     - 2.225.0
     - Source
     - `HuggingFace <https://huggingface.co/mistralai/Pixtral-12B-2409>`__ |external-link|
   * - huggingface-vlm-qvq-72b-preview
     - False
     - 1.0.8
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/Qwen/QVQ-72B-Preview>`__ |external-link|
   * - huggingface-vlm-qwen2-vl-7b-instruct
     - False
     - 1.0.8
     - 2.237.1
     - Source
     - `HuggingFace <https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct>`__ |external-link|
   * - huggingface-zstc-cross-encoder-nli-deberta-base
     - False
     - 2.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/cross-encoder/nli-deberta-base>`__ |external-link|
   * - huggingface-zstc-cross-encoder-nli-distilroberta-base
     - False
     - 2.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/cross-encoder/nli-distilroberta-base>`__ |external-link|
   * - huggingface-zstc-cross-encoder-nli-minilm2-l6-h768
     - False
     - 2.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768>`__ |external-link|
   * - huggingface-zstc-cross-encoder-nli-roberta-base
     - False
     - 2.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/cross-encoder/nli-roberta-base>`__ |external-link|
   * - huggingface-zstc-digitalepidemiologylab-covid-twit-bert2-mnli
     - False
     - 1.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/digitalepidemiologylab/covid-twitter-bert-v2-mnli>`__ |external-link|
   * - huggingface-zstc-digitalepidemiologylab-covid-twitter-bert-v2-mnli
     - False
     - 2.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/digitalepidemiologylab/covid-twitter-bert-v2-mnli>`__ |external-link|
   * - huggingface-zstc-eleldar-theme-classification
     - False
     - 2.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/eleldar/theme-classification>`__ |external-link|
   * - huggingface-zstc-emrecan-bert-base-multilingual-cased-allnli-tr
     - False
     - 2.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-multilingual-cased-allnli_tr>`__ |external-link|
   * - huggingface-zstc-emrecan-bert-base-multilingual-cased-multinli-tr
     - False
     - 2.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-multilingual-cased-multinli_tr>`__ |external-link|
   * - huggingface-zstc-emrecan-bert-base-multilingual-cased-snli-tr
     - False
     - 2.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-multilingual-cased-snli_tr>`__ |external-link|
   * - huggingface-zstc-emrecan-bert-base-turkish-cased-allnli-tr
     - False
     - 2.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-turkish-cased-allnli_tr>`__ |external-link|
   * - huggingface-zstc-emrecan-bert-base-turkish-cased-multinli-tr
     - False
     - 2.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-turkish-cased-multinli_tr>`__ |external-link|
   * - huggingface-zstc-emrecan-bert-base-turkish-cased-snli-tr
     - False
     - 2.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-turkish-cased-snli_tr>`__ |external-link|
   * - huggingface-zstc-emrecan-bertbase-mling-cased-allnli-tr
     - False
     - 1.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-multilingual-cased-allnli_tr>`__ |external-link|
   * - huggingface-zstc-emrecan-bertbase-mling-cased-multinli-tr
     - False
     - 1.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-multilingual-cased-multinli_tr>`__ |external-link|
   * - huggingface-zstc-emrecan-cbertbase-turkish-mc4cased-multinlitr
     - False
     - 1.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/emrecan/convbert-base-turkish-mc4-cased-multinli_tr>`__ |external-link|
   * - huggingface-zstc-emrecan-cbertbase-turkish-mc4cased-snlitr
     - False
     - 1.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/emrecan/convbert-base-turkish-mc4-cased-snli_tr>`__ |external-link|
   * - huggingface-zstc-emrecan-cbertbase-turkishmc4-cased-allnlitr
     - False
     - 1.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/emrecan/convbert-base-turkish-mc4-cased-allnli_tr>`__ |external-link|
   * - huggingface-zstc-emrecan-convbert-base-turkish-mc4-cased-allnli-tr
     - False
     - 2.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/emrecan/convbert-base-turkish-mc4-cased-allnli_tr>`__ |external-link|
   * - huggingface-zstc-emrecan-convbert-base-turkish-mc4-cased-multinli-tr
     - False
     - 2.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/emrecan/convbert-base-turkish-mc4-cased-multinli_tr>`__ |external-link|
   * - huggingface-zstc-emrecan-convbert-base-turkish-mc4-cased-snli-tr
     - False
     - 2.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/emrecan/convbert-base-turkish-mc4-cased-snli_tr>`__ |external-link|
   * - huggingface-zstc-emrecan-dbase-turkish-cased-allnlitr
     - False
     - 1.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/emrecan/distilbert-base-turkish-cased-allnli_tr>`__ |external-link|
   * - huggingface-zstc-emrecan-dbertbase-turkish-cased-multinli-tr
     - False
     - 1.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/emrecan/distilbert-base-turkish-cased-multinli_tr>`__ |external-link|
   * - huggingface-zstc-emrecan-distilbert-base-turkish-cased-allnli-tr
     - False
     - 2.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/emrecan/distilbert-base-turkish-cased-allnli_tr>`__ |external-link|
   * - huggingface-zstc-emrecan-distilbert-base-turkish-cased-multinli-tr
     - False
     - 2.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/emrecan/distilbert-base-turkish-cased-multinli_tr>`__ |external-link|
   * - huggingface-zstc-emrecan-distilbert-base-turkish-cased-snli-tr
     - False
     - 2.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/emrecan/distilbert-base-turkish-cased-snli_tr>`__ |external-link|
   * - huggingface-zstc-facebook-bart-large-mnli
     - False
     - 2.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/facebook/bart-large-mnli>`__ |external-link|
   * - huggingface-zstc-jiva-xlm-roberta-large-it-mnli
     - False
     - 2.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/Jiva/xlm-roberta-large-it-mnli>`__ |external-link|
   * - huggingface-zstc-lighteternal-nli-xlm-r-greek
     - False
     - 2.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/lighteternal/nli-xlm-r-greek>`__ |external-link|
   * - huggingface-zstc-moritzlaurer-deberta-v3-large-mnli-fever-anli-ling-wanli
     - False
     - 2.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli>`__ |external-link|
   * - huggingface-zstc-moritzlaurer-deberta3large-mnli-fever
     - False
     - 1.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli>`__ |external-link|
   * - huggingface-zstc-moritzlaurer-mdeberta-v3-base-xnli-multilingual-nli-2mil7
     - False
     - 2.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7>`__ |external-link|
   * - huggingface-zstc-moritzlaurer-mdeberta3base-xnli-mling-nli-2m7
     - False
     - 1.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7>`__ |external-link|
   * - huggingface-zstc-narsil-bart-large-mnli-opti
     - False
     - 2.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/Narsil/bart-large-mnli-opti>`__ |external-link|
   * - huggingface-zstc-narsil-deberta-large-mnli-zero-cls
     - False
     - 2.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/Narsil/deberta-large-mnli-zero-cls>`__ |external-link|
   * - huggingface-zstc-navteca-bart-large-mnli
     - False
     - 2.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/navteca/bart-large-mnli>`__ |external-link|
   * - huggingface-zstc-recognai-bert-base-spanish-wwm-cased-xnli
     - False
     - 2.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/Recognai/bert-base-spanish-wwm-cased-xnli>`__ |external-link|
   * - huggingface-zstc-recognai-zeroshot-selectra-medium
     - False
     - 2.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/Recognai/zeroshot_selectra_medium>`__ |external-link|
   * - huggingface-zstc-recognai-zeroshot-selectra-small
     - False
     - 2.0.14
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/Recognai/zeroshot_selectra_small>`__ |external-link|
   * - lightgbm-classification-model
     - True
     - 2.2.7
     - 2.189.0
     - Classification
     - `LightGBM <https://lightgbm.readthedocs.io/en/latest/>`__ |external-link|
   * - lightgbm-regression-model
     - True
     - 2.2.7
     - 2.189.0
     - Regression
     - `LightGBM <https://lightgbm.readthedocs.io/en/latest/>`__ |external-link|
   * - meta-tc-llama-prompt-guard-86m
     - False
     - 1.2.5
     - 2.198.0
     - Text Classification
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-2-13b
     - True
     - 4.18.0
     - 2.225.0
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-2-13b-f
     - True
     - 4.17.0
     - 2.225.0
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-2-70b
     - True
     - 4.18.0
     - 2.225.0
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-2-70b-f
     - True
     - 4.17.0
     - 2.225.0
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-2-7b
     - True
     - 4.18.0
     - 2.225.0
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-2-7b-f
     - True
     - 4.17.0
     - 2.225.0
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-3-1-405b-fp8
     - True
     - 2.10.0
     - 2.237.1
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-3-1-405b-instruct-fp8
     - True
     - 2.10.0
     - 2.237.1
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-3-1-70b
     - True
     - 2.12.0
     - 2.232.1
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-3-1-70b-instruct
     - True
     - 2.12.0
     - 2.232.1
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-3-1-8b
     - True
     - 2.10.0
     - 2.232.1
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-3-1-8b-instruct
     - True
     - 2.10.0
     - 2.232.1
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-3-2-1b
     - True
     - 1.2.6
     - 2.225.0
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-3-2-1b-instruct
     - True
     - 1.2.6
     - 2.225.0
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-3-2-3b
     - True
     - 1.1.7
     - 2.225.0
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-3-2-3b-instruct
     - True
     - 1.1.7
     - 2.225.0
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-3-3-70b-instruct
     - False
     - 1.9.0
     - 2.237.1
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-3-70b
     - True
     - 2.17.0
     - 2.225.0
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-3-70b-instruct
     - True
     - 2.16.0
     - 2.225.0
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-3-8b
     - True
     - 2.16.0
     - 2.225.0
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-3-8b-instruct
     - True
     - 2.16.0
     - 2.225.0
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-codellama-13b
     - True
     - 3.17.0
     - 2.225.0
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-codellama-13b-instruct
     - False
     - 2.16.0
     - 2.225.0
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-codellama-13b-python
     - True
     - 3.17.0
     - 2.225.0
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-codellama-34b
     - True
     - 3.17.0
     - 2.225.0
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-codellama-34b-instruct
     - False
     - 2.16.0
     - 2.225.0
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-codellama-34b-python
     - True
     - 3.17.0
     - 2.225.0
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-codellama-70b
     - True
     - 2.17.0
     - 2.225.0
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-codellama-70b-instruct
     - False
     - 1.16.0
     - 2.225.0
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-codellama-70b-python
     - True
     - 2.17.0
     - 2.225.0
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-codellama-7b
     - True
     - 3.17.0
     - 2.225.0
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-codellama-7b-instruct
     - False
     - 2.16.0
     - 2.225.0
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-codellama-7b-python
     - True
     - 3.17.0
     - 2.225.0
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-guard-3-1b
     - False
     - 1.1.6
     - 2.225.0
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-guard-3-8b
     - False
     - 1.1.6
     - 2.225.0
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgeneration-llama-guard-7b
     - False
     - 1.2.14
     - 2.198.0
     - Text Generation
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgenerationneuron-llama-3-1-70b
     - False
     - 1.0.8
     - 2.229.0
     - Source
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgenerationneuron-llama-3-1-70b-instruct
     - False
     - 1.0.8
     - 2.229.0
     - Source
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgenerationneuron-llama-3-1-8b
     - False
     - 1.0.8
     - 2.229.0
     - Source
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgenerationneuron-llama-3-1-8b-instruct
     - False
     - 1.0.8
     - 2.229.0
     - Source
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgenerationneuron-llama-3-2-1b
     - False
     - 1.0.8
     - 2.229.0
     - Source
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgenerationneuron-llama-3-2-1b-instruct
     - False
     - 1.0.8
     - 2.229.0
     - Source
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgenerationneuron-llama-3-2-3b
     - False
     - 1.0.8
     - 2.229.0
     - Source
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgenerationneuron-llama-3-2-3b-instruct
     - False
     - 1.0.8
     - 2.229.0
     - Source
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgenerationneuron-llama-3-70b
     - False
     - 1.1.8
     - 2.198.0
     - Source
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgenerationneuron-llama-3-70b-instruct
     - False
     - 1.1.8
     - 2.198.0
     - Source
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgenerationneuron-llama-3-8b
     - False
     - 1.1.8
     - 2.198.0
     - Source
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgenerationneuron-llama-3-8b-instruct
     - False
     - 1.1.8
     - 2.198.0
     - Source
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgenerationneuron-llama-codellama-70b
     - False
     - 1.0.10
     - 2.198.0
     - Source
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgenerationneuron-llama-codellama-7b
     - False
     - 1.0.10
     - 2.198.0
     - Source
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgenerationneuron-llama-codellama-7b-python
     - False
     - 1.0.10
     - 2.198.0
     - Source
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgenerationneuron-llama-guard-3-1b
     - False
     - 1.0.8
     - 2.229.0
     - Source
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-textgenerationneuron-llama-guard-3-8b
     - False
     - 1.0.9
     - 2.229.0
     - Source
     - `Source <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`__ |external-link|
   * - meta-vs-sam-2-1-hiera-base-plus
     - False
     - 1.0.13
     - 2.237.1
     - Source
     - `Source <https://github.com/facebookresearch/segment-anything>`__ |external-link|
   * - meta-vs-sam-2-1-hiera-large
     - False
     - 1.0.13
     - 2.237.1
     - Source
     - `Source <https://github.com/facebookresearch/segment-anything>`__ |external-link|
   * - meta-vs-sam-2-1-hiera-small
     - False
     - 1.0.13
     - 2.237.1
     - Source
     - `Source <https://github.com/facebookresearch/segment-anything>`__ |external-link|
   * - meta-vs-sam-2-1-hiera-tiny
     - False
     - 1.0.13
     - 2.237.1
     - Source
     - `Source <https://github.com/facebookresearch/segment-anything>`__ |external-link|
   * - model-depth2img-stable-diffusion-2-depth-fp16
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/stabilityai/stable-diffusion-2-depth>`__ |external-link|
   * - model-depth2img-stable-diffusion-v1-5-controlnet
     - False
     - 2.1.5
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/lllyasviel/sd-controlnet-depth>`__ |external-link|
   * - model-depth2img-stable-diffusion-v1-5-controlnet-fp16
     - False
     - 2.1.5
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/lllyasviel/sd-controlnet-depth>`__ |external-link|
   * - model-depth2img-stable-diffusion-v1-5-controlnet-v1-1
     - False
     - 2.1.5
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth>`__ |external-link|
   * - model-depth2img-stable-diffusion-v1-5-controlnet-v1-1-fp16
     - False
     - 2.1.5
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth>`__ |external-link|
   * - model-depth2img-stable-diffusion-v2-1-controlnet
     - False
     - 2.1.5
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/thibaud/controlnet-sd21-depth-diffusers>`__ |external-link|
   * - model-depth2img-stable-diffusion-v2-1-controlnet-fp16
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/thibaud/controlnet-sd21-depth-diffusers>`__ |external-link|
   * - model-imagegeneration-stabilityai-stable-diffusion-v2-1
     - False
     - 1.0.11
     - 2.181.0
     - Source
     - `HuggingFace <https://huggingface.co/stabilityai/stable-diffusion-2-1-base>`__ |external-link|
   * - model-imagegeneration-stabilityai-stable-diffusion-xl-base-1-0
     - False
     - 1.0.12
     - 2.181.0
     - Source
     - `HuggingFace <https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0>`__ |external-link|
   * - model-inpainting-runwayml-stable-diffusion-inpainting
     - False
     - 2.1.6
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/runwayml/stable-diffusion-inpainting>`__ |external-link|
   * - model-inpainting-runwayml-stable-diffusion-inpainting-fp16
     - False
     - 2.1.6
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/runwayml/stable-diffusion-inpainting>`__ |external-link|
   * - model-inpainting-stabilityai-stable-diffusion-2-inpainting
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/stabilityai/stable-diffusion-2-inpainting>`__ |external-link|
   * - model-inpainting-stabilityai-stable-diffusion-2-inpainting-fp16
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/stabilityai/stable-diffusion-2-inpainting>`__ |external-link|
   * - model-inpainting-stabilityai-stable-diffusion2-inpainting-fp16
     - False
     - 1.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/stabilityai/stable-diffusion-2-inpainting>`__ |external-link|
   * - model-textgenerationjp-japanese-stablelm-instruct-alpha-7b-v2
     - False
     - 1.0.6
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/stabilityai/japanese-stablelm-instruct-alpha-7b-v2>`__ |external-link|
   * - model-txt2img-stabilityai-stable-diffusion-v1-4
     - False
     - 2.1.6
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/CompVis/stable-diffusion-v1-4>`__ |external-link|
   * - model-txt2img-stabilityai-stable-diffusion-v1-4-fp16
     - False
     - 2.1.6
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/CompVis/stable-diffusion-v1-4>`__ |external-link|
   * - model-txt2img-stabilityai-stable-diffusion-v2
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/stabilityai/stable-diffusion-2>`__ |external-link|
   * - model-txt2img-stabilityai-stable-diffusion-v2-1-base
     - True
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/stabilityai/stable-diffusion-2-1-base>`__ |external-link|
   * - model-txt2img-stabilityai-stable-diffusion-v2-fp16
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/stabilityai/stable-diffusion-2>`__ |external-link|
   * - model-upscaling-stabilityai-stable-diffusion-x4-upscaler-fp16
     - False
     - 2.0.13
     - 2.189.0
     - Source
     - `HuggingFace <https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler>`__ |external-link|
   * - mxnet-is-mask-rcnn-fpn-resnet101-v1d-coco
     - False
     - 2.0.14
     - 2.189.0
     - Instance Segmentation
     - `GluonCV <https://cv.gluon.ai/model_zoo/segmentation.html>`__ |external-link|
   * - mxnet-is-mask-rcnn-fpn-resnet18-v1b-coco
     - False
     - 2.0.14
     - 2.189.0
     - Instance Segmentation
     - `GluonCV <https://cv.gluon.ai/model_zoo/segmentation.html>`__ |external-link|
   * - mxnet-is-mask-rcnn-fpn-resnet50-v1b-coco
     - False
     - 2.0.14
     - 2.189.0
     - Instance Segmentation
     - `GluonCV <https://cv.gluon.ai/model_zoo/segmentation.html>`__ |external-link|
   * - mxnet-is-mask-rcnn-resnet101-v1d-coco
     - False
     - 1.0.5
     - 2.189.0
     - Instance Segmentation
     - `GluonCV <https://cv.gluon.ai/model_zoo/segmentation.html>`__ |external-link|
   * - mxnet-is-mask-rcnn-resnet18-v1b-coco
     - False
     - 2.0.14
     - 2.189.0
     - Instance Segmentation
     - `GluonCV <https://cv.gluon.ai/model_zoo/segmentation.html>`__ |external-link|
   * - mxnet-is-mask-rcnn-resnet50-v1b-coco
     - False
     - 1.0.5
     - 2.189.0
     - Instance Segmentation
     - `GluonCV <https://cv.gluon.ai/model_zoo/segmentation.html>`__ |external-link|
   * - mxnet-od-faster-rcnn-fpn-resnet101-v1d-coco
     - False
     - 2.0.14
     - 2.189.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-faster-rcnn-fpn-resnet50-v1b-coco
     - False
     - 2.0.14
     - 2.189.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-faster-rcnn-resnet101-v1d-coco
     - False
     - 2.0.14
     - 2.189.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-faster-rcnn-resnet50-v1b-coco
     - False
     - 2.0.14
     - 2.189.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-faster-rcnn-resnet50-v1b-voc
     - False
     - 2.0.14
     - 2.189.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-ssd-300-vgg16-atrous-coco
     - True
     - 2.0.14
     - 2.189.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-ssd-300-vgg16-atrous-voc
     - True
     - 2.0.14
     - 2.189.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-ssd-512-mobilenet1-0-coco
     - True
     - 2.0.14
     - 2.189.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-ssd-512-mobilenet1-0-voc
     - True
     - 2.0.14
     - 2.189.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-ssd-512-resnet50-v1-coco
     - True
     - 2.0.14
     - 2.189.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-ssd-512-resnet50-v1-voc
     - True
     - 2.0.14
     - 2.189.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-ssd-512-vgg16-atrous-coco
     - True
     - 2.0.14
     - 2.189.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-ssd-512-vgg16-atrous-voc
     - True
     - 2.0.14
     - 2.189.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-yolo3-darknet53-coco
     - False
     - 2.0.14
     - 2.189.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-yolo3-darknet53-voc
     - False
     - 2.0.14
     - 2.189.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-yolo3-mobilenet1-0-coco
     - False
     - 2.0.14
     - 2.189.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-yolo3-mobilenet1-0-voc
     - False
     - 2.0.14
     - 2.189.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-semseg-fcn-resnet101-ade
     - True
     - 2.0.14
     - 2.189.0
     - Semantic Segmentation
     - `GluonCV <https://cv.gluon.ai/model_zoo/segmentation.html>`__ |external-link|
   * - mxnet-semseg-fcn-resnet101-coco
     - True
     - 2.0.14
     - 2.189.0
     - Semantic Segmentation
     - `GluonCV <https://cv.gluon.ai/model_zoo/segmentation.html>`__ |external-link|
   * - mxnet-semseg-fcn-resnet101-voc
     - True
     - 2.0.14
     - 2.189.0
     - Semantic Segmentation
     - `GluonCV <https://cv.gluon.ai/model_zoo/segmentation.html>`__ |external-link|
   * - mxnet-semseg-fcn-resnet50-ade
     - True
     - 2.0.14
     - 2.189.0
     - Semantic Segmentation
     - `GluonCV <https://cv.gluon.ai/model_zoo/segmentation.html>`__ |external-link|
   * - mxnet-tcembedding-robertafin-base-uncased
     - False
     - 2.0.14
     - 2.189.0
     - Text Embedding
     - `GluonCV <https://nlp.gluon.ai/master/_modules/gluonnlp/models/roberta.html>`__ |external-link|
   * - mxnet-tcembedding-robertafin-base-wiki-uncased
     - False
     - 2.0.14
     - 2.189.0
     - Text Embedding
     - `GluonCV <https://nlp.gluon.ai/master/_modules/gluonnlp/models/roberta.html>`__ |external-link|
   * - mxnet-tcembedding-robertafin-large-uncased
     - False
     - 2.0.14
     - 2.189.0
     - Text Embedding
     - `GluonCV <https://nlp.gluon.ai/master/_modules/gluonnlp/models/roberta.html>`__ |external-link|
   * - mxnet-tcembedding-robertafin-large-wiki-uncased
     - False
     - 2.0.14
     - 2.189.0
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
     - 3.1.7
     - 2.189.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_alexnet/>`__ |external-link|
   * - pytorch-ic-densenet121
     - True
     - 3.1.7
     - 2.189.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_densenet/>`__ |external-link|
   * - pytorch-ic-densenet161
     - True
     - 3.1.7
     - 2.189.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_densenet/>`__ |external-link|
   * - pytorch-ic-densenet169
     - True
     - 3.1.7
     - 2.189.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_densenet/>`__ |external-link|
   * - pytorch-ic-densenet201
     - True
     - 3.1.7
     - 2.189.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_densenet/>`__ |external-link|
   * - pytorch-ic-googlenet
     - True
     - 3.1.7
     - 2.189.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_googlenet/>`__ |external-link|
   * - pytorch-ic-mobilenet-v2
     - True
     - 3.0.17
     - 2.189.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_mobilenet_v2/>`__ |external-link|
   * - pytorch-ic-resnet101
     - True
     - 3.1.7
     - 2.189.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_resnet/>`__ |external-link|
   * - pytorch-ic-resnet152
     - True
     - 3.1.7
     - 2.189.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_resnet/>`__ |external-link|
   * - pytorch-ic-resnet18
     - True
     - 3.1.7
     - 2.189.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_resnet/>`__ |external-link|
   * - pytorch-ic-resnet34
     - True
     - 3.1.7
     - 2.189.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_resnet/>`__ |external-link|
   * - pytorch-ic-resnet50
     - True
     - 3.1.7
     - 2.189.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_resnet/>`__ |external-link|
   * - pytorch-ic-resnext101-32x8d
     - True
     - 3.1.7
     - 2.189.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_resnext/>`__ |external-link|
   * - pytorch-ic-resnext50-32x4d
     - True
     - 3.1.7
     - 2.189.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_resnext/>`__ |external-link|
   * - pytorch-ic-shufflenet-v2-x1-0
     - True
     - 3.1.7
     - 2.189.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_shufflenet_v2/>`__ |external-link|
   * - pytorch-ic-squeezenet1-0
     - True
     - 3.1.7
     - 2.189.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_squeezenet/>`__ |external-link|
   * - pytorch-ic-squeezenet1-1
     - True
     - 3.1.7
     - 2.189.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_squeezenet/>`__ |external-link|
   * - pytorch-ic-vgg11
     - True
     - 3.1.7
     - 2.189.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_vgg/>`__ |external-link|
   * - pytorch-ic-vgg11-bn
     - True
     - 3.1.7
     - 2.189.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_vgg/>`__ |external-link|
   * - pytorch-ic-vgg13
     - True
     - 3.1.7
     - 2.189.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_vgg/>`__ |external-link|
   * - pytorch-ic-vgg13-bn
     - True
     - 3.1.7
     - 2.189.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_vgg/>`__ |external-link|
   * - pytorch-ic-vgg16
     - True
     - 3.1.7
     - 2.189.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_vgg/>`__ |external-link|
   * - pytorch-ic-vgg16-bn
     - True
     - 3.1.7
     - 2.189.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_vgg/>`__ |external-link|
   * - pytorch-ic-vgg19
     - True
     - 3.1.7
     - 2.189.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_vgg/>`__ |external-link|
   * - pytorch-ic-vgg19-bn
     - True
     - 3.1.7
     - 2.189.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_vgg/>`__ |external-link|
   * - pytorch-ic-wide-resnet101-2
     - True
     - 3.1.7
     - 2.189.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_wide_resnet/>`__ |external-link|
   * - pytorch-ic-wide-resnet50-2
     - True
     - 3.1.7
     - 2.189.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_wide_resnet/>`__ |external-link|
   * - pytorch-od-nvidia-ssd
     - False
     - 2.0.18
     - 2.189.0
     - Object Detection
     - `Pytorch Hub <https://pytorch.org/hub/nvidia_deeplearningexamples_ssd/>`__ |external-link|
   * - pytorch-od1-fasterrcnn-mobilenet-v3-large-320-fpn
     - False
     - 2.0.15
     - 2.189.0
     - Object Detection
     - `Pytorch Hub <https://pytorch.org/vision/stable/_modules/torchvision/models/detection/faster_rcnn.html>`__ |external-link|
   * - pytorch-od1-fasterrcnn-mobilenet-v3-large-fpn
     - False
     - 2.0.15
     - 2.189.0
     - Object Detection
     - `Pytorch Hub <https://pytorch.org/vision/stable/_modules/torchvision/models/detection/faster_rcnn.html>`__ |external-link|
   * - pytorch-od1-fasterrcnn-resnet50-fpn
     - True
     - 2.0.15
     - 2.189.0
     - Object Detection
     - `Pytorch Hub <https://pytorch.org/vision/stable/_modules/torchvision/models/detection/faster_rcnn.html>`__ |external-link|
   * - pytorch-semseg-deeplabv3-mobilenet-v3-large
     - False
     - 1.0.0
     - 2.75.0
     - Semantic Segmentation
     - `Pytorch Hub <https://pytorch.org/vision/>`__ |external-link|
   * - pytorch-semseg-deeplabv3-resnet101
     - False
     - 1.0.0
     - 2.75.0
     - Semantic Segmentation
     - `Pytorch Hub <https://pytorch.org/vision/>`__ |external-link|
   * - pytorch-semseg-deeplabv3-resnet50
     - False
     - 1.0.0
     - 2.75.0
     - Semantic Segmentation
     - `Pytorch Hub <https://pytorch.org/vision/>`__ |external-link|
   * - pytorch-semseg-fcn-resnet101
     - False
     - 1.0.0
     - 2.75.0
     - Semantic Segmentation
     - `Pytorch Hub <https://pytorch.org/vision/>`__ |external-link|
   * - pytorch-semseg-fcn-resnet50
     - False
     - 1.0.0
     - 2.75.0
     - Semantic Segmentation
     - `Pytorch Hub <https://pytorch.org/vision/>`__ |external-link|
   * - pytorch-tabtransformerclassification-model
     - True
     - 2.0.18
     - 2.189.0
     - Source
     - `Source <https://arxiv.org/abs/2012.06678>`__ |external-link|
   * - pytorch-tabtransformerregression-model
     - True
     - 2.0.18
     - 2.189.0
     - Source
     - `Source <https://arxiv.org/abs/2012.06678>`__ |external-link|
   * - pytorch-textgeneration1-alexa20b
     - False
     - 2.0.16
     - 2.189.0
     - Source
     - `Source <https://www.amazon.science/blog/20b-parameter-alexa-model-sets-new-marks-in-few-shot-learning>`__ |external-link|
   * - sklearn-classification-linear
     - True
     - 1.3.9
     - 2.188.0
     - Classification
     - `ScikitLearn <https://scikit-learn.org/stable/>`__ |external-link|
   * - sklearn-classification-snowflake
     - True
     - 1.1.9
     - 2.188.0
     - Classification
     - `ScikitLearn <https://scikit-learn.org/stable/>`__ |external-link|
   * - sklearn-regression-linear
     - True
     - 1.3.9
     - 2.188.0
     - Regression
     - `ScikitLearn <https://scikit-learn.org/stable/>`__ |external-link|
   * - sklearn-regression-snowflake
     - True
     - 1.1.9
     - 2.188.0
     - Regression
     - `ScikitLearn <https://scikit-learn.org/stable/>`__ |external-link|
   * - tensorflow-audioembedding-frill-1
     - False
     - 2.0.15
     - 2.189.0
     - Source
     - `Tensorflow Hub <https://tfhub.dev/google/nonsemantic-speech-benchmark/frill/1>`__ |external-link|
   * - tensorflow-audioembedding-trill-3
     - False
     - 2.0.15
     - 2.189.0
     - Source
     - `Tensorflow Hub <https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/3>`__ |external-link|
   * - tensorflow-audioembedding-trill-distilled-3
     - False
     - 2.0.15
     - 2.189.0
     - Source
     - `Tensorflow Hub <https://tfhub.dev/google/nonsemantic-speech-benchmark/trill-distilled/3>`__ |external-link|
   * - tensorflow-audioembedding-trillsson1-1
     - False
     - 2.0.15
     - 2.189.0
     - Source
     - `Tensorflow Hub <https://tfhub.dev/google/nonsemantic-speech-benchmark/trillsson1/1>`__ |external-link|
   * - tensorflow-audioembedding-trillsson2-1
     - False
     - 2.0.15
     - 2.189.0
     - Source
     - `Tensorflow Hub <https://tfhub.dev/google/nonsemantic-speech-benchmark/trillsson2/1>`__ |external-link|
   * - tensorflow-audioembedding-trillsson3-1
     - False
     - 2.0.15
     - 2.189.0
     - Source
     - `Tensorflow Hub <https://tfhub.dev/google/nonsemantic-speech-benchmark/trillsson3/1>`__ |external-link|
   * - tensorflow-ic-bit-m-r101x1-ilsvrc2012-classification-1
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r101x1/ilsvrc2012_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-m-r101x1-imagenet21k-classification-1
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r101x1/imagenet21k_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-m-r101x3-ilsvrc2012-classification-1
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r101x3/ilsvrc2012_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-m-r101x3-imagenet21k-classification-1
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r101x3/imagenet21k_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-m-r152x4-ilsvrc2012
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r152x4/ilsvrc2012_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-m-r152x4-imagenet21k
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r152x4/imagenet21k_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-m-r50x1-ilsvrc2012-classification-1
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r50x1/ilsvrc2012_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-m-r50x1-imagenet21k-classification-1
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r50x1/imagenet21k_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-m-r50x3-ilsvrc2012-classification-1
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r50x3/ilsvrc2012_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-m-r50x3-imagenet21k-classification-1
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r50x3/imagenet21k_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-s-r101x1-ilsvrc2012-classification-1
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r101x1/ilsvrc2012_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-s-r101x3-ilsvrc2012-classification-1
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r101x3/ilsvrc2012_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-s-r152x4-ilsvrc2012
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r152x4/ilsvrc2012_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-s-r50x1-ilsvrc2012-classification-1
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r50x1/ilsvrc2012_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-s-r50x3-ilsvrc2012-classification-1
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r50x3/ilsvrc2012_classification/1>`__ |external-link|
   * - tensorflow-ic-cait-m36-384
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/cait_m36_384/1>`__ |external-link|
   * - tensorflow-ic-cait-m48-448
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/cait_m48_448/1>`__ |external-link|
   * - tensorflow-ic-cait-s24-224
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/cait_s24_224/1>`__ |external-link|
   * - tensorflow-ic-cait-s24-384
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/cait_s24_384/1>`__ |external-link|
   * - tensorflow-ic-cait-s36-384
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/cait_s36_384/1>`__ |external-link|
   * - tensorflow-ic-cait-xs24-384
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/cait_xs24_384/1>`__ |external-link|
   * - tensorflow-ic-cait-xxs24-224
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/cait_xxs24_224/1>`__ |external-link|
   * - tensorflow-ic-cait-xxs24-384
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/cait_xxs24_384/1>`__ |external-link|
   * - tensorflow-ic-cait-xxs36-224
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/cait_xxs36_224/1>`__ |external-link|
   * - tensorflow-ic-cait-xxs36-384
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/cait_xxs36_384/1>`__ |external-link|
   * - tensorflow-ic-deit-base-distilled-patch16-224
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/deit_base_distilled_patch16_224/1>`__ |external-link|
   * - tensorflow-ic-deit-base-distilled-patch16-384
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/deit_base_distilled_patch16_384/1>`__ |external-link|
   * - tensorflow-ic-deit-base-patch16-224
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/deit_base_patch16_224/1>`__ |external-link|
   * - tensorflow-ic-deit-base-patch16-384
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/deit_base_patch16_384/1>`__ |external-link|
   * - tensorflow-ic-deit-small-distilled-patch16-224
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/deit_small_distilled_patch16_224/1>`__ |external-link|
   * - tensorflow-ic-deit-small-patch16-224
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/deit_small_patch16_224/1>`__ |external-link|
   * - tensorflow-ic-deit-tiny-distilled-patch16-224
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/deit_tiny_distilled_patch16_224/1>`__ |external-link|
   * - tensorflow-ic-deit-tiny-patch16-224
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/deit_tiny_patch16_224/1>`__ |external-link|
   * - tensorflow-ic-efficientnet-b0-classification-1
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/b0/classification/1>`__ |external-link|
   * - tensorflow-ic-efficientnet-b1-classification-1
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/b1/classification/1>`__ |external-link|
   * - tensorflow-ic-efficientnet-b2-classification-1
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/b2/classification/1>`__ |external-link|
   * - tensorflow-ic-efficientnet-b3-classification-1
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/b3/classification/1>`__ |external-link|
   * - tensorflow-ic-efficientnet-b4-classification-1
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/b4/classification/1>`__ |external-link|
   * - tensorflow-ic-efficientnet-b5-classification-1
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/b5/classification/1>`__ |external-link|
   * - tensorflow-ic-efficientnet-b6-classification-1
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/b6/classification/1>`__ |external-link|
   * - tensorflow-ic-efficientnet-b7-classification-1
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/b7/classification/1>`__ |external-link|
   * - tensorflow-ic-efficientnet-lite0-classification-2
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite0/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-lite1-classification-2
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite1/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-lite2-classification-2
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite2/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-lite3-classification-2
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite3/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-lite4-classification-2
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite4/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet1k-b0
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet1k-b1
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b1/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet1k-b2
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b2/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet1k-b3
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b3/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet1k-l
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_l/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet1k-m
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_m/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet1k-s
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet21k-b0
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b0/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet21k-b1
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b1/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet21k-b2
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b2/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet21k-b3
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b3/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet21k-ft1k-b0
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b0/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet21k-ft1k-b1
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b1/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet21k-ft1k-b2
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b2/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet21k-ft1k-b3
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b3/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet21k-ft1k-l
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_l/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet21k-ft1k-m
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_m/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet21k-ft1k-s
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_s/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet21k-ft1k-xl
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_xl/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet21k-l
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_l/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet21k-m
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_m/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet21k-s
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_s/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-v2-imagenet21k-xl
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_xl/classification/2>`__ |external-link|
   * - tensorflow-ic-imagenet-inception-resnet-v2-classification-4
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-inception-v1-classification-4
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/inception_v1/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-inception-v2-classification-4
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/inception_v2/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-inception-v3-classification-4
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/inception_v3/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-025-128-classification-4
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_128/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-025-160-classification-4
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_160/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-025-192-classification-4
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_192/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-025-224-classification-4
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-050-128-classification-4
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_128/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-050-160-classification-4
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-050-192-classification-4
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_192/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-050-224-classification-4
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_224/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-075-128-classification-4
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_128/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-075-160-classification-4
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_160/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-075-192-classification-4
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_192/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-075-224-classification-4
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_224/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-100-128-classification-4
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_128/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-100-160-classification-4
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_160/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-100-192-classification-4
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_192/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-100-224-classification-4
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-035-128
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_035_128/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-035-160
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_035_160/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-035-192
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_035_192/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-035-224-classification-4
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_035_224/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-035-96
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_035_96/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-050-128
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_050_128/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-050-160
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_050_160/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-050-192
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_050_192/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-050-224-classification-4
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_050_224/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-050-96
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_050_96/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-075-128
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_075_128/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-075-160
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_075_160/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-075-192
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_075_192/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-075-224-classification-4
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_075_224/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-075-96
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_075_96/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-100-160
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_100_160/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-100-192
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_100_192/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-100-224-classification-4
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-100-96
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-130-224-classification-4
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-140-224-classification-4
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v3-large-075-224
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v3-large-100-224
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v3-small-075-224
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v3_small_075_224/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v3-small-100-224
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-nasnet-large
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/nasnet_large/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-nasnet-mobile
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/nasnet_mobile/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-pnasnet-large
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/pnasnet_large/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-resnet-v1-101-classification-4
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v1_101/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-resnet-v1-152-classification-4
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v1_152/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-resnet-v1-50-classification-4
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v1_50/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-resnet-v2-101-classification-4
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v2_101/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-resnet-v2-152-classification-4
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v2_152/classification/5>`__ |external-link|
   * - tensorflow-ic-imagenet-resnet-v2-50-classification-4
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v2_50/classification/5>`__ |external-link|
   * - tensorflow-ic-resnet-50-classification-1
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/resnet_50/classification/1>`__ |external-link|
   * - tensorflow-ic-swin-base-patch4-window12-384
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/swin_base_patch4_window12_384/1>`__ |external-link|
   * - tensorflow-ic-swin-base-patch4-window7-224
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/swin_base_patch4_window7_224/1>`__ |external-link|
   * - tensorflow-ic-swin-large-patch4-window12-384
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/swin_large_patch4_window12_384/1>`__ |external-link|
   * - tensorflow-ic-swin-large-patch4-window7-224
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/swin_large_patch4_window7_224/1>`__ |external-link|
   * - tensorflow-ic-swin-s3-base-224
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/swin_s3_base_224/1>`__ |external-link|
   * - tensorflow-ic-swin-s3-small-224
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/swin_s3_small_224/1>`__ |external-link|
   * - tensorflow-ic-swin-s3-tiny-224
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/swin_s3_tiny_224/1>`__ |external-link|
   * - tensorflow-ic-swin-small-patch4-window7-224
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/swin_small_patch4_window7_224/1>`__ |external-link|
   * - tensorflow-ic-swin-tiny-patch4-window7-224
     - True
     - 2.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/swin_tiny_patch4_window7_224/1>`__ |external-link|
   * - tensorflow-ic-tf2-preview-inception-v3-classification-4
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/tf2-preview/inception_v3/classification/4>`__ |external-link|
   * - tensorflow-ic-tf2-preview-mobilenet-v2-classification-4
     - True
     - 4.0.17
     - 2.189.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4>`__ |external-link|
   * - tensorflow-icembedding-bit-m-r101x1-ilsvrc2012-featurevector-1
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r101x1/1>`__ |external-link|
   * - tensorflow-icembedding-bit-m-r101x3-imagenet21k-featurevector-1
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r101x3/1>`__ |external-link|
   * - tensorflow-icembedding-bit-m-r101x3-imagenet21k-fv-1
     - False
     - 1.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r101x3/1>`__ |external-link|
   * - tensorflow-icembedding-bit-m-r50x1-ilsvrc2012-featurevector-1
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r50x1/1>`__ |external-link|
   * - tensorflow-icembedding-bit-m-r50x3-imagenet21k-featurevector-1
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r50x3/1>`__ |external-link|
   * - tensorflow-icembedding-bit-s-r101x1-ilsvrc2012-featurevector-1
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r101x1/1>`__ |external-link|
   * - tensorflow-icembedding-bit-s-r101x3-ilsvrc2012-featurevector-1
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r101x3/1>`__ |external-link|
   * - tensorflow-icembedding-bit-s-r50x1-ilsvrc2012-featurevector-1
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r50x1/1>`__ |external-link|
   * - tensorflow-icembedding-bit-s-r50x3-ilsvrc2012-featurevector-1
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r50x3/1>`__ |external-link|
   * - tensorflow-icembedding-efficientnet-b0-featurevector-1
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b0/feature-vector/1>`__ |external-link|
   * - tensorflow-icembedding-efficientnet-b1-featurevector-1
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b1/feature-vector/1>`__ |external-link|
   * - tensorflow-icembedding-efficientnet-b2-featurevector-1
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b2/feature-vector/1>`__ |external-link|
   * - tensorflow-icembedding-efficientnet-b3-featurevector-1
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b3/feature-vector/1>`__ |external-link|
   * - tensorflow-icembedding-efficientnet-b6-featurevector-1
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b6/feature-vector/1>`__ |external-link|
   * - tensorflow-icembedding-efficientnet-lite0-featurevector-2
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2>`__ |external-link|
   * - tensorflow-icembedding-efficientnet-lite1-featurevector-2
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite1/feature-vector/2>`__ |external-link|
   * - tensorflow-icembedding-efficientnet-lite2-featurevector-2
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite2/feature-vector/2>`__ |external-link|
   * - tensorflow-icembedding-efficientnet-lite3-featurevector-2
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite3/feature-vector/2>`__ |external-link|
   * - tensorflow-icembedding-efficientnet-lite4-featurevector-2
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite4/feature-vector/2>`__ |external-link|
   * - tensorflow-icembedding-imagenet-inception-v1-featurevector-4
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/inception_v1/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-inception-v2-featurevector-4
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/inception_v2/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-inception-v3-featurevector-4
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-025-128-featurevector-4
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_128/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-025-128-fv-4
     - False
     - 1.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_128/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-025-160-featurevector-4
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_160/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-025-160-fv-4
     - False
     - 1.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_160/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-025-192-featurevector-4
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_192/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-025-192-fv-4
     - False
     - 1.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_192/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-025-224-featurevector-4
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-025-224-fv-4
     - False
     - 1.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-050-128-featurevector-4
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_128/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-050-128-fv-4
     - False
     - 1.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_128/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-050-160-featurevector-4
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-050-160-fv-4
     - False
     - 1.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-050-192-featurevector-4
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_192/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-050-192-fv-4
     - False
     - 1.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_192/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-050-224-featurevector-4
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-050-224-fv-4
     - False
     - 1.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-075-128-featurevector-4
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_128/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-075-128-fv-4
     - False
     - 1.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_128/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-075-160-featurevector-4
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_160/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-075-160-fv-4
     - False
     - 1.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_160/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-075-192-featurevector-4
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_192/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-075-192-fv-4
     - False
     - 1.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_192/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-075-224-featurevector-4
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-075-224-fv-4
     - False
     - 1.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-100-128-featurevector-4
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_128/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-100-128-fv-4
     - False
     - 1.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_128/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-100-160-featurevector-4
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_160/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-100-160-fv-4
     - False
     - 1.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_160/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-100-192-featurevector-4
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_192/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-100-192-fv-4
     - False
     - 1.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_192/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-100-224-featurevector-4
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-100-224-fv-4
     - False
     - 1.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v2-035-224-featurevector-4
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_035_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v2-035-224-fv-4
     - False
     - 1.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_035_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v2-050-224-featurevector-4
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_050_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v2-050-224-fv-4
     - False
     - 1.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_050_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v2-075-224-featurevector-4
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_075_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v2-075-224-fv-4
     - False
     - 1.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_075_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v2-100-224-featurevector-4
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v2-100-224-fv-4
     - False
     - 1.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v2-130-224-featurevector-4
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v2-130-224-fv-4
     - False
     - 1.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v2-140-224-featurevector-4
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v2-140-224-fv-4
     - False
     - 1.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-resnet-v1-101-featurevector-4
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v1_101/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-resnet-v1-152-featurevector-4
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v1_152/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-resnet-v1-50-featurevector-4
     - False
     - 2.0.17
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-resnet-v2-101-featurevector-4
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-resnet-v2-152-featurevector-4
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-resnet-v2-50-featurevector-4
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-resnet-50-featurevector-1
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/resnet_50/feature_vector/1>`__ |external-link|
   * - tensorflow-icembedding-tf2-preview-inception-v3-featurevector-4
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-tf2-preview-inception-v3-fv-4
     - False
     - 1.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-tf2-preview-mobilenet-v2-featurevector-4
     - False
     - 3.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-tf2-preview-mobilenet-v2-fv-4
     - False
     - 1.0.15
     - 2.189.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4>`__ |external-link|
   * - tensorflow-od-centernet-hourglass-1024x1024-1
     - False
     - 3.0.15
     - 2.189.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/hourglass_1024x1024/1>`__ |external-link|
   * - tensorflow-od-centernet-hourglass-1024x1024-kpts-1
     - False
     - 3.0.15
     - 2.189.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/hourglass_1024x1024_kpts/1>`__ |external-link|
   * - tensorflow-od-centernet-hourglass-512x512-1
     - False
     - 3.0.15
     - 2.189.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1>`__ |external-link|
   * - tensorflow-od-centernet-hourglass-512x512-kpts-1
     - False
     - 3.0.15
     - 2.189.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/hourglass_512x512_kpts/1>`__ |external-link|
   * - tensorflow-od-centernet-resnet101v1-fpn-512x512-1
     - False
     - 3.0.15
     - 2.189.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/resnet101v1_fpn_512x512/1>`__ |external-link|
   * - tensorflow-od-centernet-resnet50v1-fpn-512x512-1
     - False
     - 3.0.15
     - 2.189.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512/1>`__ |external-link|
   * - tensorflow-od-centernet-resnet50v1-fpn-512x512-kpts-1
     - False
     - 3.0.15
     - 2.189.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512_kpts/1>`__ |external-link|
   * - tensorflow-od-centernet-resnet50v2-512x512-1
     - False
     - 3.0.15
     - 2.189.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/resnet50v2_512x512/1>`__ |external-link|
   * - tensorflow-od-centernet-resnet50v2-512x512-kpts-1
     - False
     - 3.0.15
     - 2.189.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/resnet50v2_512x512_kpts/1>`__ |external-link|
   * - tensorflow-od-efficientdet-d0-1
     - False
     - 3.0.15
     - 2.189.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientdet/d0/1>`__ |external-link|
   * - tensorflow-od-efficientdet-d1-1
     - False
     - 3.0.15
     - 2.189.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientdet/d1/1>`__ |external-link|
   * - tensorflow-od-efficientdet-d2-1
     - False
     - 3.0.15
     - 2.189.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientdet/d2/1>`__ |external-link|
   * - tensorflow-od-efficientdet-d3-1
     - False
     - 3.0.15
     - 2.189.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientdet/d3/1>`__ |external-link|
   * - tensorflow-od-efficientdet-d4-1
     - False
     - 3.0.15
     - 2.189.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientdet/d4/1>`__ |external-link|
   * - tensorflow-od-efficientdet-d5-1
     - False
     - 3.0.15
     - 2.189.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientdet/d5/1>`__ |external-link|
   * - tensorflow-od-faster-rcnn-inception-resnet-v2-1024x1024-1
     - False
     - 3.0.15
     - 2.189.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_1024x1024/1>`__ |external-link|
   * - tensorflow-od-faster-rcnn-inception-resnet-v2-640x640-1
     - False
     - 3.0.15
     - 2.189.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1>`__ |external-link|
   * - tensorflow-od-faster-rcnn-resnet101-v1-1024x1024-1
     - False
     - 3.0.15
     - 2.189.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_1024x1024/1>`__ |external-link|
   * - tensorflow-od-faster-rcnn-resnet101-v1-640x640-1
     - False
     - 3.0.15
     - 2.189.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_640x640/1>`__ |external-link|
   * - tensorflow-od-faster-rcnn-resnet101-v1-800x1333-1
     - False
     - 3.0.15
     - 2.189.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_800x1333/1>`__ |external-link|
   * - tensorflow-od-faster-rcnn-resnet152-v1-1024x1024-1
     - False
     - 3.0.15
     - 2.189.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_1024x1024/1>`__ |external-link|
   * - tensorflow-od-faster-rcnn-resnet152-v1-640x640-1
     - False
     - 3.0.15
     - 2.189.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_640x640/1>`__ |external-link|
   * - tensorflow-od-faster-rcnn-resnet152-v1-800x1333-1
     - False
     - 3.0.15
     - 2.189.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_800x1333/1>`__ |external-link|
   * - tensorflow-od-faster-rcnn-resnet50-v1-1024x1024-1
     - False
     - 3.0.15
     - 2.189.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_1024x1024/1>`__ |external-link|
   * - tensorflow-od-faster-rcnn-resnet50-v1-640x640-1
     - False
     - 3.0.15
     - 2.189.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1>`__ |external-link|
   * - tensorflow-od-faster-rcnn-resnet50-v1-800x1333-1
     - False
     - 3.0.15
     - 2.189.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_800x1333/1>`__ |external-link|
   * - tensorflow-od-retinanet-resnet101-v1-fpn-1024x1024-1
     - False
     - 3.0.15
     - 2.189.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/retinanet/resnet101_v1_fpn_1024x1024/1>`__ |external-link|
   * - tensorflow-od-retinanet-resnet101-v1-fpn-640x640-1
     - False
     - 3.0.15
     - 2.189.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/retinanet/resnet101_v1_fpn_640x640/1>`__ |external-link|
   * - tensorflow-od-retinanet-resnet152-v1-fpn-1024x1024-1
     - False
     - 3.0.15
     - 2.189.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/retinanet/resnet152_v1_fpn_1024x1024/1>`__ |external-link|
   * - tensorflow-od-retinanet-resnet152-v1-fpn-640x640-1
     - False
     - 3.0.15
     - 2.189.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/retinanet/resnet152_v1_fpn_640x640/1>`__ |external-link|
   * - tensorflow-od-retinanet-resnet50-v1-fpn-1024x1024-1
     - False
     - 3.0.15
     - 2.189.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/retinanet/resnet50_v1_fpn_1024x1024/1>`__ |external-link|
   * - tensorflow-od-retinanet-resnet50-v1-fpn-640x640-1
     - False
     - 3.0.15
     - 2.189.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/retinanet/resnet50_v1_fpn_640x640/1>`__ |external-link|
   * - tensorflow-od-ssd-mobilenet-v1-fpn-640x640-1
     - False
     - 3.0.15
     - 2.189.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/ssd_mobilenet_v1/fpn_640x640/1>`__ |external-link|
   * - tensorflow-od-ssd-mobilenet-v2-2
     - False
     - 3.0.15
     - 2.189.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2>`__ |external-link|
   * - tensorflow-od-ssd-mobilenet-v2-fpnlite-320x320-1
     - False
     - 3.0.15
     - 2.189.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1>`__ |external-link|
   * - tensorflow-od-ssd-mobilenet-v2-fpnlite-640x640-1
     - False
     - 3.0.15
     - 2.189.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_640x640/1>`__ |external-link|
   * - tensorflow-od1-ssd-efficientdet-d0-512x512-coco17-tpu-8
     - True
     - 2.0.15
     - 2.189.0
     - Object Detection
     - `Source <http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz>`__ |external-link|
   * - tensorflow-od1-ssd-efficientdet-d1-640x640-coco17-tpu-8
     - True
     - 2.0.15
     - 2.189.0
     - Object Detection
     - `Source <http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz>`__ |external-link|
   * - tensorflow-od1-ssd-efficientdet-d2-768x768-coco17-tpu-8
     - True
     - 2.0.15
     - 2.189.0
     - Object Detection
     - `Source <http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d2_coco17_tpu-32.tar.gz>`__ |external-link|
   * - tensorflow-od1-ssd-efficientdet-d3-896x896-coco17-tpu-32
     - True
     - 2.0.15
     - 2.189.0
     - Object Detection
     - `Source <http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d3_coco17_tpu-32.tar.gz>`__ |external-link|
   * - tensorflow-od1-ssd-mobilenet-v1-fpn-640x640-coco17-tpu-8
     - True
     - 2.0.15
     - 2.189.0
     - Object Detection
     - `Source <http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz>`__ |external-link|
   * - tensorflow-od1-ssd-mobilenet-v2-fpnlite-320x320-coco17-tpu-8
     - True
     - 2.0.15
     - 2.189.0
     - Object Detection
     - `Source <http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz>`__ |external-link|
   * - tensorflow-od1-ssd-mobilenet-v2-fpnlite-640x640-coco17-tpu-8
     - True
     - 2.0.15
     - 2.189.0
     - Object Detection
     - `Source <http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz>`__ |external-link|
   * - tensorflow-od1-ssd-resnet101-v1-fpn-1024x1024-coco17-tpu-8
     - True
     - 2.0.15
     - 2.189.0
     - Object Detection
     - `Source <http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8.tar.gz>`__ |external-link|
   * - tensorflow-od1-ssd-resnet101-v1-fpn-640x640-coco17-tpu-8
     - True
     - 2.0.15
     - 2.189.0
     - Object Detection
     - `Source <http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz>`__ |external-link|
   * - tensorflow-od1-ssd-resnet152-v1-fpn-1024x1024-coco17-tpu-8
     - True
     - 2.0.15
     - 2.189.0
     - Object Detection
     - `Source <http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8.tar.gz>`__ |external-link|
   * - tensorflow-od1-ssd-resnet152-v1-fpn-640x640-coco17-tpu-8
     - True
     - 2.0.15
     - 2.189.0
     - Object Detection
     - `Source <http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_640x640_coco17_tpu-8.tar.gz>`__ |external-link|
   * - tensorflow-od1-ssd-resnet50-v1-fpn-1024x1024-coco17-tpu-8
     - True
     - 2.0.15
     - 2.189.0
     - Object Detection
     - `Source <http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8.tar.gz>`__ |external-link|
   * - tensorflow-od1-ssd-resnet50-v1-fpn-640x640-coco17-tpu-8
     - True
     - 2.0.15
     - 2.189.0
     - Object Detection
     - `Source <http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz>`__ |external-link|
   * - tensorflow-spc-bert-en-cased-L-12-H-768-A-12-2
     - True
     - 2.0.16
     - 2.189.0
     - Sentence Pair Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/2>`__ |external-link|
   * - tensorflow-spc-bert-en-uncased-L-12-H-768-A-12-2
     - True
     - 2.0.16
     - 2.189.0
     - Sentence Pair Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2>`__ |external-link|
   * - tensorflow-spc-bert-en-uncased-L-24-H-1024-A-16-2
     - True
     - 2.0.16
     - 2.189.0
     - Sentence Pair Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/2>`__ |external-link|
   * - tensorflow-spc-bert-en-wwm-cased-L-24-H-1024-A-16-2
     - True
     - 2.0.16
     - 2.189.0
     - Sentence Pair Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_wwm_cased_L-24_H-1024_A-16/2>`__ |external-link|
   * - tensorflow-spc-bert-en-wwm-uncased-L-24-H-1024-A-16-2
     - True
     - 2.0.16
     - 2.189.0
     - Sentence Pair Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_wwm_uncased_L-24_H-1024_A-16/2>`__ |external-link|
   * - tensorflow-spc-bert-multi-cased-L-12-H-768-A-12-2
     - True
     - 2.0.16
     - 2.189.0
     - Sentence Pair Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/2>`__ |external-link|
   * - tensorflow-spc-electra-base-1
     - True
     - 2.0.16
     - 2.189.0
     - Sentence Pair Classification
     - `Tensorflow Hub <https://tfhub.dev/google/electra_base/1>`__ |external-link|
   * - tensorflow-spc-electra-small-1
     - True
     - 2.0.16
     - 2.189.0
     - Sentence Pair Classification
     - `Tensorflow Hub <https://tfhub.dev/google/electra_small/1>`__ |external-link|
   * - tensorflow-spc-experts-bert-pubmed-1
     - True
     - 2.0.16
     - 2.189.0
     - Sentence Pair Classification
     - `Tensorflow Hub <https://tfhub.dev/google/experts/bert/pubmed/1>`__ |external-link|
   * - tensorflow-spc-experts-bert-wiki-books-1
     - True
     - 2.0.16
     - 2.189.0
     - Sentence Pair Classification
     - `Tensorflow Hub <https://tfhub.dev/google/experts/bert/wiki_books/1>`__ |external-link|
   * - tensorflow-tc-albert-en-base
     - True
     - 2.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/albert_en_base/2>`__ |external-link|
   * - tensorflow-tc-bert-en-cased-L-12-H-768-A-12-2
     - True
     - 3.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3>`__ |external-link|
   * - tensorflow-tc-bert-en-cased-L-24-H-1024-A-16-2
     - True
     - 3.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_cased_L-24_H-1024_A-16/3>`__ |external-link|
   * - tensorflow-tc-bert-en-uncased-L-12-H-768-A-12-2
     - True
     - 3.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3>`__ |external-link|
   * - tensorflow-tc-bert-en-uncased-L-24-H-1024-A-16-2
     - True
     - 2.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/3>`__ |external-link|
   * - tensorflow-tc-bert-en-wwm-cased-L-24-H-1024-A-16-2
     - True
     - 3.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_wwm_cased_L-24_H-1024_A-16/3>`__ |external-link|
   * - tensorflow-tc-bert-en-wwm-uncased-L-24-H-1024-A-16-2
     - True
     - 3.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_wwm_uncased_L-24_H-1024_A-16/3>`__ |external-link|
   * - tensorflow-tc-bert-multi-cased-L-12-H-768-A-12-2
     - True
     - 3.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3>`__ |external-link|
   * - tensorflow-tc-electra-base-1
     - True
     - 3.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/google/electra_base/2>`__ |external-link|
   * - tensorflow-tc-electra-small-1
     - True
     - 3.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/google/electra_small/2>`__ |external-link|
   * - tensorflow-tc-experts-bert-pubmed-1
     - True
     - 3.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/google/experts/bert/pubmed/2>`__ |external-link|
   * - tensorflow-tc-experts-bert-wiki-books-1
     - True
     - 3.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/google/experts/bert/wiki_books/2>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-10-H-128-A-2
     - True
     - 2.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-10-H-256-A-4
     - True
     - 2.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-10-H-512-A-8
     - True
     - 2.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-10-H-768-A-12
     - True
     - 2.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-12-H-128-A-2
     - True
     - 2.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-12-H-256-A-4
     - True
     - 2.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-12-H-512-A-8
     - True
     - 2.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-12-H-768-A-12
     - True
     - 2.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-2-H-128-A-2
     - True
     - 2.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-2-H-256-A-4
     - True
     - 2.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-2-H-512-A-8
     - True
     - 2.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-2-H-768-A-12
     - True
     - 2.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-4-H-128-A-2
     - True
     - 2.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-4-H-256-A-4
     - True
     - 2.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-4-H-512-A-8
     - True
     - 2.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-4-H-768-A-12
     - True
     - 2.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-6-H-128-A-2
     - True
     - 2.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-6-H-256-A-4
     - True
     - 2.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-6-H-512-A-8
     - True
     - 2.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-6-H-768-A-12
     - True
     - 2.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-8-H-128-A-2
     - True
     - 2.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-8-H-256-A-4
     - True
     - 2.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-8-H-512-A-8
     - True
     - 2.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1>`__ |external-link|
   * - tensorflow-tc-small-bert-bert-en-uncased-L-8-H-768-A-12
     - True
     - 2.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1>`__ |external-link|
   * - tensorflow-tc-talking-heads-base
     - True
     - 2.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1>`__ |external-link|
   * - tensorflow-tc-talking-heads-large
     - True
     - 2.0.15
     - 2.189.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_large/1>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-10-H-128-A-2-2
     - False
     - 2.0.15
     - 2.189.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-10-H-256-A-4-2
     - False
     - 2.0.15
     - 2.189.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-10-H-512-A-8-2
     - False
     - 2.0.15
     - 2.189.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-10-H-768-A-12-2
     - False
     - 2.0.15
     - 2.189.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-12-H-128-A-2-2
     - False
     - 2.0.15
     - 2.189.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-12-H-256-A-4
     - False
     - 2.0.15
     - 2.189.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-12-H-512-A-8-2
     - False
     - 2.0.15
     - 2.189.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-12-H-768-A-12-2
     - False
     - 2.0.15
     - 2.189.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-12-H-768-A-12-4
     - False
     - 2.0.15
     - 2.189.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-2-H-128-A-2-2
     - False
     - 2.0.15
     - 2.189.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-2-H-256-A-4
     - False
     - 2.0.15
     - 2.189.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-2-H-512-A-8-2
     - False
     - 2.0.15
     - 2.189.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-2-H-768-A-12-2
     - False
     - 2.0.15
     - 2.189.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-4-H-128-A-2-2
     - False
     - 2.0.15
     - 2.189.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-4-H-256-A-4-2
     - False
     - 2.0.15
     - 2.189.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-4-H-512-A-8-2
     - False
     - 2.0.15
     - 2.189.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-4-H-768-A-12-2
     - False
     - 2.0.15
     - 2.189.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-6-H-128-A-2-2
     - False
     - 2.0.15
     - 2.189.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-6-H-256-A-4
     - False
     - 2.0.15
     - 2.189.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-6-H-512-A-8-2
     - False
     - 2.0.15
     - 2.189.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-6-H-768-A-12-2
     - False
     - 2.0.15
     - 2.189.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-8-H-256-A-4-2
     - False
     - 2.0.15
     - 2.189.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-8-H-512-A-8-2
     - False
     - 2.0.15
     - 2.189.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-8-H-768-A-12-2
     - False
     - 2.0.15
     - 2.189.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-wiki-books-mnli-2
     - False
     - 2.0.15
     - 2.189.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/experts/bert/wiki_books/mnli/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-wiki-books-sst2
     - False
     - 2.0.15
     - 2.189.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/experts/bert/wiki_books/sst2/2>`__ |external-link|
   * - tensorflow-tcembedding-talkheads-ggelu-bert-en-base-2
     - False
     - 2.0.15
     - 2.189.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/2>`__ |external-link|
   * - tensorflow-tcembedding-talkheads-ggelu-bert-en-large-2
     - False
     - 2.0.15
     - 2.189.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_large/2>`__ |external-link|
   * - tensorflow-tcembedding-universal-sentc-encoder-cmlm-en-base-1
     - False
     - 1.0.15
     - 2.189.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/1>`__ |external-link|
   * - tensorflow-tcembedding-universal-sentc-encoder-cmlm-en-large-1
     - False
     - 1.0.15
     - 2.189.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-large/1>`__ |external-link|
   * - tensorflow-tcembedding-universal-sentence-encoder-cmlm-en-base-1
     - False
     - 2.0.15
     - 2.189.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/1>`__ |external-link|
   * - tensorflow-tcembedding-universal-sentence-encoder-cmlm-en-large-1
     - False
     - 2.0.15
     - 2.189.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-large/1>`__ |external-link|
   * - xgboost-classification-model
     - True
     - 2.1.12
     - 2.188.0
     - Classification
     - `XGBoost <https://xgboost.readthedocs.io/en/release_1.7.0/>`__ |external-link|
   * - xgboost-classification-snowflake
     - True
     - 1.1.12
     - 2.188.0
     - Classification
     - `XGBoost <https://xgboost.readthedocs.io/en/release_1.7.0/>`__ |external-link|
   * - xgboost-regression-model
     - True
     - 2.1.12
     - 2.188.0
     - Regression
     - `XGBoost <https://xgboost.readthedocs.io/en/release_1.7.0/>`__ |external-link|
   * - xgboost-regression-snowflake
     - True
     - 1.1.12
     - 2.188.0
     - Regression
     - `XGBoost <https://xgboost.readthedocs.io/en/release_1.7.0/>`__ |external-link|

.. list-table:: Available Proprietary Models
   :widths: 50 20 20 20 20
   :header-rows: 1
   :class: datatable

   * - Model ID
     - Fine Tunable?
     - Supported Version
     - Min SDK Version
     - Source
   * - nvidia-nemotron-4-15b-nim
     - False
     - 1.2.3
     - 2.237.2
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-cjge44tau4g36>`__ |external-link|
   * - jinaai-embeddings-v2-base-en
     - False
     - 3.2
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-5iljbegvoi66w>`__ |external-link|
   * - cohere-command-r-a100
     - False
     - v1.5.0
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-w7ukdez7zfjfo>`__ |external-link|
   * - cohere-command-r-h100
     - False
     - v1.5.0
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-jfhyfeewxqbr2>`__ |external-link|
   * - cohere-command-r-plus-a100
     - False
     - v1.5.0
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-xfznx42jopjfy>`__ |external-link|
   * - cohere-command-r-plus-h100
     - False
     - v1.5.0
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-jnw3ilzmy7tcg>`__ |external-link|
   * - cohere-command-r-08-2024-h100
     - False
     - v1.0.5
     - 2.237.2
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-n5frh4bneaukw>`__ |external-link|
   * - cohere-command-r-plus-08-2024-h100
     - False
     - v1.0.5
     - 2.237.2
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-xjjtezrkhsoyc>`__ |external-link|
   * - cohere-embed-multilingual
     - False
     - v3.3.9
     - 2.237.2
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-b4mpgdxvpa3v6>`__ |external-link|
   * - cohere-embed-english
     - False
     - v3.6.0
     - 2.237.2
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-qd64mji3pbnvk>`__ |external-link|
   * - cohere-embed-light-multilingual
     - False
     - v3.3.9
     - 2.237.2
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-7uw3bomgawhre>`__ |external-link|
   * - cohere-embed-light-english
     - False
     - v3.3.9
     - 2.237.2
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-c4dwczm5vr6bs>`__ |external-link|
   * - lgresearch-exaone
     - False
     - 1.0.0
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-pzm7ute4auwz2>`__ |external-link|
   * - ncsoft-ko-13b-ist
     - False
     - v1.1.0
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-usyosolf3an3u>`__ |external-link|
   * - ncsoft-ko-6-4b-ist
     - False
     - v1.0.0
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-orzrv3od3xiru>`__ |external-link|
   * - ncsoft-ko-1-3b-ist
     - False
     - v1.0.0
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-n4hluwlfe7nns>`__ |external-link|
   * - ncsoft-llama-3-varco-offsetbias-8b
     - False
     - v1.1.0
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-mv3tify4tfrbk>`__ |external-link|
   * - stabilityai-stable-diffusion-3-5-large
     - False
     - SD3_5L_v1
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-ajc3gw4mjy7my>`__ |external-link|
   * - stabilityai-sdxl-1-0
     - False
     - 20230726
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-pe7wqwehghdtm>`__ |external-link|
   * - ai21-summarization
     - False
     - 1.2.000
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-dkwy6chb63hk2>`__ |external-link|
   * - lighton-mini-instruct40b
     - False
     - v1.0
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-ahkh5ofqnojzm>`__ |external-link|
   * - ai21-paraphrase
     - False
     - 1.0.005
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-6ivt5p34sljua>`__ |external-link|
   * - lighton-lyra-fr
     - False
     - v1.1
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-3o34zcamifo3i>`__ |external-link|
   * - ai21-jurassic-2-light
     - False
     - 2.0.004
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-roz6zicyvi666>`__ |external-link|
   * - ai21-jurassic-2-grande-instruct
     - False
     - 2.2.004
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-bzjpjkgd542au>`__ |external-link|
   * - stabilityai-sdxl-beta-0-8
     - False
     - 1.0-rc3
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-3j5jzj4k6slxs>`__ |external-link|
   * - ai21-contextual-answers
     - False
     - 2.2.001
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-gwbjdp3tmh3bw>`__ |external-link|
   * - ai21-jurassic-2-jumbo-instruct
     - False
     - 2.2.004
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-f4y5ksmu5kccy>`__ |external-link|
   * - cohere-rerank-english-v2
     - False
     - v2.0.3
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-xwsyvhz7rkjqe>`__ |external-link|
   * - cohere-rerank-multilingual-v2
     - False
     - v2.0.3
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-pf7d2umihcseq>`__ |external-link|
   * - upstage-solar-mini-chat-quant
     - False
     - v1.2.0
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-5z6tpjgbfxbjg>`__ |external-link|
   * - upstage-solar-pro
     - False
     - 250422.0
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-yar5lgioxenj4>`__ |external-link|
   * - upstage-solar-pro-quantized
     - False
     - 250422.0
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-tnlnnbd73jklg>`__ |external-link|
   * - voyage-2-embedding
     - False
     - v1.0
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-hmlsydfvb5xnc>`__ |external-link|
   * - voyage-large-2-embedding
     - False
     - v1.0.1
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-erofjpgna7gtq>`__ |external-link|
   * - voyage-code-2-embedding
     - False
     - v1.0.1
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-ofmg2lztgphzy>`__ |external-link|
   * - nomic-embed-text
     - False
     - 0.0.3
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-xume634dhbnyu>`__ |external-link|
   * - nomic-embed-image
     - False
     - v0.0.3
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-okk2aryiqiway>`__ |external-link|
   * - evolutionary-scale-esm3
     - False
     - 1.05
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-xbvra5ylcu4xq>`__ |external-link|
   * - cohere-rerank-nimble-english
     - False
     - v1.0.5
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-rq7ik6yx6jnzc>`__ |external-link|
   * - cohere-rerank-nimble-multi
     - False
     - v1.0.4
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-ea3rcr6y56jp2>`__ |external-link|
   * - cohere-rerank-v3-english
     - False
     - v1.0.8
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-rqhxjsjanb3gy>`__ |external-link|
   * - cohere-rerank-v3-multilingual
     - False
     - v1.0.9
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-ydysc72qticsw>`__ |external-link|
   * - bria-ai-2-3-commercial
     - False
     - 2.3
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-man54dmpkarki>`__ |external-link|
   * - bria-ai-2-3-fast-commercial
     - False
     - 2.3-Fast
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-qwwlgkbtm2bsq>`__ |external-link|
   * - bria-ai-2-2-hd-commercial
     - False
     - HD
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-2pbgsqtuvobbq>`__ |external-link|
   * - liquid-lfm-7b-l40s
     - False
     - 1.0.0
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-a7hfjtjhloy36>`__ |external-link|
   * - liquid-lfm-40b-a100
     - False
     - 1.0.9
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-a4lnuwrcusyb2>`__ |external-link|
   * - liquid-lfm-40b-h100
     - False
     - 1.0.8
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-fg547nle7mkvq>`__ |external-link|
   * - liquid-lfm-40b-l40s
     - False
     - 1.0.8
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-qifkyrmmcu7n4>`__ |external-link|
   * - john-snow-labs-summarization-qa
     - False
     - 5.4.4
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-yrajldynampw4>`__ |external-link|
   * - john-snow-labs-medical-summarization-qa-8b
     - False
     - 5.4.7
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-z4jqmczvwgtby>`__ |external-link|
   * - john-snow-labs-medical-translation-en-es
     - False
     - 5.4.2
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-lwpnc66c6bcug>`__ |external-link|
   * - upstage-document-layout-analysis
     - False
     - 250508c.1
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-lv5bnpdco7xoq>`__ |external-link|
   * - upstage-document-ocr
     - False
     - 2.2.1-2
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-anvrh24vv3yiw>`__ |external-link|
   * - upstage-solar-embedding-large
     - False
     - v1.0.0
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-dldbdmws5l52i>`__ |external-link|
   * - arcee-lite
     - False
     - v2
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-pbwtcva3ym5kw>`__ |external-link|
   * - arcee-supernova
     - False
     - v2
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-sb2ndlhwmzbhi>`__ |external-link|
   * - arcee-llama-spark
     - False
     - v1
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-qia6qe4xocov2>`__ |external-link|
   * - arcee-llama-3-1-supernova-lite
     - False
     - v3
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-ekwgfr2qryqrw>`__ |external-link|
   * - voyage-rerank-lite-1-reranker
     - False
     - v1
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-pv546iwekelmc>`__ |external-link|
   * - writer-palmyra-fin-70b-32k
     - False
     - v1.0.0
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-yaembld3zumja>`__ |external-link|
   * - writer-palmyra-med-70b-32k
     - False
     - v1.0.0
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-qwwrqvgyxvgx6>`__ |external-link|
   * - ibm-granite-20b-code-instruct-8k
     - False
     - v1
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-ezh4cr7om23rm>`__ |external-link|
   * - ibm-granite-34b-code-instruct-8k
     - False
     - v1
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-brtbmvlgxmayw>`__ |external-link|
   * - ibm-granite-3b-code-instruct-128k
     - False
     - v1
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-wb6hb4222lc3y>`__ |external-link|
   * - ibm-granite-8b-code-instruct-128k
     - False
     - v1
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-v2bk72ufp4enq>`__ |external-link|
   * - karakuri-lm-8x7b-instruct
     - False
     - v0.1.1
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-pwsqrkkq2rxv4>`__ |external-link|
   * - stockmark-llm-13b
     - False
     - 20241010
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-wwlcs37oj2vlq>`__ |external-link|
   * - exaone-v3-0-7-8b-instruct
     - False
     - v3.0.0
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-yg5edmwlh4tko>`__ |external-link|
   * - granite-3-0-2b-instruct
     - False
     - v1
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-btj7x52r77uza>`__ |external-link|
   * - granite-3-0-8b-instruct
     - False
     - v1
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-vweaesrobaglk>`__ |external-link|
   * - writer-palmyra-x-004
     - False
     - v1.0.0
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-hkfl25soo6b5g>`__ |external-link|
   * - widn-tower-anthill
     - False
     - v4.0
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-xskn6yectpscq>`__ |external-link|
   * - widn-tower-sugarloaf
     - False
     - v3.1
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-l7e262ywsn4cy>`__ |external-link|
   * - widn-llama3-tower-vesuvius
     - False
     - v4.0
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-p6qef5uamaqb6>`__ |external-link|
   * - cambai-mars6
     - False
     - 1.0.2
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-pfu4q6zws4j76>`__ |external-link|
   * - preferred-networks-plamo-api
     - False
     - 0.5.1.4bit
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-eybaqh3shlfmk>`__ |external-link|
   * - gretel-navigator-tabular
     - False
     - 1.0.1
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-rlwqitrye62bi>`__ |external-link|
   * - bioptimus-h-optimus-0
     - False
     - 1.0.0
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-vqdtpo4pyoicy>`__ |external-link|
   * - orbital-materials-orb
     - False
     - v1
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-ysg3nhoa7sewu>`__ |external-link|
   * - voyage-3-embedding
     - False
     - v1.0.0
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-qe5h5bzldzkts>`__ |external-link|
   * - arcee-virtuoso-small
     - False
     - v1
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-rz3vgq54p5izq>`__ |external-link|
   * - nvidia-llama3-2-nv-embedqa-1b-v2-nim
     - False
     - v1.3.3
     - 2.239.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-lbys3gfdwi37o>`__ |external-link|
   * - nvidia-llama3-2-nv-rerankqa-1b-v2-nim
     - False
     - v1.3.3
     - 2.239.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-yidbbfz2m25aq>`__ |external-link|
   * - ibm-granite-3-2-2b-instruct
     - False
     - v1.1
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-4rr3lx6tg224a>`__ |external-link|
   * - ibm-granite-3-2-8b-instruct
     - False
     - v1.1
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-ljgnfeqd6dezu>`__ |external-link|
   * - cohere-embed-v4-0
     - False
     - v.2.0
     - 2.213.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-g53hj27nurqc6>`__ |external-link|
   * - nvidia-nemotron-nano-8b-nim
     - False
     - v1.8.0
     - 2.239.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-6wde2hzvft7y4>`__ |external-link|
   * - nvidia-nemotron-super-49b-nim
     - False
     - v1.8.0
     - 2.239.0
     - `Source <https://aws.amazon.com/marketplace/pp/prodview-tnxnt6nqrnore>`__ |external-link|

