################################################
Zero-Shot Text Classification - HuggingFace
################################################

This is a zero-shot text classification algorithm which supports many pre-trained models available in HuggingFace Hub. The following
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
   * - huggingface-zstc-cross-encoder-nli-deberta-base
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/cross-encoder/nli-deberta-base>`__
   * - huggingface-zstc-cross-encoder-nli-distilroberta-base
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/cross-encoder/nli-distilroberta-base>`__
   * - huggingface-zstc-cross-encoder-nli-minilm2-l6-h768
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768>`__
   * - huggingface-zstc-cross-encoder-nli-roberta-base
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/cross-encoder/nli-roberta-base>`__
   * - huggingface-zstc-digitalepidemiologylab-covid-twitter-bert-v2-mnli
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/digitalepidemiologylab/covid-twitter-bert-v2-mnli>`__
   * - huggingface-zstc-eleldar-theme-classification
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/eleldar/theme-classification>`__
   * - huggingface-zstc-emrecan-bert-base-multilingual-cased-allnli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-multilingual-cased-allnli_tr>`__
   * - huggingface-zstc-emrecan-bert-base-multilingual-cased-multinli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-multilingual-cased-multinli_tr>`__
   * - huggingface-zstc-emrecan-bert-base-multilingual-cased-snli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-multilingual-cased-snli_tr>`__
   * - huggingface-zstc-emrecan-bert-base-turkish-cased-allnli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-turkish-cased-allnli_tr>`__
   * - huggingface-zstc-emrecan-bert-base-turkish-cased-multinli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-turkish-cased-multinli_tr>`__
   * - huggingface-zstc-emrecan-bert-base-turkish-cased-snli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-turkish-cased-snli_tr>`__
   * - huggingface-zstc-emrecan-convbert-base-turkish-mc4-cased-allnli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/convbert-base-turkish-mc4-cased-allnli_tr>`__
   * - huggingface-zstc-emrecan-convbert-base-turkish-mc4-cased-multinli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/convbert-base-turkish-mc4-cased-multinli_tr>`__
   * - huggingface-zstc-emrecan-convbert-base-turkish-mc4-cased-snli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/convbert-base-turkish-mc4-cased-snli_tr>`__
   * - huggingface-zstc-emrecan-distilbert-base-turkish-cased-allnli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/distilbert-base-turkish-cased-allnli_tr>`__
   * - huggingface-zstc-emrecan-distilbert-base-turkish-cased-multinli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/distilbert-base-turkish-cased-multinli_tr>`__
   * - huggingface-zstc-emrecan-distilbert-base-turkish-cased-snli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/distilbert-base-turkish-cased-snli_tr>`__
   * - huggingface-zstc-facebook-bart-large-mnli
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/facebook/bart-large-mnli>`__
   * - huggingface-zstc-jiva-xlm-roberta-large-it-mnli
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/Jiva/xlm-roberta-large-it-mnli>`__
   * - huggingface-zstc-lighteternal-nli-xlm-r-greek
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/lighteternal/nli-xlm-r-greek>`__
   * - huggingface-zstc-moritzlaurer-deberta-v3-large-mnli-fever-anli-ling-wanli
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli>`__
   * - huggingface-zstc-moritzlaurer-mdeberta-v3-base-xnli-multilingual-nli-2mil7
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7>`__
   * - huggingface-zstc-narsil-bart-large-mnli-opti
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/Narsil/bart-large-mnli-opti>`__
   * - huggingface-zstc-narsil-deberta-large-mnli-zero-cls
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/Narsil/deberta-large-mnli-zero-cls>`__
   * - huggingface-zstc-navteca-bart-large-mnli
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/navteca/bart-large-mnli>`__
   * - huggingface-zstc-recognai-bert-base-spanish-wwm-cased-xnli
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/Recognai/bert-base-spanish-wwm-cased-xnli>`__
   * - huggingface-zstc-recognai-zeroshot-selectra-medium
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/Recognai/zeroshot_selectra_medium>`__
   * - huggingface-zstc-recognai-zeroshot-selectra-small
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/Recognai/zeroshot_selectra_small>`__

.. list-table:: Available Models
   :widths: 50 20 20 20 20
   :header-rows: 1
   :class: datatable

   * - Model ID
     - Fine Tunable?
     - Latest Version
     - Min SDK Version
     - Source
   * - huggingface-zstc-cross-encoder-nli-deberta-base
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/cross-encoder/nli-deberta-base>`__
   * - huggingface-zstc-cross-encoder-nli-distilroberta-base
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/cross-encoder/nli-distilroberta-base>`__
   * - huggingface-zstc-cross-encoder-nli-minilm2-l6-h768
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768>`__
   * - huggingface-zstc-cross-encoder-nli-roberta-base
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/cross-encoder/nli-roberta-base>`__
   * - huggingface-zstc-digitalepidemiologylab-covid-twitter-bert-v2-mnli
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/digitalepidemiologylab/covid-twitter-bert-v2-mnli>`__
   * - huggingface-zstc-eleldar-theme-classification
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/eleldar/theme-classification>`__
   * - huggingface-zstc-emrecan-bert-base-multilingual-cased-allnli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-multilingual-cased-allnli_tr>`__
   * - huggingface-zstc-emrecan-bert-base-multilingual-cased-multinli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-multilingual-cased-multinli_tr>`__
   * - huggingface-zstc-emrecan-bert-base-multilingual-cased-snli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-multilingual-cased-snli_tr>`__
   * - huggingface-zstc-emrecan-bert-base-turkish-cased-allnli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-turkish-cased-allnli_tr>`__
   * - huggingface-zstc-emrecan-bert-base-turkish-cased-multinli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-turkish-cased-multinli_tr>`__
   * - huggingface-zstc-emrecan-bert-base-turkish-cased-snli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-turkish-cased-snli_tr>`__
   * - huggingface-zstc-emrecan-convbert-base-turkish-mc4-cased-allnli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/convbert-base-turkish-mc4-cased-allnli_tr>`__
   * - huggingface-zstc-emrecan-convbert-base-turkish-mc4-cased-multinli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/convbert-base-turkish-mc4-cased-multinli_tr>`__
   * - huggingface-zstc-emrecan-convbert-base-turkish-mc4-cased-snli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/convbert-base-turkish-mc4-cased-snli_tr>`__
   * - huggingface-zstc-emrecan-distilbert-base-turkish-cased-allnli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/distilbert-base-turkish-cased-allnli_tr>`__
   * - huggingface-zstc-emrecan-distilbert-base-turkish-cased-multinli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/distilbert-base-turkish-cased-multinli_tr>`__
   * - huggingface-zstc-emrecan-distilbert-base-turkish-cased-snli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/distilbert-base-turkish-cased-snli_tr>`__
   * - huggingface-zstc-facebook-bart-large-mnli
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/facebook/bart-large-mnli>`__
   * - huggingface-zstc-jiva-xlm-roberta-large-it-mnli
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/Jiva/xlm-roberta-large-it-mnli>`__
   * - huggingface-zstc-lighteternal-nli-xlm-r-greek
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/lighteternal/nli-xlm-r-greek>`__
   * - huggingface-zstc-moritzlaurer-deberta-v3-large-mnli-fever-anli-ling-wanli
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli>`__
   * - huggingface-zstc-moritzlaurer-mdeberta-v3-base-xnli-multilingual-nli-2mil7
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7>`__
   * - huggingface-zstc-narsil-bart-large-mnli-opti
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/Narsil/bart-large-mnli-opti>`__
   * - huggingface-zstc-narsil-deberta-large-mnli-zero-cls
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/Narsil/deberta-large-mnli-zero-cls>`__
   * - huggingface-zstc-navteca-bart-large-mnli
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/navteca/bart-large-mnli>`__
   * - huggingface-zstc-recognai-bert-base-spanish-wwm-cased-xnli
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/Recognai/bert-base-spanish-wwm-cased-xnli>`__
   * - huggingface-zstc-recognai-zeroshot-selectra-medium
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/Recognai/zeroshot_selectra_medium>`__
   * - huggingface-zstc-recognai-zeroshot-selectra-small
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/Recognai/zeroshot_selectra_small>`__

.. list-table:: Available Models
   :widths: 50 20 20 20 20
   :header-rows: 1
   :class: datatable

   * - Model ID
     - Fine Tunable?
     - Latest Version
     - Min SDK Version
     - Source
   * - huggingface-zstc-cross-encoder-nli-deberta-base
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/cross-encoder/nli-deberta-base>`__
   * - huggingface-zstc-cross-encoder-nli-distilroberta-base
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/cross-encoder/nli-distilroberta-base>`__
   * - huggingface-zstc-cross-encoder-nli-minilm2-l6-h768
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768>`__
   * - huggingface-zstc-cross-encoder-nli-roberta-base
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/cross-encoder/nli-roberta-base>`__
   * - huggingface-zstc-digitalepidemiologylab-covid-twitter-bert-v2-mnli
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/digitalepidemiologylab/covid-twitter-bert-v2-mnli>`__
   * - huggingface-zstc-eleldar-theme-classification
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/eleldar/theme-classification>`__
   * - huggingface-zstc-emrecan-bert-base-multilingual-cased-allnli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-multilingual-cased-allnli_tr>`__
   * - huggingface-zstc-emrecan-bert-base-multilingual-cased-multinli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-multilingual-cased-multinli_tr>`__
   * - huggingface-zstc-emrecan-bert-base-multilingual-cased-snli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-multilingual-cased-snli_tr>`__
   * - huggingface-zstc-emrecan-bert-base-turkish-cased-allnli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-turkish-cased-allnli_tr>`__
   * - huggingface-zstc-emrecan-bert-base-turkish-cased-multinli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-turkish-cased-multinli_tr>`__
   * - huggingface-zstc-emrecan-bert-base-turkish-cased-snli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-turkish-cased-snli_tr>`__
   * - huggingface-zstc-emrecan-convbert-base-turkish-mc4-cased-allnli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/convbert-base-turkish-mc4-cased-allnli_tr>`__
   * - huggingface-zstc-emrecan-convbert-base-turkish-mc4-cased-multinli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/convbert-base-turkish-mc4-cased-multinli_tr>`__
   * - huggingface-zstc-emrecan-convbert-base-turkish-mc4-cased-snli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/convbert-base-turkish-mc4-cased-snli_tr>`__
   * - huggingface-zstc-emrecan-distilbert-base-turkish-cased-allnli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/distilbert-base-turkish-cased-allnli_tr>`__
   * - huggingface-zstc-emrecan-distilbert-base-turkish-cased-multinli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/distilbert-base-turkish-cased-multinli_tr>`__
   * - huggingface-zstc-emrecan-distilbert-base-turkish-cased-snli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/distilbert-base-turkish-cased-snli_tr>`__
   * - huggingface-zstc-facebook-bart-large-mnli
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/facebook/bart-large-mnli>`__
   * - huggingface-zstc-jiva-xlm-roberta-large-it-mnli
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/Jiva/xlm-roberta-large-it-mnli>`__
   * - huggingface-zstc-lighteternal-nli-xlm-r-greek
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/lighteternal/nli-xlm-r-greek>`__
   * - huggingface-zstc-moritzlaurer-deberta-v3-large-mnli-fever-anli-ling-wanli
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli>`__
   * - huggingface-zstc-moritzlaurer-mdeberta-v3-base-xnli-multilingual-nli-2mil7
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7>`__
   * - huggingface-zstc-narsil-bart-large-mnli-opti
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/Narsil/bart-large-mnli-opti>`__
   * - huggingface-zstc-narsil-deberta-large-mnli-zero-cls
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/Narsil/deberta-large-mnli-zero-cls>`__
   * - huggingface-zstc-navteca-bart-large-mnli
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/navteca/bart-large-mnli>`__
   * - huggingface-zstc-recognai-bert-base-spanish-wwm-cased-xnli
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/Recognai/bert-base-spanish-wwm-cased-xnli>`__
   * - huggingface-zstc-recognai-zeroshot-selectra-medium
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/Recognai/zeroshot_selectra_medium>`__
   * - huggingface-zstc-recognai-zeroshot-selectra-small
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/Recognai/zeroshot_selectra_small>`__

.. list-table:: Available Models
   :widths: 50 20 20 20 20
   :header-rows: 1
   :class: datatable

   * - Model ID
     - Fine Tunable?
     - Latest Version
     - Min SDK Version
     - Source
   * - huggingface-zstc-cross-encoder-nli-deberta-base
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/cross-encoder/nli-deberta-base>`__
   * - huggingface-zstc-cross-encoder-nli-distilroberta-base
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/cross-encoder/nli-distilroberta-base>`__
   * - huggingface-zstc-cross-encoder-nli-minilm2-l6-h768
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768>`__
   * - huggingface-zstc-cross-encoder-nli-roberta-base
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/cross-encoder/nli-roberta-base>`__
   * - huggingface-zstc-digitalepidemiologylab-covid-twitter-bert-v2-mnli
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/digitalepidemiologylab/covid-twitter-bert-v2-mnli>`__
   * - huggingface-zstc-eleldar-theme-classification
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/eleldar/theme-classification>`__
   * - huggingface-zstc-emrecan-bert-base-multilingual-cased-allnli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-multilingual-cased-allnli_tr>`__
   * - huggingface-zstc-emrecan-bert-base-multilingual-cased-multinli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-multilingual-cased-multinli_tr>`__
   * - huggingface-zstc-emrecan-bert-base-multilingual-cased-snli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-multilingual-cased-snli_tr>`__
   * - huggingface-zstc-emrecan-bert-base-turkish-cased-allnli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-turkish-cased-allnli_tr>`__
   * - huggingface-zstc-emrecan-bert-base-turkish-cased-multinli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-turkish-cased-multinli_tr>`__
   * - huggingface-zstc-emrecan-bert-base-turkish-cased-snli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-turkish-cased-snli_tr>`__
   * - huggingface-zstc-emrecan-convbert-base-turkish-mc4-cased-allnli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/convbert-base-turkish-mc4-cased-allnli_tr>`__
   * - huggingface-zstc-emrecan-convbert-base-turkish-mc4-cased-multinli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/convbert-base-turkish-mc4-cased-multinli_tr>`__
   * - huggingface-zstc-emrecan-convbert-base-turkish-mc4-cased-snli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/convbert-base-turkish-mc4-cased-snli_tr>`__
   * - huggingface-zstc-emrecan-distilbert-base-turkish-cased-allnli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/distilbert-base-turkish-cased-allnli_tr>`__
   * - huggingface-zstc-emrecan-distilbert-base-turkish-cased-multinli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/distilbert-base-turkish-cased-multinli_tr>`__
   * - huggingface-zstc-emrecan-distilbert-base-turkish-cased-snli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/distilbert-base-turkish-cased-snli_tr>`__
   * - huggingface-zstc-facebook-bart-large-mnli
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/facebook/bart-large-mnli>`__
   * - huggingface-zstc-jiva-xlm-roberta-large-it-mnli
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/Jiva/xlm-roberta-large-it-mnli>`__
   * - huggingface-zstc-lighteternal-nli-xlm-r-greek
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/lighteternal/nli-xlm-r-greek>`__
   * - huggingface-zstc-moritzlaurer-deberta-v3-large-mnli-fever-anli-ling-wanli
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli>`__
   * - huggingface-zstc-moritzlaurer-mdeberta-v3-base-xnli-multilingual-nli-2mil7
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7>`__
   * - huggingface-zstc-narsil-bart-large-mnli-opti
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/Narsil/bart-large-mnli-opti>`__
   * - huggingface-zstc-narsil-deberta-large-mnli-zero-cls
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/Narsil/deberta-large-mnli-zero-cls>`__
   * - huggingface-zstc-navteca-bart-large-mnli
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/navteca/bart-large-mnli>`__
   * - huggingface-zstc-recognai-bert-base-spanish-wwm-cased-xnli
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/Recognai/bert-base-spanish-wwm-cased-xnli>`__
   * - huggingface-zstc-recognai-zeroshot-selectra-medium
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/Recognai/zeroshot_selectra_medium>`__
   * - huggingface-zstc-recognai-zeroshot-selectra-small
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/Recognai/zeroshot_selectra_small>`__

.. list-table:: Available Models
   :widths: 50 20 20 20 20
   :header-rows: 1
   :class: datatable

   * - Model ID
     - Fine Tunable?
     - Latest Version
     - Min SDK Version
     - Source
   * - huggingface-zstc-cross-encoder-nli-deberta-base
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/cross-encoder/nli-deberta-base>`__
   * - huggingface-zstc-cross-encoder-nli-distilroberta-base
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/cross-encoder/nli-distilroberta-base>`__
   * - huggingface-zstc-cross-encoder-nli-minilm2-l6-h768
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768>`__
   * - huggingface-zstc-cross-encoder-nli-roberta-base
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/cross-encoder/nli-roberta-base>`__
   * - huggingface-zstc-digitalepidemiologylab-covid-twitter-bert-v2-mnli
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/digitalepidemiologylab/covid-twitter-bert-v2-mnli>`__
   * - huggingface-zstc-eleldar-theme-classification
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/eleldar/theme-classification>`__
   * - huggingface-zstc-emrecan-bert-base-multilingual-cased-allnli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-multilingual-cased-allnli_tr>`__
   * - huggingface-zstc-emrecan-bert-base-multilingual-cased-multinli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-multilingual-cased-multinli_tr>`__
   * - huggingface-zstc-emrecan-bert-base-multilingual-cased-snli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-multilingual-cased-snli_tr>`__
   * - huggingface-zstc-emrecan-bert-base-turkish-cased-allnli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-turkish-cased-allnli_tr>`__
   * - huggingface-zstc-emrecan-bert-base-turkish-cased-multinli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-turkish-cased-multinli_tr>`__
   * - huggingface-zstc-emrecan-bert-base-turkish-cased-snli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-turkish-cased-snli_tr>`__
   * - huggingface-zstc-emrecan-convbert-base-turkish-mc4-cased-allnli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/convbert-base-turkish-mc4-cased-allnli_tr>`__
   * - huggingface-zstc-emrecan-convbert-base-turkish-mc4-cased-multinli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/convbert-base-turkish-mc4-cased-multinli_tr>`__
   * - huggingface-zstc-emrecan-convbert-base-turkish-mc4-cased-snli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/convbert-base-turkish-mc4-cased-snli_tr>`__
   * - huggingface-zstc-emrecan-distilbert-base-turkish-cased-allnli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/distilbert-base-turkish-cased-allnli_tr>`__
   * - huggingface-zstc-emrecan-distilbert-base-turkish-cased-multinli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/distilbert-base-turkish-cased-multinli_tr>`__
   * - huggingface-zstc-emrecan-distilbert-base-turkish-cased-snli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/distilbert-base-turkish-cased-snli_tr>`__
   * - huggingface-zstc-facebook-bart-large-mnli
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/facebook/bart-large-mnli>`__
   * - huggingface-zstc-jiva-xlm-roberta-large-it-mnli
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/Jiva/xlm-roberta-large-it-mnli>`__
   * - huggingface-zstc-lighteternal-nli-xlm-r-greek
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/lighteternal/nli-xlm-r-greek>`__
   * - huggingface-zstc-moritzlaurer-deberta-v3-large-mnli-fever-anli-ling-wanli
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli>`__
   * - huggingface-zstc-moritzlaurer-mdeberta-v3-base-xnli-multilingual-nli-2mil7
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7>`__
   * - huggingface-zstc-narsil-bart-large-mnli-opti
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/Narsil/bart-large-mnli-opti>`__
   * - huggingface-zstc-narsil-deberta-large-mnli-zero-cls
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/Narsil/deberta-large-mnli-zero-cls>`__
   * - huggingface-zstc-navteca-bart-large-mnli
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/navteca/bart-large-mnli>`__
   * - huggingface-zstc-recognai-bert-base-spanish-wwm-cased-xnli
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/Recognai/bert-base-spanish-wwm-cased-xnli>`__
   * - huggingface-zstc-recognai-zeroshot-selectra-medium
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/Recognai/zeroshot_selectra_medium>`__
   * - huggingface-zstc-recognai-zeroshot-selectra-small
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/Recognai/zeroshot_selectra_small>`__

.. list-table:: Available Models
   :widths: 50 20 20 20 20
   :header-rows: 1
   :class: datatable

   * - Model ID
     - Fine Tunable?
     - Latest Version
     - Min SDK Version
     - Source
   * - huggingface-zstc-cross-encoder-nli-deberta-base
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/cross-encoder/nli-deberta-base>`__
   * - huggingface-zstc-cross-encoder-nli-distilroberta-base
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/cross-encoder/nli-distilroberta-base>`__
   * - huggingface-zstc-cross-encoder-nli-minilm2-l6-h768
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768>`__
   * - huggingface-zstc-cross-encoder-nli-roberta-base
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/cross-encoder/nli-roberta-base>`__
   * - huggingface-zstc-digitalepidemiologylab-covid-twitter-bert-v2-mnli
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/digitalepidemiologylab/covid-twitter-bert-v2-mnli>`__
   * - huggingface-zstc-eleldar-theme-classification
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/eleldar/theme-classification>`__
   * - huggingface-zstc-emrecan-bert-base-multilingual-cased-allnli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-multilingual-cased-allnli_tr>`__
   * - huggingface-zstc-emrecan-bert-base-multilingual-cased-multinli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-multilingual-cased-multinli_tr>`__
   * - huggingface-zstc-emrecan-bert-base-multilingual-cased-snli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-multilingual-cased-snli_tr>`__
   * - huggingface-zstc-emrecan-bert-base-turkish-cased-allnli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-turkish-cased-allnli_tr>`__
   * - huggingface-zstc-emrecan-bert-base-turkish-cased-multinli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-turkish-cased-multinli_tr>`__
   * - huggingface-zstc-emrecan-bert-base-turkish-cased-snli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/bert-base-turkish-cased-snli_tr>`__
   * - huggingface-zstc-emrecan-convbert-base-turkish-mc4-cased-allnli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/convbert-base-turkish-mc4-cased-allnli_tr>`__
   * - huggingface-zstc-emrecan-convbert-base-turkish-mc4-cased-multinli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/convbert-base-turkish-mc4-cased-multinli_tr>`__
   * - huggingface-zstc-emrecan-convbert-base-turkish-mc4-cased-snli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/convbert-base-turkish-mc4-cased-snli_tr>`__
   * - huggingface-zstc-emrecan-distilbert-base-turkish-cased-allnli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/distilbert-base-turkish-cased-allnli_tr>`__
   * - huggingface-zstc-emrecan-distilbert-base-turkish-cased-multinli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/distilbert-base-turkish-cased-multinli_tr>`__
   * - huggingface-zstc-emrecan-distilbert-base-turkish-cased-snli-tr
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/emrecan/distilbert-base-turkish-cased-snli_tr>`__
   * - huggingface-zstc-facebook-bart-large-mnli
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/facebook/bart-large-mnli>`__
   * - huggingface-zstc-jiva-xlm-roberta-large-it-mnli
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/Jiva/xlm-roberta-large-it-mnli>`__
   * - huggingface-zstc-lighteternal-nli-xlm-r-greek
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/lighteternal/nli-xlm-r-greek>`__
   * - huggingface-zstc-moritzlaurer-deberta-v3-large-mnli-fever-anli-ling-wanli
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli>`__
   * - huggingface-zstc-moritzlaurer-mdeberta-v3-base-xnli-multilingual-nli-2mil7
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7>`__
   * - huggingface-zstc-narsil-bart-large-mnli-opti
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/Narsil/bart-large-mnli-opti>`__
   * - huggingface-zstc-narsil-deberta-large-mnli-zero-cls
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/Narsil/deberta-large-mnli-zero-cls>`__
   * - huggingface-zstc-navteca-bart-large-mnli
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/navteca/bart-large-mnli>`__
   * - huggingface-zstc-recognai-bert-base-spanish-wwm-cased-xnli
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/Recognai/bert-base-spanish-wwm-cased-xnli>`__
   * - huggingface-zstc-recognai-zeroshot-selectra-medium
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/Recognai/zeroshot_selectra_medium>`__
   * - huggingface-zstc-recognai-zeroshot-selectra-small
     - False
     - 1.0.0
     - 2.81.0
     - `HuggingFace <https://huggingface.co/Recognai/zeroshot_selectra_small>`__
