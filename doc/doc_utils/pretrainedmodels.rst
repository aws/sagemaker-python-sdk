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
     - 1.0.1
     - 2.80.0
     - Classification
     - `GluonCV <https://auto.gluon.ai/stable/index.html>`__ |external-link|
   * - autogluon-regression-ensemble
     - True
     - 1.0.1
     - 2.80.0
     - Regression
     - `GluonCV <https://auto.gluon.ai/stable/index.html>`__ |external-link|
   * - catboost-classification-model
     - True
     - 1.2.4
     - 2.75.0
     - Classification
     - `Catboost <https://catboost.ai/>`__ |external-link|
   * - catboost-regression-model
     - True
     - 1.2.4
     - 2.75.0
     - Regression
     - `Catboost <https://catboost.ai/>`__ |external-link|
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
     - 1.1.0
     - 2.75.0
     - Text Summarization
     - `HuggingFace <https://huggingface.co/philschmid/bart-large-cnn-samsum>`__ |external-link|
   * - huggingface-summarization-bert-small2bert-small-finetuned-cnn-daily-mail-summarization
     - False
     - 1.1.0
     - 2.75.0
     - Text Summarization
     - `HuggingFace <https://huggingface.co/mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization>`__ |external-link|
   * - huggingface-summarization-bigbird-pegasus-large-arxiv
     - False
     - 1.1.0
     - 2.75.0
     - Text Summarization
     - `HuggingFace <https://huggingface.co/google/bigbird-pegasus-large-arxiv>`__ |external-link|
   * - huggingface-summarization-bigbird-pegasus-large-pubmed
     - False
     - 1.1.0
     - 2.75.0
     - Text Summarization
     - `HuggingFace <https://huggingface.co/google/bigbird-pegasus-large-pubmed>`__ |external-link|
   * - huggingface-summarization-distilbart-cnn-12-6
     - False
     - 1.1.0
     - 2.75.0
     - Text Summarization
     - `HuggingFace <https://huggingface.co/sshleifer/distilbart-cnn-12-6>`__ |external-link|
   * - huggingface-summarization-distilbart-cnn-6-6
     - False
     - 1.1.0
     - 2.75.0
     - Text Summarization
     - `HuggingFace <https://huggingface.co/sshleifer/distilbart-cnn-6-6>`__ |external-link|
   * - huggingface-summarization-distilbart-xsum-1-1
     - False
     - 1.1.0
     - 2.75.0
     - Text Summarization
     - `HuggingFace <https://huggingface.co/sshleifer/distilbart-xsum-1-1>`__ |external-link|
   * - huggingface-summarization-distilbart-xsum-12-3
     - False
     - 1.1.0
     - 2.75.0
     - Text Summarization
     - `HuggingFace <https://huggingface.co/sshleifer/distilbart-xsum-12-3>`__ |external-link|
   * - huggingface-textgeneration-distilgpt2
     - False
     - 1.1.0
     - 2.75.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/distilgpt2>`__ |external-link|
   * - huggingface-textgeneration-gpt2
     - False
     - 1.1.0
     - 2.75.0
     - Text Generation
     - `HuggingFace <https://huggingface.co/gpt2>`__ |external-link|
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
   * - lightgbm-classification-model
     - True
     - 1.2.3
     - 2.75.0
     - Classification
     - `LightGBM <https://lightgbm.readthedocs.io/en/latest/>`__ |external-link|
   * - lightgbm-regression-model
     - True
     - 1.2.3
     - 2.75.0
     - Regression
     - `LightGBM <https://lightgbm.readthedocs.io/en/latest/>`__ |external-link|
   * - mxnet-is-mask-rcnn-fpn-resnet101-v1d-coco
     - False
     - 1.2.0
     - 2.100.0
     - Instance Segmentation
     - `GluonCV <https://cv.gluon.ai/model_zoo/segmentation.html>`__ |external-link|
   * - mxnet-is-mask-rcnn-fpn-resnet18-v1b-coco
     - False
     - 1.2.0
     - 2.100.0
     - Instance Segmentation
     - `GluonCV <https://cv.gluon.ai/model_zoo/segmentation.html>`__ |external-link|
   * - mxnet-is-mask-rcnn-fpn-resnet50-v1b-coco
     - False
     - 1.2.0
     - 2.100.0
     - Instance Segmentation
     - `GluonCV <https://cv.gluon.ai/model_zoo/segmentation.html>`__ |external-link|
   * - mxnet-is-mask-rcnn-resnet18-v1b-coco
     - False
     - 1.2.0
     - 2.100.0
     - Instance Segmentation
     - `GluonCV <https://cv.gluon.ai/model_zoo/segmentation.html>`__ |external-link|
   * - mxnet-od-faster-rcnn-fpn-resnet101-v1d-coco
     - False
     - 1.2.0
     - 2.100.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-faster-rcnn-fpn-resnet50-v1b-coco
     - False
     - 1.2.0
     - 2.100.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-faster-rcnn-resnet101-v1d-coco
     - False
     - 1.2.0
     - 2.100.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-faster-rcnn-resnet50-v1b-coco
     - False
     - 1.2.0
     - 2.100.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-faster-rcnn-resnet50-v1b-voc
     - False
     - 1.2.0
     - 2.100.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-ssd-300-vgg16-atrous-coco
     - True
     - 1.3.0
     - 2.100.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-ssd-300-vgg16-atrous-voc
     - True
     - 1.3.0
     - 2.100.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-ssd-512-mobilenet1-0-coco
     - True
     - 1.3.0
     - 2.100.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-ssd-512-mobilenet1-0-voc
     - True
     - 1.3.0
     - 2.100.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-ssd-512-resnet50-v1-coco
     - True
     - 1.3.0
     - 2.100.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-ssd-512-resnet50-v1-voc
     - True
     - 1.3.0
     - 2.100.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-ssd-512-vgg16-atrous-coco
     - True
     - 1.3.0
     - 2.100.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-ssd-512-vgg16-atrous-voc
     - True
     - 1.3.0
     - 2.100.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-yolo3-darknet53-coco
     - False
     - 1.2.0
     - 2.100.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-yolo3-darknet53-voc
     - False
     - 1.2.0
     - 2.100.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-yolo3-mobilenet1-0-coco
     - False
     - 1.2.0
     - 2.100.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-od-yolo3-mobilenet1-0-voc
     - False
     - 1.2.0
     - 2.100.0
     - Object Detection
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__ |external-link|
   * - mxnet-semseg-fcn-resnet101-ade
     - True
     - 1.4.0
     - 2.100.0
     - Semantic Segmentation
     - `GluonCV <https://cv.gluon.ai/model_zoo/segmentation.html>`__ |external-link|
   * - mxnet-semseg-fcn-resnet101-coco
     - True
     - 1.4.0
     - 2.100.0
     - Semantic Segmentation
     - `GluonCV <https://cv.gluon.ai/model_zoo/segmentation.html>`__ |external-link|
   * - mxnet-semseg-fcn-resnet101-voc
     - True
     - 1.4.0
     - 2.100.0
     - Semantic Segmentation
     - `GluonCV <https://cv.gluon.ai/model_zoo/segmentation.html>`__ |external-link|
   * - mxnet-semseg-fcn-resnet50-ade
     - True
     - 1.4.0
     - 2.100.0
     - Semantic Segmentation
     - `GluonCV <https://cv.gluon.ai/model_zoo/segmentation.html>`__ |external-link|
   * - mxnet-tcembedding-robertafin-base-uncased
     - False
     - 1.2.0
     - 2.100.0
     - Text Embedding
     - `GluonCV <https://nlp.gluon.ai/master/_modules/gluonnlp/models/roberta.html>`__ |external-link|
   * - mxnet-tcembedding-robertafin-base-wiki-uncased
     - False
     - 1.2.0
     - 2.100.0
     - Text Embedding
     - `GluonCV <https://nlp.gluon.ai/master/_modules/gluonnlp/models/roberta.html>`__ |external-link|
   * - mxnet-tcembedding-robertafin-large-uncased
     - False
     - 1.2.0
     - 2.100.0
     - Text Embedding
     - `GluonCV <https://nlp.gluon.ai/master/_modules/gluonnlp/models/roberta.html>`__ |external-link|
   * - mxnet-tcembedding-robertafin-large-wiki-uncased
     - False
     - 1.2.0
     - 2.100.0
     - Text Embedding
     - `GluonCV <https://nlp.gluon.ai/master/_modules/gluonnlp/models/roberta.html>`__ |external-link|
   * - pytorch-eqa-bert-base-cased
     - True
     - 1.2.0
     - 2.75.0
     - Question Answering
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__ |external-link|
   * - pytorch-eqa-bert-base-multilingual-cased
     - True
     - 1.2.0
     - 2.75.0
     - Question Answering
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__ |external-link|
   * - pytorch-eqa-bert-base-multilingual-uncased
     - True
     - 1.2.0
     - 2.75.0
     - Question Answering
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__ |external-link|
   * - pytorch-eqa-bert-base-uncased
     - True
     - 1.2.0
     - 2.75.0
     - Question Answering
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__ |external-link|
   * - pytorch-eqa-bert-large-cased
     - True
     - 1.2.0
     - 2.75.0
     - Question Answering
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__ |external-link|
   * - pytorch-eqa-bert-large-cased-whole-word-masking
     - True
     - 1.2.0
     - 2.75.0
     - Question Answering
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__ |external-link|
   * - pytorch-eqa-bert-large-cased-whole-word-masking-finetuned-squad
     - True
     - 1.2.0
     - 2.75.0
     - Question Answering
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__ |external-link|
   * - pytorch-eqa-bert-large-uncased
     - True
     - 1.2.0
     - 2.75.0
     - Question Answering
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__ |external-link|
   * - pytorch-eqa-bert-large-uncased-whole-word-masking
     - True
     - 1.2.0
     - 2.75.0
     - Question Answering
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__ |external-link|
   * - pytorch-eqa-bert-large-uncased-whole-word-masking-finetuned-squad
     - True
     - 1.2.0
     - 2.75.0
     - Question Answering
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__ |external-link|
   * - pytorch-eqa-distilbert-base-cased
     - True
     - 1.2.0
     - 2.75.0
     - Question Answering
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__ |external-link|
   * - pytorch-eqa-distilbert-base-multilingual-cased
     - True
     - 1.2.0
     - 2.75.0
     - Question Answering
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__ |external-link|
   * - pytorch-eqa-distilbert-base-uncased
     - True
     - 1.2.0
     - 2.75.0
     - Question Answering
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__ |external-link|
   * - pytorch-eqa-distilroberta-base
     - True
     - 1.2.0
     - 2.75.0
     - Question Answering
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__ |external-link|
   * - pytorch-eqa-roberta-base
     - True
     - 1.2.0
     - 2.75.0
     - Question Answering
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__ |external-link|
   * - pytorch-eqa-roberta-base-openai-detector
     - True
     - 1.2.0
     - 2.75.0
     - Question Answering
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__ |external-link|
   * - pytorch-eqa-roberta-large
     - True
     - 1.2.0
     - 2.75.0
     - Question Answering
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__ |external-link|
   * - pytorch-eqa-roberta-large-openai-detector
     - True
     - 1.2.0
     - 2.75.0
     - Question Answering
     - `Pytorch Hub <https://pytorch.org/hub/huggingface_pytorch-transformers/>`__ |external-link|
   * - pytorch-ic-alexnet
     - True
     - 2.2.3
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_alexnet/>`__ |external-link|
   * - pytorch-ic-densenet121
     - True
     - 2.2.3
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_densenet/>`__ |external-link|
   * - pytorch-ic-densenet161
     - True
     - 2.2.3
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_densenet/>`__ |external-link|
   * - pytorch-ic-densenet169
     - True
     - 2.2.3
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_densenet/>`__ |external-link|
   * - pytorch-ic-densenet201
     - True
     - 2.2.3
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_densenet/>`__ |external-link|
   * - pytorch-ic-googlenet
     - True
     - 2.2.3
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_googlenet/>`__ |external-link|
   * - pytorch-ic-mobilenet-v2
     - True
     - 2.2.3
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_mobilenet_v2/>`__ |external-link|
   * - pytorch-ic-resnet101
     - True
     - 2.2.3
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_resnet/>`__ |external-link|
   * - pytorch-ic-resnet152
     - True
     - 2.2.3
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_resnet/>`__ |external-link|
   * - pytorch-ic-resnet18
     - True
     - 2.2.3
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_resnet/>`__ |external-link|
   * - pytorch-ic-resnet34
     - True
     - 2.2.3
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_resnet/>`__ |external-link|
   * - pytorch-ic-resnet50
     - True
     - 2.2.3
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_resnet/>`__ |external-link|
   * - pytorch-ic-resnext101-32x8d
     - True
     - 2.2.3
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_resnext/>`__ |external-link|
   * - pytorch-ic-resnext50-32x4d
     - True
     - 2.2.3
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_resnext/>`__ |external-link|
   * - pytorch-ic-shufflenet-v2-x1-0
     - True
     - 2.2.3
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_shufflenet_v2/>`__ |external-link|
   * - pytorch-ic-squeezenet1-0
     - True
     - 2.2.3
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_squeezenet/>`__ |external-link|
   * - pytorch-ic-squeezenet1-1
     - True
     - 2.2.3
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_squeezenet/>`__ |external-link|
   * - pytorch-ic-vgg11
     - True
     - 2.2.3
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_vgg/>`__ |external-link|
   * - pytorch-ic-vgg11-bn
     - True
     - 2.2.3
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_vgg/>`__ |external-link|
   * - pytorch-ic-vgg13
     - True
     - 2.2.3
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_vgg/>`__ |external-link|
   * - pytorch-ic-vgg13-bn
     - True
     - 2.2.3
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_vgg/>`__ |external-link|
   * - pytorch-ic-vgg16
     - True
     - 2.2.3
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_vgg/>`__ |external-link|
   * - pytorch-ic-vgg16-bn
     - True
     - 2.2.3
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_vgg/>`__ |external-link|
   * - pytorch-ic-vgg19
     - True
     - 2.2.3
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_vgg/>`__ |external-link|
   * - pytorch-ic-vgg19-bn
     - True
     - 2.2.3
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_vgg/>`__ |external-link|
   * - pytorch-ic-wide-resnet101-2
     - True
     - 2.2.3
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_wide_resnet/>`__ |external-link|
   * - pytorch-ic-wide-resnet50-2
     - True
     - 2.2.3
     - 2.75.0
     - Image Classification
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_wide_resnet/>`__ |external-link|
   * - pytorch-od-nvidia-ssd
     - False
     - 1.0.1
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
     - 1.0.1
     - 2.75.0
     - Source
     - `Source <https://arxiv.org/abs/2012.06678>`__ |external-link|
   * - pytorch-tabtransformerregression-model
     - True
     - 1.0.0
     - 2.75.0
     - Source
     - `Source <https://arxiv.org/abs/2012.06678>`__ |external-link|
   * - sklearn-classification-linear
     - True
     - 1.1.1
     - 2.75.0
     - Classification
     - `ScikitLearn <https://scikit-learn.org/stable/>`__ |external-link|
   * - sklearn-regression-linear
     - True
     - 1.1.1
     - 2.75.0
     - Regression
     - `ScikitLearn <https://scikit-learn.org/stable/>`__ |external-link|
   * - tensorflow-ic-bit-m-r101x1-ilsvrc2012-classification-1
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r101x1/ilsvrc2012_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-m-r101x1-imagenet21k-classification-1
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r101x1/imagenet21k_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-m-r101x3-ilsvrc2012-classification-1
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r101x3/ilsvrc2012_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-m-r101x3-imagenet21k-classification-1
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r101x3/imagenet21k_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-m-r50x1-ilsvrc2012-classification-1
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r50x1/ilsvrc2012_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-m-r50x1-imagenet21k-classification-1
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r50x1/imagenet21k_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-m-r50x3-ilsvrc2012-classification-1
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r50x3/ilsvrc2012_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-m-r50x3-imagenet21k-classification-1
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r50x3/imagenet21k_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-s-r101x1-ilsvrc2012-classification-1
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r101x1/ilsvrc2012_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-s-r101x3-ilsvrc2012-classification-1
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r101x3/ilsvrc2012_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-s-r50x1-ilsvrc2012-classification-1
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r50x1/ilsvrc2012_classification/1>`__ |external-link|
   * - tensorflow-ic-bit-s-r50x3-ilsvrc2012-classification-1
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r50x3/ilsvrc2012_classification/1>`__ |external-link|
   * - tensorflow-ic-efficientnet-b0-classification-1
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b0/classification/1>`__ |external-link|
   * - tensorflow-ic-efficientnet-b1-classification-1
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b1/classification/1>`__ |external-link|
   * - tensorflow-ic-efficientnet-b2-classification-1
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b2/classification/1>`__ |external-link|
   * - tensorflow-ic-efficientnet-b3-classification-1
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b3/classification/1>`__ |external-link|
   * - tensorflow-ic-efficientnet-b4-classification-1
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b4/classification/1>`__ |external-link|
   * - tensorflow-ic-efficientnet-b5-classification-1
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b5/classification/1>`__ |external-link|
   * - tensorflow-ic-efficientnet-b6-classification-1
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b6/classification/1>`__ |external-link|
   * - tensorflow-ic-efficientnet-b7-classification-1
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b7/classification/1>`__ |external-link|
   * - tensorflow-ic-efficientnet-lite0-classification-2
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite0/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-lite1-classification-2
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite1/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-lite2-classification-2
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite2/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-lite3-classification-2
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite3/classification/2>`__ |external-link|
   * - tensorflow-ic-efficientnet-lite4-classification-2
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite4/classification/2>`__ |external-link|
   * - tensorflow-ic-imagenet-inception-resnet-v2-classification-4
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/4>`__ |external-link|
   * - tensorflow-ic-imagenet-inception-v1-classification-4
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/inception_v1/classification/4>`__ |external-link|
   * - tensorflow-ic-imagenet-inception-v2-classification-4
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/inception_v2/classification/4>`__ |external-link|
   * - tensorflow-ic-imagenet-inception-v3-classification-4
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/inception_v3/classification/4>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-025-128-classification-4
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_128/classification/4>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-025-160-classification-4
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_160/classification/4>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-025-192-classification-4
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_192/classification/4>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-025-224-classification-4
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/classification/4>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-050-128-classification-4
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_128/classification/4>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-050-160-classification-4
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/classification/4>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-050-192-classification-4
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_192/classification/4>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-050-224-classification-4
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_224/classification/4>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-075-128-classification-4
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_128/classification/4>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-075-160-classification-4
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_160/classification/4>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-075-192-classification-4
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_192/classification/4>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-075-224-classification-4
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_224/classification/4>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-100-128-classification-4
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_128/classification/4>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-100-160-classification-4
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_160/classification/4>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-100-192-classification-4
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_192/classification/4>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v1-100-224-classification-4
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/4>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-035-224-classification-4
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_035_224/classification/4>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-050-224-classification-4
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_050_224/classification/4>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-075-224-classification-4
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_075_224/classification/4>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-100-224-classification-4
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-130-224-classification-4
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4>`__ |external-link|
   * - tensorflow-ic-imagenet-mobilenet-v2-140-224-classification-4
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/4>`__ |external-link|
   * - tensorflow-ic-imagenet-resnet-v1-101-classification-4
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v1_101/classification/4>`__ |external-link|
   * - tensorflow-ic-imagenet-resnet-v1-152-classification-4
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v1_152/classification/4>`__ |external-link|
   * - tensorflow-ic-imagenet-resnet-v1-50-classification-4
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v1_50/classification/4>`__ |external-link|
   * - tensorflow-ic-imagenet-resnet-v2-101-classification-4
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v2_101/classification/4>`__ |external-link|
   * - tensorflow-ic-imagenet-resnet-v2-152-classification-4
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v2_152/classification/4>`__ |external-link|
   * - tensorflow-ic-imagenet-resnet-v2-50-classification-4
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v2_50/classification/4>`__ |external-link|
   * - tensorflow-ic-resnet-50-classification-1
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/resnet_50/classification/1>`__ |external-link|
   * - tensorflow-ic-tf2-preview-inception-v3-classification-4
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/tf2-preview/inception_v3/classification/4>`__ |external-link|
   * - tensorflow-ic-tf2-preview-mobilenet-v2-classification-4
     - True
     - 2.0.1
     - 2.80.0
     - Image Classification
     - `Tensorflow Hub <https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4>`__ |external-link|
   * - tensorflow-icembedding-bit-m-r101x1-ilsvrc2012-featurevector-1
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r101x1/1>`__ |external-link|
   * - tensorflow-icembedding-bit-m-r101x3-imagenet21k-featurevector-1
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r101x3/1>`__ |external-link|
   * - tensorflow-icembedding-bit-m-r50x1-ilsvrc2012-featurevector-1
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r50x1/1>`__ |external-link|
   * - tensorflow-icembedding-bit-m-r50x3-imagenet21k-featurevector-1
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r50x3/1>`__ |external-link|
   * - tensorflow-icembedding-bit-s-r101x1-ilsvrc2012-featurevector-1
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r101x1/1>`__ |external-link|
   * - tensorflow-icembedding-bit-s-r101x3-ilsvrc2012-featurevector-1
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r101x3/1>`__ |external-link|
   * - tensorflow-icembedding-bit-s-r50x1-ilsvrc2012-featurevector-1
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r50x1/1>`__ |external-link|
   * - tensorflow-icembedding-bit-s-r50x3-ilsvrc2012-featurevector-1
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r50x3/1>`__ |external-link|
   * - tensorflow-icembedding-efficientnet-b0-featurevector-1
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b0/feature-vector/1>`__ |external-link|
   * - tensorflow-icembedding-efficientnet-b1-featurevector-1
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b1/feature-vector/1>`__ |external-link|
   * - tensorflow-icembedding-efficientnet-b2-featurevector-1
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b2/feature-vector/1>`__ |external-link|
   * - tensorflow-icembedding-efficientnet-b3-featurevector-1
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b3/feature-vector/1>`__ |external-link|
   * - tensorflow-icembedding-efficientnet-b6-featurevector-1
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b6/feature-vector/1>`__ |external-link|
   * - tensorflow-icembedding-efficientnet-lite0-featurevector-2
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2>`__ |external-link|
   * - tensorflow-icembedding-efficientnet-lite1-featurevector-2
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite1/feature-vector/2>`__ |external-link|
   * - tensorflow-icembedding-efficientnet-lite2-featurevector-2
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite2/feature-vector/2>`__ |external-link|
   * - tensorflow-icembedding-efficientnet-lite3-featurevector-2
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite3/feature-vector/2>`__ |external-link|
   * - tensorflow-icembedding-efficientnet-lite4-featurevector-2
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite4/feature-vector/2>`__ |external-link|
   * - tensorflow-icembedding-imagenet-inception-v1-featurevector-4
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/inception_v1/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-inception-v2-featurevector-4
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/inception_v2/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-inception-v3-featurevector-4
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-025-128-featurevector-4
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_128/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-025-160-featurevector-4
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_160/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-025-192-featurevector-4
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_192/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-025-224-featurevector-4
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-050-128-featurevector-4
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_128/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-050-160-featurevector-4
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-050-192-featurevector-4
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_192/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-050-224-featurevector-4
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-075-128-featurevector-4
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_128/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-075-160-featurevector-4
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_160/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-075-192-featurevector-4
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_192/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-075-224-featurevector-4
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-100-128-featurevector-4
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_128/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-100-160-featurevector-4
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_160/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-100-192-featurevector-4
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_192/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v1-100-224-featurevector-4
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v2-035-224-featurevector-4
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_035_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v2-050-224-featurevector-4
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_050_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v2-075-224-featurevector-4
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_075_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v2-100-224-featurevector-4
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v2-130-224-featurevector-4
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-mobilenet-v2-140-224-featurevector-4
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-resnet-v1-101-featurevector-4
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v1_101/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-resnet-v1-152-featurevector-4
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v1_152/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-resnet-v1-50-featurevector-4
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-resnet-v2-101-featurevector-4
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-resnet-v2-152-featurevector-4
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-imagenet-resnet-v2-50-featurevector-4
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-resnet-50-featurevector-1
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/resnet_50/feature_vector/1>`__ |external-link|
   * - tensorflow-icembedding-tf2-preview-inception-v3-featurevector-4
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4>`__ |external-link|
   * - tensorflow-icembedding-tf2-preview-mobilenet-v2-featurevector-4
     - False
     - 2.0.0
     - 2.80.0
     - Image Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4>`__ |external-link|
   * - tensorflow-od-centernet-hourglass-1024x1024-1
     - False
     - 2.0.0
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/hourglass_1024x1024/1>`__ |external-link|
   * - tensorflow-od-centernet-hourglass-1024x1024-kpts-1
     - False
     - 2.0.0
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/hourglass_1024x1024_kpts/1>`__ |external-link|
   * - tensorflow-od-centernet-hourglass-512x512-1
     - False
     - 2.0.0
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1>`__ |external-link|
   * - tensorflow-od-centernet-hourglass-512x512-kpts-1
     - False
     - 2.0.0
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/hourglass_512x512_kpts/1>`__ |external-link|
   * - tensorflow-od-centernet-resnet101v1-fpn-512x512-1
     - False
     - 2.0.0
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/resnet101v1_fpn_512x512/1>`__ |external-link|
   * - tensorflow-od-centernet-resnet50v1-fpn-512x512-1
     - False
     - 2.0.0
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512/1>`__ |external-link|
   * - tensorflow-od-centernet-resnet50v1-fpn-512x512-kpts-1
     - False
     - 2.0.0
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512_kpts/1>`__ |external-link|
   * - tensorflow-od-centernet-resnet50v2-512x512-1
     - False
     - 2.0.0
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/resnet50v2_512x512/1>`__ |external-link|
   * - tensorflow-od-centernet-resnet50v2-512x512-kpts-1
     - False
     - 2.0.0
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/resnet50v2_512x512_kpts/1>`__ |external-link|
   * - tensorflow-od-efficientdet-d0-1
     - False
     - 2.0.0
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientdet/d0/1>`__ |external-link|
   * - tensorflow-od-efficientdet-d1-1
     - False
     - 2.0.0
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientdet/d1/1>`__ |external-link|
   * - tensorflow-od-efficientdet-d2-1
     - False
     - 2.0.0
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientdet/d2/1>`__ |external-link|
   * - tensorflow-od-efficientdet-d3-1
     - False
     - 2.0.0
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientdet/d3/1>`__ |external-link|
   * - tensorflow-od-efficientdet-d4-1
     - False
     - 2.0.0
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientdet/d4/1>`__ |external-link|
   * - tensorflow-od-efficientdet-d5-1
     - False
     - 2.0.0
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientdet/d5/1>`__ |external-link|
   * - tensorflow-od-faster-rcnn-inception-resnet-v2-1024x1024-1
     - False
     - 2.0.0
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_1024x1024/1>`__ |external-link|
   * - tensorflow-od-faster-rcnn-inception-resnet-v2-640x640-1
     - False
     - 2.0.0
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1>`__ |external-link|
   * - tensorflow-od-faster-rcnn-resnet101-v1-1024x1024-1
     - False
     - 2.0.0
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_1024x1024/1>`__ |external-link|
   * - tensorflow-od-faster-rcnn-resnet101-v1-640x640-1
     - False
     - 2.0.0
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_640x640/1>`__ |external-link|
   * - tensorflow-od-faster-rcnn-resnet101-v1-800x1333-1
     - False
     - 2.0.0
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_800x1333/1>`__ |external-link|
   * - tensorflow-od-faster-rcnn-resnet152-v1-1024x1024-1
     - False
     - 2.0.0
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_1024x1024/1>`__ |external-link|
   * - tensorflow-od-faster-rcnn-resnet152-v1-640x640-1
     - False
     - 2.0.0
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_640x640/1>`__ |external-link|
   * - tensorflow-od-faster-rcnn-resnet152-v1-800x1333-1
     - False
     - 2.0.0
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_800x1333/1>`__ |external-link|
   * - tensorflow-od-faster-rcnn-resnet50-v1-1024x1024-1
     - False
     - 2.0.0
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_1024x1024/1>`__ |external-link|
   * - tensorflow-od-faster-rcnn-resnet50-v1-640x640-1
     - False
     - 2.0.0
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1>`__ |external-link|
   * - tensorflow-od-faster-rcnn-resnet50-v1-800x1333-1
     - False
     - 2.0.0
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_800x1333/1>`__ |external-link|
   * - tensorflow-od-retinanet-resnet101-v1-fpn-1024x1024-1
     - False
     - 2.0.0
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/retinanet/resnet101_v1_fpn_1024x1024/1>`__ |external-link|
   * - tensorflow-od-retinanet-resnet101-v1-fpn-640x640-1
     - False
     - 2.0.0
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/retinanet/resnet101_v1_fpn_640x640/1>`__ |external-link|
   * - tensorflow-od-retinanet-resnet152-v1-fpn-1024x1024-1
     - False
     - 2.0.0
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/retinanet/resnet152_v1_fpn_1024x1024/1>`__ |external-link|
   * - tensorflow-od-retinanet-resnet152-v1-fpn-640x640-1
     - False
     - 2.0.0
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/retinanet/resnet152_v1_fpn_640x640/1>`__ |external-link|
   * - tensorflow-od-retinanet-resnet50-v1-fpn-1024x1024-1
     - False
     - 2.0.0
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/retinanet/resnet50_v1_fpn_1024x1024/1>`__ |external-link|
   * - tensorflow-od-retinanet-resnet50-v1-fpn-640x640-1
     - False
     - 2.0.0
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/retinanet/resnet50_v1_fpn_640x640/1>`__ |external-link|
   * - tensorflow-od-ssd-mobilenet-v1-fpn-640x640-1
     - False
     - 2.0.0
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/ssd_mobilenet_v1/fpn_640x640/1>`__ |external-link|
   * - tensorflow-od-ssd-mobilenet-v2-2
     - False
     - 2.0.0
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2>`__ |external-link|
   * - tensorflow-od-ssd-mobilenet-v2-fpnlite-320x320-1
     - False
     - 2.0.0
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1>`__ |external-link|
   * - tensorflow-od-ssd-mobilenet-v2-fpnlite-640x640-1
     - False
     - 2.0.0
     - 2.80.0
     - Object Detection
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_640x640/1>`__ |external-link|
   * - tensorflow-spc-bert-en-cased-L-12-H-768-A-12-2
     - True
     - 1.2.2
     - 2.75.0
     - Sentence Pair Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/2>`__ |external-link|
   * - tensorflow-spc-bert-en-uncased-L-12-H-768-A-12-2
     - True
     - 1.2.2
     - 2.75.0
     - Sentence Pair Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2>`__ |external-link|
   * - tensorflow-spc-bert-en-uncased-L-24-H-1024-A-16-2
     - True
     - 1.2.2
     - 2.75.0
     - Sentence Pair Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/2>`__ |external-link|
   * - tensorflow-spc-bert-en-wwm-cased-L-24-H-1024-A-16-2
     - True
     - 1.2.2
     - 2.75.0
     - Sentence Pair Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_wwm_cased_L-24_H-1024_A-16/2>`__ |external-link|
   * - tensorflow-spc-bert-en-wwm-uncased-L-24-H-1024-A-16-2
     - True
     - 1.2.2
     - 2.75.0
     - Sentence Pair Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_wwm_uncased_L-24_H-1024_A-16/2>`__ |external-link|
   * - tensorflow-spc-bert-multi-cased-L-12-H-768-A-12-2
     - True
     - 1.2.2
     - 2.75.0
     - Sentence Pair Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/2>`__ |external-link|
   * - tensorflow-spc-electra-base-1
     - True
     - 1.2.2
     - 2.75.0
     - Sentence Pair Classification
     - `Tensorflow Hub <https://tfhub.dev/google/electra_base/1>`__ |external-link|
   * - tensorflow-spc-electra-small-1
     - True
     - 1.2.2
     - 2.75.0
     - Sentence Pair Classification
     - `Tensorflow Hub <https://tfhub.dev/google/electra_small/1>`__ |external-link|
   * - tensorflow-spc-experts-bert-pubmed-1
     - True
     - 1.2.2
     - 2.75.0
     - Sentence Pair Classification
     - `Tensorflow Hub <https://tfhub.dev/google/experts/bert/pubmed/1>`__ |external-link|
   * - tensorflow-spc-experts-bert-wiki-books-1
     - True
     - 1.2.2
     - 2.75.0
     - Sentence Pair Classification
     - `Tensorflow Hub <https://tfhub.dev/google/experts/bert/wiki_books/1>`__ |external-link|
   * - tensorflow-tc-bert-en-cased-L-12-H-768-A-12-2
     - True
     - 1.1.2
     - 2.75.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/2>`__ |external-link|
   * - tensorflow-tc-bert-en-cased-L-24-H-1024-A-16-2
     - True
     - 1.1.2
     - 2.75.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_cased_L-24_H-1024_A-16/2>`__ |external-link|
   * - tensorflow-tc-bert-en-uncased-L-12-H-768-A-12-2
     - True
     - 1.1.2
     - 2.75.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2>`__ |external-link|
   * - tensorflow-tc-bert-en-wwm-cased-L-24-H-1024-A-16-2
     - True
     - 1.1.2
     - 2.75.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_wwm_cased_L-24_H-1024_A-16/2>`__ |external-link|
   * - tensorflow-tc-bert-en-wwm-uncased-L-24-H-1024-A-16-2
     - True
     - 1.1.2
     - 2.75.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_wwm_uncased_L-24_H-1024_A-16/2>`__ |external-link|
   * - tensorflow-tc-bert-multi-cased-L-12-H-768-A-12-2
     - True
     - 1.1.2
     - 2.75.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/2>`__ |external-link|
   * - tensorflow-tc-electra-base-1
     - True
     - 1.1.2
     - 2.75.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/google/electra_base/1>`__ |external-link|
   * - tensorflow-tc-electra-small-1
     - True
     - 1.1.2
     - 2.75.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/google/electra_small/1>`__ |external-link|
   * - tensorflow-tc-experts-bert-pubmed-1
     - True
     - 1.1.2
     - 2.75.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/google/experts/bert/pubmed/1>`__ |external-link|
   * - tensorflow-tc-experts-bert-wiki-books-1
     - True
     - 1.1.2
     - 2.75.0
     - Text Classification
     - `Tensorflow Hub <https://tfhub.dev/google/experts/bert/wiki_books/1>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-10-H-128-A-2-2
     - False
     - 1.1.0
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-10-H-256-A-4-2
     - False
     - 1.1.0
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-10-H-512-A-8-2
     - False
     - 1.1.0
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-10-H-768-A-12-2
     - False
     - 1.1.0
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-12-H-128-A-2-2
     - False
     - 1.1.0
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-12-H-256-A-4
     - False
     - 1.1.0
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-12-H-512-A-8-2
     - False
     - 1.1.0
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-12-H-768-A-12-2
     - False
     - 1.1.0
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-12-H-768-A-12-4
     - False
     - 1.1.0
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-2-H-128-A-2-2
     - False
     - 1.1.0
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-2-H-256-A-4
     - False
     - 1.1.0
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-2-H-512-A-8-2
     - False
     - 1.1.0
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-2-H-768-A-12-2
     - False
     - 1.1.0
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-4-H-128-A-2-2
     - False
     - 1.1.0
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-4-H-256-A-4-2
     - False
     - 1.1.0
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-4-H-512-A-8-2
     - False
     - 1.1.0
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-4-H-768-A-12-2
     - False
     - 1.1.0
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-6-H-128-A-2-2
     - False
     - 1.1.0
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-6-H-256-A-4
     - False
     - 1.1.0
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-6-H-512-A-8-2
     - False
     - 1.1.0
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-6-H-768-A-12-2
     - False
     - 1.1.0
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-8-H-256-A-4-2
     - False
     - 1.1.0
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-8-H-512-A-8-2
     - False
     - 1.1.0
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-en-uncased-L-8-H-768-A-12-2
     - False
     - 1.1.0
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-wiki-books-mnli-2
     - False
     - 1.1.0
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/experts/bert/wiki_books/mnli/2>`__ |external-link|
   * - tensorflow-tcembedding-bert-wiki-books-sst2
     - False
     - 1.1.0
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/experts/bert/wiki_books/sst2/2>`__ |external-link|
   * - tensorflow-tcembedding-talkheads-ggelu-bert-en-base-2
     - False
     - 1.1.0
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/2>`__ |external-link|
   * - tensorflow-tcembedding-talkheads-ggelu-bert-en-large-2
     - False
     - 1.1.0
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_large/2>`__ |external-link|
   * - tensorflow-tcembedding-universal-sentence-encoder-cmlm-en-base-1
     - False
     - 1.1.0
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/1>`__ |external-link|
   * - tensorflow-tcembedding-universal-sentence-encoder-cmlm-en-large-1
     - False
     - 1.1.0
     - 2.75.0
     - Text Embedding
     - `Tensorflow Hub <https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-large/1>`__ |external-link|
   * - xgboost-classification-model
     - True
     - 1.2.1
     - 2.75.0
     - Classification
     - `XGBoost <https://xgboost.readthedocs.io/en/release_1.3.0/>`__ |external-link|
   * - xgboost-regression-model
     - True
     - 1.2.1
     - 2.75.0
     - Regression
     - `XGBoost <https://xgboost.readthedocs.io/en/release_1.3.0/>`__ |external-link|
