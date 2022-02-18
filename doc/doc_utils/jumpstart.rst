==================================
JumpStart Available Model Table
==================================

    JumpStart for the SageMaker Python SDK uses model ids and model versions to access the necessary
    utilities. This table serves to provide the core material plus some extra information that can be useful
    in selecting the correct model id and corresponding parameters.

    
    If you want to automatically use the latest version of the model, use "*" for the `model_version` attribute.
    We highly suggest pinning an exact model version however.

    
.. list-table:: Available Models
   :widths: 50 20 20 20
   :header-rows: 1
   :class: datatable

   * - Model ID
     - Fine Tunable?
     - Latest Version
     - Min SDK Version
   * - catboost-classification-model
     - True
     - 1.0.0
     - 2.68.1
   * - catboost-regression-model
     - True
     - 1.0.0
     - 2.68.1
   * - huggingface-ner-distilbert-base-cased-finetuned-conll03-english
     - False
     - 1.0.0
     - 2.68.1
   * - huggingface-ner-distilbert-base-uncased-finetuned-conll03-english
     - False
     - 1.0.0
     - 2.68.1
   * - huggingface-spc-bert-base-cased
     - True
     - 1.0.0
     - 2.68.1
   * - huggingface-spc-bert-base-multilingual-cased
     - True
     - 1.0.0
     - 2.68.1
   * - huggingface-spc-bert-base-multilingual-uncased
     - True
     - 1.0.0
     - 2.68.1
   * - huggingface-spc-bert-base-uncased
     - True
     - 1.0.0
     - 2.68.1
   * - huggingface-spc-bert-large-cased
     - True
     - 1.0.0
     - 2.68.1
   * - huggingface-spc-bert-large-cased-whole-word-masking
     - True
     - 1.0.0
     - 2.68.1
   * - huggingface-spc-bert-large-uncased
     - True
     - 1.0.0
     - 2.68.1
   * - huggingface-spc-bert-large-uncased-whole-word-masking
     - True
     - 1.0.0
     - 2.68.1
   * - huggingface-spc-distilbert-base-cased
     - True
     - 1.0.0
     - 2.68.1
   * - huggingface-spc-distilbert-base-multilingual-cased
     - True
     - 1.0.0
     - 2.68.1
   * - huggingface-spc-distilbert-base-uncased
     - True
     - 1.0.0
     - 2.68.1
   * - huggingface-spc-distilroberta-base
     - True
     - 1.0.0
     - 2.68.1
   * - huggingface-spc-roberta-base
     - True
     - 1.0.0
     - 2.68.1
   * - huggingface-spc-roberta-base-openai-detector
     - True
     - 1.0.0
     - 2.68.1
   * - huggingface-spc-roberta-large
     - True
     - 1.0.0
     - 2.68.1
   * - huggingface-spc-roberta-large-openai-detector
     - True
     - 1.0.0
     - 2.68.1
   * - huggingface-spc-xlm-clm-ende-1024
     - True
     - 1.0.0
     - 2.68.1
   * - huggingface-spc-xlm-mlm-ende-1024
     - True
     - 1.0.0
     - 2.68.1
   * - huggingface-spc-xlm-mlm-enro-1024
     - True
     - 1.0.0
     - 2.68.1
   * - huggingface-spc-xlm-mlm-tlm-xnli15-1024
     - True
     - 1.0.0
     - 2.68.1
   * - huggingface-spc-xlm-mlm-xnli15-1024
     - True
     - 1.0.0
     - 2.68.1
   * - huggingface-summarization-bart-large-cnn-samsum
     - False
     - 1.0.0
     - 2.68.1
   * - huggingface-summarization-bert-small2bert-small-finetuned-cnn-daily-mail-summarization
     - False
     - 1.0.0
     - 2.68.1
   * - huggingface-summarization-bigbird-pegasus-large-arxiv
     - False
     - 1.0.0
     - 2.68.1
   * - huggingface-summarization-bigbird-pegasus-large-pubmed
     - False
     - 1.0.0
     - 2.68.1
   * - huggingface-summarization-distilbart-cnn-12-6
     - False
     - 1.0.0
     - 2.68.1
   * - huggingface-summarization-distilbart-cnn-6-6
     - False
     - 1.0.0
     - 2.68.1
   * - huggingface-summarization-distilbart-xsum-1-1
     - False
     - 1.0.0
     - 2.68.1
   * - huggingface-summarization-distilbart-xsum-12-3
     - False
     - 1.0.0
     - 2.68.1
   * - huggingface-textgeneration-distilgpt2
     - False
     - 1.0.0
     - 2.68.1
   * - huggingface-textgeneration-gpt2
     - False
     - 1.0.0
     - 2.68.1
   * - huggingface-translation-opus-mt-en-es
     - False
     - 1.0.0
     - 2.68.1
   * - huggingface-translation-opus-mt-en-vi
     - False
     - 1.0.0
     - 2.68.1
   * - huggingface-translation-t5-base
     - False
     - 1.0.0
     - 2.68.1
   * - huggingface-translation-t5-large
     - False
     - 1.0.0
     - 2.68.1
   * - huggingface-translation-t5-small
     - False
     - 1.0.0
     - 2.68.1
   * - lightgbm-classification-model
     - True
     - 1.0.0
     - 2.68.1
   * - lightgbm-regression-model
     - True
     - 1.0.0
     - 2.68.1
   * - mxnet-is-mask-rcnn-fpn-resnet101-v1d-coco
     - False
     - 1.0.0
     - 2.68.1
   * - mxnet-is-mask-rcnn-fpn-resnet18-v1b-coco
     - False
     - 1.0.0
     - 2.68.1
   * - mxnet-is-mask-rcnn-fpn-resnet50-v1b-coco
     - False
     - 1.0.0
     - 2.68.1
   * - mxnet-is-mask-rcnn-resnet18-v1b-coco
     - False
     - 1.0.0
     - 2.68.1
   * - mxnet-od-faster-rcnn-fpn-resnet101-v1d-coco
     - False
     - 1.0.0
     - 2.68.1
   * - mxnet-od-faster-rcnn-fpn-resnet50-v1b-coco
     - False
     - 1.0.0
     - 2.68.1
   * - mxnet-od-faster-rcnn-resnet101-v1d-coco
     - False
     - 1.0.0
     - 2.68.1
   * - mxnet-od-faster-rcnn-resnet50-v1b-coco
     - False
     - 1.0.0
     - 2.68.1
   * - mxnet-od-faster-rcnn-resnet50-v1b-voc
     - False
     - 1.0.0
     - 2.68.1
   * - mxnet-od-ssd-300-vgg16-atrous-coco
     - True
     - 1.0.0
     - 2.68.1
   * - mxnet-od-ssd-300-vgg16-atrous-voc
     - True
     - 1.0.0
     - 2.68.1
   * - mxnet-od-ssd-512-mobilenet1-0-coco
     - True
     - 1.0.0
     - 2.68.1
   * - mxnet-od-ssd-512-mobilenet1-0-voc
     - True
     - 1.0.0
     - 2.68.1
   * - mxnet-od-ssd-512-resnet50-v1-coco
     - True
     - 1.0.0
     - 2.68.1
   * - mxnet-od-ssd-512-resnet50-v1-voc
     - True
     - 1.0.0
     - 2.68.1
   * - mxnet-od-ssd-512-vgg16-atrous-coco
     - True
     - 1.0.0
     - 2.68.1
   * - mxnet-od-ssd-512-vgg16-atrous-voc
     - True
     - 1.0.0
     - 2.68.1
   * - mxnet-od-yolo3-darknet53-coco
     - False
     - 1.0.0
     - 2.68.1
   * - mxnet-od-yolo3-darknet53-voc
     - False
     - 1.0.0
     - 2.68.1
   * - mxnet-od-yolo3-mobilenet1-0-coco
     - False
     - 1.0.0
     - 2.68.1
   * - mxnet-od-yolo3-mobilenet1-0-voc
     - False
     - 1.0.0
     - 2.68.1
   * - mxnet-semseg-fcn-resnet101-ade
     - True
     - 1.0.0
     - 2.68.1
   * - mxnet-semseg-fcn-resnet101-coco
     - True
     - 1.0.0
     - 2.68.1
   * - mxnet-semseg-fcn-resnet101-voc
     - True
     - 1.0.0
     - 2.68.1
   * - mxnet-semseg-fcn-resnet50-ade
     - True
     - 1.0.0
     - 2.68.1
   * - mxnet-tcembedding-robertafin-base-uncased
     - False
     - 1.0.0
     - 2.68.1
   * - mxnet-tcembedding-robertafin-base-wiki-uncased
     - False
     - 1.0.0
     - 2.68.1
   * - mxnet-tcembedding-robertafin-large-uncased
     - False
     - 1.0.0
     - 2.68.1
   * - mxnet-tcembedding-robertafin-large-wiki-uncased
     - False
     - 1.0.0
     - 2.68.1
   * - pytorch-eqa-bert-base-cased
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-eqa-bert-base-multilingual-cased
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-eqa-bert-base-multilingual-uncased
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-eqa-bert-base-uncased
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-eqa-bert-large-cased
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-eqa-bert-large-cased-whole-word-masking
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-eqa-bert-large-cased-whole-word-masking-finetuned-squad
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-eqa-bert-large-uncased
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-eqa-bert-large-uncased-whole-word-masking
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-eqa-bert-large-uncased-whole-word-masking-finetuned-squad
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-eqa-distilbert-base-cased
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-eqa-distilbert-base-multilingual-cased
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-eqa-distilbert-base-uncased
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-eqa-distilroberta-base
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-eqa-roberta-base
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-eqa-roberta-base-openai-detector
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-eqa-roberta-large
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-eqa-roberta-large-openai-detector
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-ic-densenet121
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-ic-densenet161
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-ic-densenet169
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-ic-densenet201
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-ic-googlenet
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-ic-mobilenet-v2
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-ic-resnet101
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-ic-resnet152
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-ic-resnet18
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-ic-resnet34
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-ic-resnet50
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-ic-resnext101-32x8d
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-ic-resnext50-32x4d
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-ic-shufflenet-v2-x1-0
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-ic-squeezenet1-0
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-ic-squeezenet1-1
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-ic-vgg11
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-ic-vgg11-bn
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-ic-vgg13
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-ic-vgg13-bn
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-ic-vgg16
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-ic-vgg16-bn
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-ic-vgg19
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-ic-vgg19-bn
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-ic-wide-resnet101-2
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-ic-wide-resnet50-2
     - True
     - 1.0.0
     - 2.68.1
   * - pytorch-od-nvidia-ssd
     - False
     - 1.0.0
     - 2.68.1
   * - pytorch-od1-fasterrcnn-mobilenet-v3-large-320-fpn
     - False
     - 1.0.0
     - 2.68.1
   * - pytorch-od1-fasterrcnn-mobilenet-v3-large-fpn
     - False
     - 1.0.0
     - 2.68.1
   * - pytorch-od1-fasterrcnn-resnet50-fpn
     - True
     - 1.0.0
     - 2.68.1
   * - sklearn-classification-linear
     - True
     - 1.0.0
     - 2.68.1
   * - sklearn-regression-linear
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-bit-m-r101x1-ilsvrc2012-classification-1
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-bit-m-r101x1-imagenet21k-classification-1
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-bit-m-r101x3-ilsvrc2012-classification-1
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-bit-m-r101x3-imagenet21k-classification-1
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-bit-m-r50x1-ilsvrc2012-classification-1
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-bit-m-r50x1-imagenet21k-classification-1
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-bit-m-r50x3-ilsvrc2012-classification-1
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-bit-m-r50x3-imagenet21k-classification-1
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-bit-s-r101x1-ilsvrc2012-classification-1
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-bit-s-r101x3-ilsvrc2012-classification-1
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-bit-s-r50x1-ilsvrc2012-classification-1
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-bit-s-r50x3-ilsvrc2012-classification-1
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-efficientnet-b0-classification-1
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-efficientnet-b1-classification-1
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-efficientnet-b2-classification-1
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-efficientnet-b3-classification-1
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-efficientnet-b4-classification-1
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-efficientnet-b5-classification-1
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-efficientnet-b6-classification-1
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-efficientnet-b7-classification-1
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-efficientnet-lite0-classification-2
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-efficientnet-lite1-classification-2
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-efficientnet-lite2-classification-2
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-efficientnet-lite3-classification-2
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-efficientnet-lite4-classification-2
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-imagenet-inception-resnet-v2-classification-4
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-imagenet-inception-v1-classification-4
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-imagenet-inception-v2-classification-4
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-imagenet-inception-v3-classification-4
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-imagenet-mobilenet-v1-025-128-classification-4
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-imagenet-mobilenet-v1-025-160-classification-4
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-imagenet-mobilenet-v1-025-192-classification-4
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-imagenet-mobilenet-v1-025-224-classification-4
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-imagenet-mobilenet-v1-050-128-classification-4
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-imagenet-mobilenet-v1-050-160-classification-4
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-imagenet-mobilenet-v1-050-192-classification-4
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-imagenet-mobilenet-v1-050-224-classification-4
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-imagenet-mobilenet-v1-075-128-classification-4
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-imagenet-mobilenet-v1-075-160-classification-4
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-imagenet-mobilenet-v1-075-192-classification-4
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-imagenet-mobilenet-v1-075-224-classification-4
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-imagenet-mobilenet-v1-100-128-classification-4
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-imagenet-mobilenet-v1-100-160-classification-4
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-imagenet-mobilenet-v1-100-192-classification-4
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-imagenet-mobilenet-v1-100-224-classification-4
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-imagenet-mobilenet-v2-035-224-classification-4
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-imagenet-mobilenet-v2-050-224-classification-4
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-imagenet-mobilenet-v2-075-224-classification-4
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-imagenet-mobilenet-v2-100-224-classification-4
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-imagenet-mobilenet-v2-130-224-classification-4
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-imagenet-mobilenet-v2-140-224-classification-4
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-imagenet-resnet-v1-101-classification-4
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-imagenet-resnet-v1-152-classification-4
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-imagenet-resnet-v1-50-classification-4
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-imagenet-resnet-v2-101-classification-4
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-imagenet-resnet-v2-152-classification-4
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-imagenet-resnet-v2-50-classification-4
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-resnet-50-classification-1
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-tf2-preview-inception-v3-classification-4
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-ic-tf2-preview-mobilenet-v2-classification-4
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-bit-m-r101x1-ilsvrc2012-featurevector-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-bit-m-r101x3-imagenet21k-featurevector-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-bit-m-r50x1-ilsvrc2012-featurevector-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-bit-m-r50x3-imagenet21k-featurevector-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-bit-s-r101x1-ilsvrc2012-featurevector-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-bit-s-r101x3-ilsvrc2012-featurevector-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-bit-s-r50x1-ilsvrc2012-featurevector-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-bit-s-r50x3-ilsvrc2012-featurevector-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-efficientnet-b0-featurevector-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-efficientnet-b1-featurevector-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-efficientnet-b2-featurevector-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-efficientnet-b3-featurevector-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-efficientnet-b6-featurevector-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-efficientnet-lite0-featurevector-2
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-efficientnet-lite1-featurevector-2
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-efficientnet-lite2-featurevector-2
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-efficientnet-lite3-featurevector-2
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-efficientnet-lite4-featurevector-2
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-imagenet-inception-v1-featurevector-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-imagenet-inception-v2-featurevector-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-imagenet-inception-v3-featurevector-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-imagenet-mobilenet-v1-025-128-featurevector-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-imagenet-mobilenet-v1-025-160-featurevector-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-imagenet-mobilenet-v1-025-192-featurevector-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-imagenet-mobilenet-v1-025-224-featurevector-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-imagenet-mobilenet-v1-050-128-featurevector-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-imagenet-mobilenet-v1-050-160-featurevector-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-imagenet-mobilenet-v1-050-192-featurevector-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-imagenet-mobilenet-v1-050-224-featurevector-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-imagenet-mobilenet-v1-075-128-featurevector-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-imagenet-mobilenet-v1-075-160-featurevector-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-imagenet-mobilenet-v1-075-192-featurevector-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-imagenet-mobilenet-v1-075-224-featurevector-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-imagenet-mobilenet-v1-100-128-featurevector-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-imagenet-mobilenet-v1-100-160-featurevector-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-imagenet-mobilenet-v1-100-192-featurevector-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-imagenet-mobilenet-v1-100-224-featurevector-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-imagenet-mobilenet-v2-035-224-featurevector-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-imagenet-mobilenet-v2-050-224-featurevector-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-imagenet-mobilenet-v2-075-224-featurevector-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-imagenet-mobilenet-v2-100-224-featurevector-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-imagenet-mobilenet-v2-130-224-featurevector-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-imagenet-mobilenet-v2-140-224-featurevector-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-imagenet-resnet-v1-101-featurevector-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-imagenet-resnet-v1-152-featurevector-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-imagenet-resnet-v1-50-featurevector-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-imagenet-resnet-v2-101-featurevector-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-imagenet-resnet-v2-152-featurevector-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-imagenet-resnet-v2-50-featurevector-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-resnet-50-featurevector-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-tf2-preview-inception-v3-featurevector-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-icembedding-tf2-preview-mobilenet-v2-featurevector-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-od-centernet-hourglass-1024x1024-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-od-centernet-hourglass-1024x1024-kpts-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-od-centernet-hourglass-512x512-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-od-centernet-hourglass-512x512-kpts-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-od-centernet-resnet101v1-fpn-512x512-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-od-centernet-resnet50v1-fpn-512x512-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-od-centernet-resnet50v1-fpn-512x512-kpts-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-od-centernet-resnet50v2-512x512-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-od-centernet-resnet50v2-512x512-kpts-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-od-efficientdet-d0-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-od-efficientdet-d1-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-od-efficientdet-d2-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-od-efficientdet-d3-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-od-efficientdet-d4-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-od-efficientdet-d5-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-od-faster-rcnn-inception-resnet-v2-1024x1024-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-od-faster-rcnn-inception-resnet-v2-640x640-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-od-faster-rcnn-resnet101-v1-1024x1024-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-od-faster-rcnn-resnet101-v1-640x640-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-od-faster-rcnn-resnet101-v1-800x1333-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-od-faster-rcnn-resnet152-v1-1024x1024-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-od-faster-rcnn-resnet152-v1-640x640-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-od-faster-rcnn-resnet152-v1-800x1333-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-od-faster-rcnn-resnet50-v1-1024x1024-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-od-faster-rcnn-resnet50-v1-640x640-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-od-faster-rcnn-resnet50-v1-800x1333-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-od-retinanet-resnet101-v1-fpn-1024x1024-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-od-retinanet-resnet101-v1-fpn-640x640-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-od-retinanet-resnet152-v1-fpn-1024x1024-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-od-retinanet-resnet152-v1-fpn-640x640-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-od-retinanet-resnet50-v1-fpn-1024x1024-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-od-retinanet-resnet50-v1-fpn-640x640-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-od-ssd-mobilenet-v1-fpn-640x640-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-od-ssd-mobilenet-v2-2
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-od-ssd-mobilenet-v2-fpnlite-320x320-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-od-ssd-mobilenet-v2-fpnlite-640x640-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-spc-bert-en-cased-L-12-H-768-A-12-2
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-spc-bert-en-uncased-L-12-H-768-A-12-2
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-spc-bert-en-uncased-L-24-H-1024-A-16-2
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-spc-bert-en-wwm-cased-L-24-H-1024-A-16-2
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-spc-bert-en-wwm-uncased-L-24-H-1024-A-16-2
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-spc-bert-multi-cased-L-12-H-768-A-12-2
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-spc-electra-base-1
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-spc-electra-small-1
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-spc-experts-bert-pubmed-1
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-spc-experts-bert-wiki-books-1
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-tc-bert-en-cased-L-12-H-768-A-12-2
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-tc-bert-en-cased-L-24-H-1024-A-16-2
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-tc-bert-en-uncased-L-12-H-768-A-12-2
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-tc-bert-en-wwm-cased-L-24-H-1024-A-16-2
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-tc-bert-en-wwm-uncased-L-24-H-1024-A-16-2
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-tc-bert-multi-cased-L-12-H-768-A-12-2
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-tc-electra-base-1
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-tc-electra-small-1
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-tc-experts-bert-pubmed-1
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-tc-experts-bert-wiki-books-1
     - True
     - 1.0.0
     - 2.68.1
   * - tensorflow-tcembedding-bert-en-uncased-L-10-H-128-A-2-2
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-tcembedding-bert-en-uncased-L-10-H-256-A-4-2
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-tcembedding-bert-en-uncased-L-10-H-512-A-8-2
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-tcembedding-bert-en-uncased-L-10-H-768-A-12-2
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-tcembedding-bert-en-uncased-L-12-H-128-A-2-2
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-tcembedding-bert-en-uncased-L-12-H-256-A-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-tcembedding-bert-en-uncased-L-12-H-512-A-8-2
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-tcembedding-bert-en-uncased-L-12-H-768-A-12-2
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-tcembedding-bert-en-uncased-L-12-H-768-A-12-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-tcembedding-bert-en-uncased-L-2-H-128-A-2-2
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-tcembedding-bert-en-uncased-L-2-H-256-A-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-tcembedding-bert-en-uncased-L-2-H-512-A-8-2
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-tcembedding-bert-en-uncased-L-2-H-768-A-12-2
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-tcembedding-bert-en-uncased-L-4-H-128-A-2-2
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-tcembedding-bert-en-uncased-L-4-H-256-A-4-2
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-tcembedding-bert-en-uncased-L-4-H-512-A-8-2
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-tcembedding-bert-en-uncased-L-4-H-768-A-12-2
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-tcembedding-bert-en-uncased-L-6-H-128-A-2-2
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-tcembedding-bert-en-uncased-L-6-H-256-A-4
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-tcembedding-bert-en-uncased-L-6-H-512-A-8-2
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-tcembedding-bert-en-uncased-L-6-H-768-A-12-2
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-tcembedding-bert-en-uncased-L-8-H-256-A-4-2
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-tcembedding-bert-en-uncased-L-8-H-512-A-8-2
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-tcembedding-bert-en-uncased-L-8-H-768-A-12-2
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-tcembedding-bert-wiki-books-mnli-2
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-tcembedding-bert-wiki-books-sst2
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-tcembedding-talkheads-ggelu-bert-en-base-2
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-tcembedding-talkheads-ggelu-bert-en-large-2
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-tcembedding-universal-sentence-encoder-cmlm-en-base-1
     - False
     - 1.0.0
     - 2.68.1
   * - tensorflow-tcembedding-universal-sentence-encoder-cmlm-en-large-1
     - False
     - 1.0.0
     - 2.68.1
   * - xgboost-classification-model
     - True
     - 1.0.0
     - 2.68.1
   * - xgboost-regression-model
     - True
     - 1.0.0
     - 2.68.1
