##############################
Semantic Segmentation - MxNet
##############################

This is a supervised semantic segmentation algorithm which supports fine-tuning of many pre-trained models available in MXNet. The following
`sample notebook <https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/jumpstart_semantic_segmentation/Amazon_JumpStart_Semantic_Segmentation.ipynb>`__
demonstrates how to use the Sagemaker Python SDK for Semantic Segmentation for using these algorithms.

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
   * - mxnet-semseg-fcn-resnet101-ade
     - True
     - 1.4.2
     - 2.100.0
     - `GluonCV <https://cv.gluon.ai/model_zoo/segmentation.html>`__
   * - mxnet-semseg-fcn-resnet101-coco
     - True
     - 1.4.2
     - 2.100.0
     - `GluonCV <https://cv.gluon.ai/model_zoo/segmentation.html>`__
   * - mxnet-semseg-fcn-resnet101-voc
     - True
     - 1.4.2
     - 2.100.0
     - `GluonCV <https://cv.gluon.ai/model_zoo/segmentation.html>`__
   * - mxnet-semseg-fcn-resnet50-ade
     - True
     - 1.4.2
     - 2.100.0
     - `GluonCV <https://cv.gluon.ai/model_zoo/segmentation.html>`__
