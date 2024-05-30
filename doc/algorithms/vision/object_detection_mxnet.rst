##########################
Object Detection - MxNet
##########################

This is a supervised object detection algorithm which supports fine-tuning of many pre-trained models available in MXNet. The following
`sample notebook <https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/jumpstart_object_detection/Amazon_JumpStart_Object_Detection.ipynb>`__
demonstrates how to use the Sagemaker Python SDK for Object Detection for using these algorithms.

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
   * - mxnet-od-faster-rcnn-fpn-resnet101-v1d-coco
     - False
     - 2.0.0
     - 2.189.0
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__
   * - mxnet-od-faster-rcnn-fpn-resnet50-v1b-coco
     - False
     - 2.0.0
     - 2.189.0
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__
   * - mxnet-od-faster-rcnn-resnet101-v1d-coco
     - False
     - 2.0.0
     - 2.189.0
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__
   * - mxnet-od-faster-rcnn-resnet50-v1b-coco
     - False
     - 2.0.0
     - 2.189.0
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__
   * - mxnet-od-faster-rcnn-resnet50-v1b-voc
     - False
     - 2.0.0
     - 2.189.0
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__
   * - mxnet-od-ssd-300-vgg16-atrous-coco
     - True
     - 2.0.0
     - 2.189.0
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__
   * - mxnet-od-ssd-300-vgg16-atrous-voc
     - True
     - 2.0.0
     - 2.189.0
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__
   * - mxnet-od-ssd-512-mobilenet1-0-coco
     - True
     - 2.0.0
     - 2.189.0
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__
   * - mxnet-od-ssd-512-mobilenet1-0-voc
     - True
     - 2.0.0
     - 2.189.0
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__
   * - mxnet-od-ssd-512-resnet50-v1-coco
     - True
     - 2.0.0
     - 2.189.0
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__
   * - mxnet-od-ssd-512-resnet50-v1-voc
     - True
     - 2.0.0
     - 2.189.0
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__
   * - mxnet-od-ssd-512-vgg16-atrous-coco
     - True
     - 2.0.0
     - 2.189.0
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__
   * - mxnet-od-ssd-512-vgg16-atrous-voc
     - True
     - 2.0.0
     - 2.189.0
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__
   * - mxnet-od-yolo3-darknet53-coco
     - False
     - 2.0.0
     - 2.189.0
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__
   * - mxnet-od-yolo3-darknet53-voc
     - False
     - 2.0.0
     - 2.189.0
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__
   * - mxnet-od-yolo3-mobilenet1-0-coco
     - False
     - 2.0.0
     - 2.189.0
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__
   * - mxnet-od-yolo3-mobilenet1-0-voc
     - False
     - 2.0.0
     - 2.189.0
     - `GluonCV <https://cv.gluon.ai/model_zoo/detection.html>`__
