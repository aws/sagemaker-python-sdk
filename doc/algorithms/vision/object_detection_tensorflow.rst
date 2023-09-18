###############################
Object Detection - TensorFlow
###############################

This is a supervised object detection algorithm which supports fine-tuning of many pre-trained models available in Tensorflow Hub. The following
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
   * - tensorflow-od-centernet-hourglass-1024x1024-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/hourglass_1024x1024/1>`__
   * - tensorflow-od-centernet-hourglass-1024x1024-kpts-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/hourglass_1024x1024_kpts/1>`__
   * - tensorflow-od-centernet-hourglass-512x512-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1>`__
   * - tensorflow-od-centernet-hourglass-512x512-kpts-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/hourglass_512x512_kpts/1>`__
   * - tensorflow-od-centernet-resnet101v1-fpn-512x512-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/resnet101v1_fpn_512x512/1>`__
   * - tensorflow-od-centernet-resnet50v1-fpn-512x512-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512/1>`__
   * - tensorflow-od-centernet-resnet50v1-fpn-512x512-kpts-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512_kpts/1>`__
   * - tensorflow-od-centernet-resnet50v2-512x512-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/resnet50v2_512x512/1>`__
   * - tensorflow-od-centernet-resnet50v2-512x512-kpts-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/resnet50v2_512x512_kpts/1>`__
   * - tensorflow-od-efficientdet-d0-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientdet/d0/1>`__
   * - tensorflow-od-efficientdet-d1-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientdet/d1/1>`__
   * - tensorflow-od-efficientdet-d2-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientdet/d2/1>`__
   * - tensorflow-od-efficientdet-d3-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientdet/d3/1>`__
   * - tensorflow-od-efficientdet-d4-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientdet/d4/1>`__
   * - tensorflow-od-efficientdet-d5-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientdet/d5/1>`__
   * - tensorflow-od-faster-rcnn-inception-resnet-v2-1024x1024-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_1024x1024/1>`__
   * - tensorflow-od-faster-rcnn-inception-resnet-v2-640x640-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1>`__
   * - tensorflow-od-faster-rcnn-resnet101-v1-1024x1024-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_1024x1024/1>`__
   * - tensorflow-od-faster-rcnn-resnet101-v1-640x640-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_640x640/1>`__
   * - tensorflow-od-faster-rcnn-resnet101-v1-800x1333-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_800x1333/1>`__
   * - tensorflow-od-faster-rcnn-resnet152-v1-1024x1024-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_1024x1024/1>`__
   * - tensorflow-od-faster-rcnn-resnet152-v1-640x640-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_640x640/1>`__
   * - tensorflow-od-faster-rcnn-resnet152-v1-800x1333-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_800x1333/1>`__
   * - tensorflow-od-faster-rcnn-resnet50-v1-1024x1024-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_1024x1024/1>`__
   * - tensorflow-od-faster-rcnn-resnet50-v1-640x640-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1>`__
   * - tensorflow-od-faster-rcnn-resnet50-v1-800x1333-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_800x1333/1>`__
   * - tensorflow-od-retinanet-resnet101-v1-fpn-1024x1024-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/retinanet/resnet101_v1_fpn_1024x1024/1>`__
   * - tensorflow-od-retinanet-resnet101-v1-fpn-640x640-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/retinanet/resnet101_v1_fpn_640x640/1>`__
   * - tensorflow-od-retinanet-resnet152-v1-fpn-1024x1024-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/retinanet/resnet152_v1_fpn_1024x1024/1>`__
   * - tensorflow-od-retinanet-resnet152-v1-fpn-640x640-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/retinanet/resnet152_v1_fpn_640x640/1>`__
   * - tensorflow-od-retinanet-resnet50-v1-fpn-1024x1024-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/retinanet/resnet50_v1_fpn_1024x1024/1>`__
   * - tensorflow-od-retinanet-resnet50-v1-fpn-640x640-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/retinanet/resnet50_v1_fpn_640x640/1>`__
   * - tensorflow-od-ssd-mobilenet-v1-fpn-640x640-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/ssd_mobilenet_v1/fpn_640x640/1>`__
   * - tensorflow-od-ssd-mobilenet-v2-2
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2>`__
   * - tensorflow-od-ssd-mobilenet-v2-fpnlite-320x320-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1>`__
   * - tensorflow-od-ssd-mobilenet-v2-fpnlite-640x640-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_640x640/1>`__

.. list-table:: Available Models
   :widths: 50 20 20 20 20
   :header-rows: 1
   :class: datatable

   * - Model ID
     - Fine Tunable?
     - Latest Version
     - Min SDK Version
     - Source
   * - tensorflow-od-centernet-hourglass-1024x1024-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/hourglass_1024x1024/1>`__
   * - tensorflow-od-centernet-hourglass-1024x1024-kpts-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/hourglass_1024x1024_kpts/1>`__
   * - tensorflow-od-centernet-hourglass-512x512-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1>`__
   * - tensorflow-od-centernet-hourglass-512x512-kpts-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/hourglass_512x512_kpts/1>`__
   * - tensorflow-od-centernet-resnet101v1-fpn-512x512-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/resnet101v1_fpn_512x512/1>`__
   * - tensorflow-od-centernet-resnet50v1-fpn-512x512-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512/1>`__
   * - tensorflow-od-centernet-resnet50v1-fpn-512x512-kpts-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512_kpts/1>`__
   * - tensorflow-od-centernet-resnet50v2-512x512-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/resnet50v2_512x512/1>`__
   * - tensorflow-od-centernet-resnet50v2-512x512-kpts-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/centernet/resnet50v2_512x512_kpts/1>`__
   * - tensorflow-od-efficientdet-d0-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientdet/d0/1>`__
   * - tensorflow-od-efficientdet-d1-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientdet/d1/1>`__
   * - tensorflow-od-efficientdet-d2-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientdet/d2/1>`__
   * - tensorflow-od-efficientdet-d3-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientdet/d3/1>`__
   * - tensorflow-od-efficientdet-d4-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientdet/d4/1>`__
   * - tensorflow-od-efficientdet-d5-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientdet/d5/1>`__
   * - tensorflow-od-faster-rcnn-inception-resnet-v2-1024x1024-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_1024x1024/1>`__
   * - tensorflow-od-faster-rcnn-inception-resnet-v2-640x640-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1>`__
   * - tensorflow-od-faster-rcnn-resnet101-v1-1024x1024-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_1024x1024/1>`__
   * - tensorflow-od-faster-rcnn-resnet101-v1-640x640-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_640x640/1>`__
   * - tensorflow-od-faster-rcnn-resnet101-v1-800x1333-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_800x1333/1>`__
   * - tensorflow-od-faster-rcnn-resnet152-v1-1024x1024-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_1024x1024/1>`__
   * - tensorflow-od-faster-rcnn-resnet152-v1-640x640-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_640x640/1>`__
   * - tensorflow-od-faster-rcnn-resnet152-v1-800x1333-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_800x1333/1>`__
   * - tensorflow-od-faster-rcnn-resnet50-v1-1024x1024-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_1024x1024/1>`__
   * - tensorflow-od-faster-rcnn-resnet50-v1-640x640-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1>`__
   * - tensorflow-od-faster-rcnn-resnet50-v1-800x1333-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_800x1333/1>`__
   * - tensorflow-od-retinanet-resnet101-v1-fpn-1024x1024-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/retinanet/resnet101_v1_fpn_1024x1024/1>`__
   * - tensorflow-od-retinanet-resnet101-v1-fpn-640x640-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/retinanet/resnet101_v1_fpn_640x640/1>`__
   * - tensorflow-od-retinanet-resnet152-v1-fpn-1024x1024-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/retinanet/resnet152_v1_fpn_1024x1024/1>`__
   * - tensorflow-od-retinanet-resnet152-v1-fpn-640x640-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/retinanet/resnet152_v1_fpn_640x640/1>`__
   * - tensorflow-od-retinanet-resnet50-v1-fpn-1024x1024-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/retinanet/resnet50_v1_fpn_1024x1024/1>`__
   * - tensorflow-od-retinanet-resnet50-v1-fpn-640x640-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/retinanet/resnet50_v1_fpn_640x640/1>`__
   * - tensorflow-od-ssd-mobilenet-v1-fpn-640x640-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/ssd_mobilenet_v1/fpn_640x640/1>`__
   * - tensorflow-od-ssd-mobilenet-v2-2
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2>`__
   * - tensorflow-od-ssd-mobilenet-v2-fpnlite-320x320-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1>`__
   * - tensorflow-od-ssd-mobilenet-v2-fpnlite-640x640-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_640x640/1>`__
