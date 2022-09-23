##################################
Image Classification - TensorFlow
##################################

This is a supervised image clasification algorithm which supports fine-tuning of many pre-trained models available in Tensorflow Hub. The following
`sample notebook <https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/jumpstart_image_classification/Amazon_JumpStart_Image_Classification.ipynb>`__
demonstrates how to use the Sagemaker Python SDK for Image Classification for using these algorithms.

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
   * - tensorflow-ic-bit-m-r101x1-ilsvrc2012-classification-1
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r101x1/ilsvrc2012_classification/1>`__
   * - tensorflow-ic-bit-m-r101x1-imagenet21k-classification-1
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r101x1/imagenet21k_classification/1>`__
   * - tensorflow-ic-bit-m-r101x3-ilsvrc2012-classification-1
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r101x3/ilsvrc2012_classification/1>`__
   * - tensorflow-ic-bit-m-r101x3-imagenet21k-classification-1
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r101x3/imagenet21k_classification/1>`__
   * - tensorflow-ic-bit-m-r50x1-ilsvrc2012-classification-1
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r50x1/ilsvrc2012_classification/1>`__
   * - tensorflow-ic-bit-m-r50x1-imagenet21k-classification-1
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r50x1/imagenet21k_classification/1>`__
   * - tensorflow-ic-bit-m-r50x3-ilsvrc2012-classification-1
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r50x3/ilsvrc2012_classification/1>`__
   * - tensorflow-ic-bit-m-r50x3-imagenet21k-classification-1
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r50x3/imagenet21k_classification/1>`__
   * - tensorflow-ic-bit-s-r101x1-ilsvrc2012-classification-1
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r101x1/ilsvrc2012_classification/1>`__
   * - tensorflow-ic-bit-s-r101x3-ilsvrc2012-classification-1
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r101x3/ilsvrc2012_classification/1>`__
   * - tensorflow-ic-bit-s-r50x1-ilsvrc2012-classification-1
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r50x1/ilsvrc2012_classification/1>`__
   * - tensorflow-ic-bit-s-r50x3-ilsvrc2012-classification-1
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r50x3/ilsvrc2012_classification/1>`__
   * - tensorflow-ic-efficientnet-b0-classification-1
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b0/classification/1>`__
   * - tensorflow-ic-efficientnet-b1-classification-1
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b1/classification/1>`__
   * - tensorflow-ic-efficientnet-b2-classification-1
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b2/classification/1>`__
   * - tensorflow-ic-efficientnet-b3-classification-1
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b3/classification/1>`__
   * - tensorflow-ic-efficientnet-b4-classification-1
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b4/classification/1>`__
   * - tensorflow-ic-efficientnet-b5-classification-1
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b5/classification/1>`__
   * - tensorflow-ic-efficientnet-b6-classification-1
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b6/classification/1>`__
   * - tensorflow-ic-efficientnet-b7-classification-1
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b7/classification/1>`__
   * - tensorflow-ic-efficientnet-lite0-classification-2
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite0/classification/2>`__
   * - tensorflow-ic-efficientnet-lite1-classification-2
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite1/classification/2>`__
   * - tensorflow-ic-efficientnet-lite2-classification-2
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite2/classification/2>`__
   * - tensorflow-ic-efficientnet-lite3-classification-2
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite3/classification/2>`__
   * - tensorflow-ic-efficientnet-lite4-classification-2
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite4/classification/2>`__
   * - tensorflow-ic-imagenet-inception-resnet-v2-classification-4
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/4>`__
   * - tensorflow-ic-imagenet-inception-v1-classification-4
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/inception_v1/classification/4>`__
   * - tensorflow-ic-imagenet-inception-v2-classification-4
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/inception_v2/classification/4>`__
   * - tensorflow-ic-imagenet-inception-v3-classification-4
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/inception_v3/classification/4>`__
   * - tensorflow-ic-imagenet-mobilenet-v1-025-128-classification-4
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_128/classification/4>`__
   * - tensorflow-ic-imagenet-mobilenet-v1-025-160-classification-4
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_160/classification/4>`__
   * - tensorflow-ic-imagenet-mobilenet-v1-025-192-classification-4
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_192/classification/4>`__
   * - tensorflow-ic-imagenet-mobilenet-v1-025-224-classification-4
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/classification/4>`__
   * - tensorflow-ic-imagenet-mobilenet-v1-050-128-classification-4
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_128/classification/4>`__
   * - tensorflow-ic-imagenet-mobilenet-v1-050-160-classification-4
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/classification/4>`__
   * - tensorflow-ic-imagenet-mobilenet-v1-050-192-classification-4
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_192/classification/4>`__
   * - tensorflow-ic-imagenet-mobilenet-v1-050-224-classification-4
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_224/classification/4>`__
   * - tensorflow-ic-imagenet-mobilenet-v1-075-128-classification-4
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_128/classification/4>`__
   * - tensorflow-ic-imagenet-mobilenet-v1-075-160-classification-4
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_160/classification/4>`__
   * - tensorflow-ic-imagenet-mobilenet-v1-075-192-classification-4
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_192/classification/4>`__
   * - tensorflow-ic-imagenet-mobilenet-v1-075-224-classification-4
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_224/classification/4>`__
   * - tensorflow-ic-imagenet-mobilenet-v1-100-128-classification-4
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_128/classification/4>`__
   * - tensorflow-ic-imagenet-mobilenet-v1-100-160-classification-4
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_160/classification/4>`__
   * - tensorflow-ic-imagenet-mobilenet-v1-100-192-classification-4
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_192/classification/4>`__
   * - tensorflow-ic-imagenet-mobilenet-v1-100-224-classification-4
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/4>`__
   * - tensorflow-ic-imagenet-mobilenet-v2-035-224-classification-4
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_035_224/classification/4>`__
   * - tensorflow-ic-imagenet-mobilenet-v2-050-224-classification-4
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_050_224/classification/4>`__
   * - tensorflow-ic-imagenet-mobilenet-v2-075-224-classification-4
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_075_224/classification/4>`__
   * - tensorflow-ic-imagenet-mobilenet-v2-100-224-classification-4
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4>`__
   * - tensorflow-ic-imagenet-mobilenet-v2-130-224-classification-4
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4>`__
   * - tensorflow-ic-imagenet-mobilenet-v2-140-224-classification-4
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/4>`__
   * - tensorflow-ic-imagenet-resnet-v1-101-classification-4
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v1_101/classification/4>`__
   * - tensorflow-ic-imagenet-resnet-v1-152-classification-4
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v1_152/classification/4>`__
   * - tensorflow-ic-imagenet-resnet-v1-50-classification-4
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v1_50/classification/4>`__
   * - tensorflow-ic-imagenet-resnet-v2-101-classification-4
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v2_101/classification/4>`__
   * - tensorflow-ic-imagenet-resnet-v2-152-classification-4
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v2_152/classification/4>`__
   * - tensorflow-ic-imagenet-resnet-v2-50-classification-4
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v2_50/classification/4>`__
   * - tensorflow-ic-resnet-50-classification-1
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/resnet_50/classification/1>`__
   * - tensorflow-ic-tf2-preview-inception-v3-classification-4
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/tf2-preview/inception_v3/classification/4>`__
   * - tensorflow-ic-tf2-preview-mobilenet-v2-classification-4
     - True
     - 2.0.3
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4>`__
