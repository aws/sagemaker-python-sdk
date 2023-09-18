#############################
Image Embedding - TensorFlow
#############################

This is a supervised image embedding algorithm which supports many pre-trained models available in Tensorflow Hub. The following
`sample notebook <https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/jumpstart_image_embedding/Amazon_JumpStart_Image_Embedding.ipynb>`__
demonstrates how to use the Sagemaker Python SDK for Image Embedding for using these algorithms.

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
   * - tensorflow-icembedding-bit-m-r101x1-ilsvrc2012-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r101x1/1>`__
   * - tensorflow-icembedding-bit-m-r101x3-imagenet21k-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r101x3/1>`__
   * - tensorflow-icembedding-bit-m-r50x1-ilsvrc2012-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r50x1/1>`__
   * - tensorflow-icembedding-bit-m-r50x3-imagenet21k-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r50x3/1>`__
   * - tensorflow-icembedding-bit-s-r101x1-ilsvrc2012-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r101x1/1>`__
   * - tensorflow-icembedding-bit-s-r101x3-ilsvrc2012-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r101x3/1>`__
   * - tensorflow-icembedding-bit-s-r50x1-ilsvrc2012-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r50x1/1>`__
   * - tensorflow-icembedding-bit-s-r50x3-ilsvrc2012-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r50x3/1>`__
   * - tensorflow-icembedding-efficientnet-b0-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b0/feature-vector/1>`__
   * - tensorflow-icembedding-efficientnet-b1-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b1/feature-vector/1>`__
   * - tensorflow-icembedding-efficientnet-b2-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b2/feature-vector/1>`__
   * - tensorflow-icembedding-efficientnet-b3-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b3/feature-vector/1>`__
   * - tensorflow-icembedding-efficientnet-b6-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b6/feature-vector/1>`__
   * - tensorflow-icembedding-efficientnet-lite0-featurevector-2
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2>`__
   * - tensorflow-icembedding-efficientnet-lite1-featurevector-2
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite1/feature-vector/2>`__
   * - tensorflow-icembedding-efficientnet-lite2-featurevector-2
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite2/feature-vector/2>`__
   * - tensorflow-icembedding-efficientnet-lite3-featurevector-2
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite3/feature-vector/2>`__
   * - tensorflow-icembedding-efficientnet-lite4-featurevector-2
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite4/feature-vector/2>`__
   * - tensorflow-icembedding-imagenet-inception-v1-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/inception_v1/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-inception-v2-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/inception_v2/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-inception-v3-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v1-025-128-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_128/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v1-025-160-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_160/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v1-025-192-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_192/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v1-025-224-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v1-050-128-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_128/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v1-050-160-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v1-050-192-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_192/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v1-050-224-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_224/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v1-075-128-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_128/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v1-075-160-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_160/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v1-075-192-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_192/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v1-075-224-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_224/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v1-100-128-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_128/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v1-100-160-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_160/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v1-100-192-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_192/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v1-100-224-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v2-035-224-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_035_224/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v2-050-224-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_050_224/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v2-075-224-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_075_224/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v2-100-224-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v2-130-224-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v2-140-224-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-resnet-v1-101-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v1_101/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-resnet-v1-152-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v1_152/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-resnet-v1-50-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-resnet-v2-101-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-resnet-v2-152-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-resnet-v2-50-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4>`__
   * - tensorflow-icembedding-resnet-50-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/resnet_50/feature_vector/1>`__
   * - tensorflow-icembedding-tf2-preview-inception-v3-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4>`__
   * - tensorflow-icembedding-tf2-preview-mobilenet-v2-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4>`__

.. list-table:: Available Models
   :widths: 50 20 20 20 20
   :header-rows: 1
   :class: datatable

   * - Model ID
     - Fine Tunable?
     - Latest Version
     - Min SDK Version
     - Source
   * - tensorflow-icembedding-bit-m-r101x1-ilsvrc2012-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r101x1/1>`__
   * - tensorflow-icembedding-bit-m-r101x3-imagenet21k-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r101x3/1>`__
   * - tensorflow-icembedding-bit-m-r50x1-ilsvrc2012-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r50x1/1>`__
   * - tensorflow-icembedding-bit-m-r50x3-imagenet21k-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r50x3/1>`__
   * - tensorflow-icembedding-bit-s-r101x1-ilsvrc2012-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r101x1/1>`__
   * - tensorflow-icembedding-bit-s-r101x3-ilsvrc2012-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r101x3/1>`__
   * - tensorflow-icembedding-bit-s-r50x1-ilsvrc2012-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r50x1/1>`__
   * - tensorflow-icembedding-bit-s-r50x3-ilsvrc2012-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r50x3/1>`__
   * - tensorflow-icembedding-efficientnet-b0-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b0/feature-vector/1>`__
   * - tensorflow-icembedding-efficientnet-b1-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b1/feature-vector/1>`__
   * - tensorflow-icembedding-efficientnet-b2-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b2/feature-vector/1>`__
   * - tensorflow-icembedding-efficientnet-b3-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b3/feature-vector/1>`__
   * - tensorflow-icembedding-efficientnet-b6-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/efficientnet/b6/feature-vector/1>`__
   * - tensorflow-icembedding-efficientnet-lite0-featurevector-2
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2>`__
   * - tensorflow-icembedding-efficientnet-lite1-featurevector-2
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite1/feature-vector/2>`__
   * - tensorflow-icembedding-efficientnet-lite2-featurevector-2
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite2/feature-vector/2>`__
   * - tensorflow-icembedding-efficientnet-lite3-featurevector-2
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite3/feature-vector/2>`__
   * - tensorflow-icembedding-efficientnet-lite4-featurevector-2
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite4/feature-vector/2>`__
   * - tensorflow-icembedding-imagenet-inception-v1-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/inception_v1/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-inception-v2-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/inception_v2/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-inception-v3-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v1-025-128-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_128/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v1-025-160-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_160/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v1-025-192-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_192/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v1-025-224-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v1-050-128-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_128/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v1-050-160-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v1-050-192-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_192/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v1-050-224-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_224/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v1-075-128-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_128/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v1-075-160-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_160/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v1-075-192-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_192/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v1-075-224-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_224/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v1-100-128-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_128/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v1-100-160-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_160/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v1-100-192-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_192/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v1-100-224-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v2-035-224-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_035_224/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v2-050-224-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_050_224/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v2-075-224-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_075_224/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v2-100-224-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v2-130-224-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-mobilenet-v2-140-224-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-resnet-v1-101-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v1_101/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-resnet-v1-152-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v1_152/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-resnet-v1-50-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-resnet-v2-101-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-resnet-v2-152-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/4>`__
   * - tensorflow-icembedding-imagenet-resnet-v2-50-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4>`__
   * - tensorflow-icembedding-resnet-50-featurevector-1
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/resnet_50/feature_vector/1>`__
   * - tensorflow-icembedding-tf2-preview-inception-v3-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4>`__
   * - tensorflow-icembedding-tf2-preview-mobilenet-v2-featurevector-4
     - False
     - 2.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4>`__
