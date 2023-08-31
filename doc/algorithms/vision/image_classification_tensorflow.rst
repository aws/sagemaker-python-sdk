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
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r101x1/ilsvrc2012_classification/1>`__
   * - tensorflow-ic-bit-m-r101x1-imagenet21k-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r101x1/imagenet21k_classification/1>`__
   * - tensorflow-ic-bit-m-r101x3-ilsvrc2012-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r101x3/ilsvrc2012_classification/1>`__
   * - tensorflow-ic-bit-m-r101x3-imagenet21k-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r101x3/imagenet21k_classification/1>`__
   * - tensorflow-ic-bit-m-r152x4-ilsvrc2012
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r152x4/ilsvrc2012_classification/1>`__
   * - tensorflow-ic-bit-m-r152x4-imagenet21k
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r152x4/imagenet21k_classification/1>`__
   * - tensorflow-ic-bit-m-r50x1-ilsvrc2012-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r50x1/ilsvrc2012_classification/1>`__
   * - tensorflow-ic-bit-m-r50x1-imagenet21k-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r50x1/imagenet21k_classification/1>`__
   * - tensorflow-ic-bit-m-r50x3-ilsvrc2012-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r50x3/ilsvrc2012_classification/1>`__
   * - tensorflow-ic-bit-m-r50x3-imagenet21k-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/m-r50x3/imagenet21k_classification/1>`__
   * - tensorflow-ic-bit-s-r101x1-ilsvrc2012-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r101x1/ilsvrc2012_classification/1>`__
   * - tensorflow-ic-bit-s-r101x3-ilsvrc2012-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r101x3/ilsvrc2012_classification/1>`__
   * - tensorflow-ic-bit-s-r152x4-ilsvrc2012
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r152x4/ilsvrc2012_classification/1>`__
   * - tensorflow-ic-bit-s-r50x1-ilsvrc2012-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r50x1/ilsvrc2012_classification/1>`__
   * - tensorflow-ic-bit-s-r50x3-ilsvrc2012-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/bit/s-r50x3/ilsvrc2012_classification/1>`__
   * - tensorflow-ic-cait-m36-384
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/cait_m36_384/1>`__
   * - tensorflow-ic-cait-m48-448
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/cait_m48_448/1>`__
   * - tensorflow-ic-cait-s24-224
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/cait_s24_224/1>`__
   * - tensorflow-ic-cait-s24-384
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/cait_s24_384/1>`__
   * - tensorflow-ic-cait-s36-384
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/cait_s36_384/1>`__
   * - tensorflow-ic-cait-xs24-384
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/cait_xs24_384/1>`__
   * - tensorflow-ic-cait-xxs24-224
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/cait_xxs24_224/1>`__
   * - tensorflow-ic-cait-xxs24-384
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/cait_xxs24_384/1>`__
   * - tensorflow-ic-cait-xxs36-224
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/cait_xxs36_224/1>`__
   * - tensorflow-ic-cait-xxs36-384
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/cait_xxs36_384/1>`__
   * - tensorflow-ic-deit-base-distilled-patch16-224
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/deit_base_distilled_patch16_224/1>`__
   * - tensorflow-ic-deit-base-distilled-patch16-384
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/deit_base_distilled_patch16_384/1>`__
   * - tensorflow-ic-deit-base-patch16-224
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/deit_base_patch16_224/1>`__
   * - tensorflow-ic-deit-base-patch16-384
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/deit_base_patch16_384/1>`__
   * - tensorflow-ic-deit-small-distilled-patch16-224
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/deit_small_distilled_patch16_224/1>`__
   * - tensorflow-ic-deit-small-patch16-224
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/deit_small_patch16_224/1>`__
   * - tensorflow-ic-deit-tiny-distilled-patch16-224
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/deit_tiny_distilled_patch16_224/1>`__
   * - tensorflow-ic-deit-tiny-patch16-224
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/deit_tiny_patch16_224/1>`__
   * - tensorflow-ic-efficientnet-b0-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/b0/classification/1>`__
   * - tensorflow-ic-efficientnet-b1-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/b1/classification/1>`__
   * - tensorflow-ic-efficientnet-b2-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/b2/classification/1>`__
   * - tensorflow-ic-efficientnet-b3-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/b3/classification/1>`__
   * - tensorflow-ic-efficientnet-b4-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/b4/classification/1>`__
   * - tensorflow-ic-efficientnet-b5-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/b5/classification/1>`__
   * - tensorflow-ic-efficientnet-b6-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/b6/classification/1>`__
   * - tensorflow-ic-efficientnet-b7-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/b7/classification/1>`__
   * - tensorflow-ic-efficientnet-lite0-classification-2
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite0/classification/2>`__
   * - tensorflow-ic-efficientnet-lite1-classification-2
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite1/classification/2>`__
   * - tensorflow-ic-efficientnet-lite2-classification-2
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite2/classification/2>`__
   * - tensorflow-ic-efficientnet-lite3-classification-2
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite3/classification/2>`__
   * - tensorflow-ic-efficientnet-lite4-classification-2
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/efficientnet/lite4/classification/2>`__
   * - tensorflow-ic-efficientnet-v2-imagenet1k-b0
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/classification/2>`__
   * - tensorflow-ic-efficientnet-v2-imagenet1k-b1
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b1/classification/2>`__
   * - tensorflow-ic-efficientnet-v2-imagenet1k-b2
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b2/classification/2>`__
   * - tensorflow-ic-efficientnet-v2-imagenet1k-b3
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b3/classification/2>`__
   * - tensorflow-ic-efficientnet-v2-imagenet1k-l
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_l/classification/2>`__
   * - tensorflow-ic-efficientnet-v2-imagenet1k-m
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_m/classification/2>`__
   * - tensorflow-ic-efficientnet-v2-imagenet1k-s
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/classification/2>`__
   * - tensorflow-ic-efficientnet-v2-imagenet21k-b0
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b0/classification/2>`__
   * - tensorflow-ic-efficientnet-v2-imagenet21k-b1
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b1/classification/2>`__
   * - tensorflow-ic-efficientnet-v2-imagenet21k-b2
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b2/classification/2>`__
   * - tensorflow-ic-efficientnet-v2-imagenet21k-b3
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b3/classification/2>`__
   * - tensorflow-ic-efficientnet-v2-imagenet21k-ft1k-b0
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b0/classification/2>`__
   * - tensorflow-ic-efficientnet-v2-imagenet21k-ft1k-b1
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b1/classification/2>`__
   * - tensorflow-ic-efficientnet-v2-imagenet21k-ft1k-b2
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b2/classification/2>`__
   * - tensorflow-ic-efficientnet-v2-imagenet21k-ft1k-b3
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b3/classification/2>`__
   * - tensorflow-ic-efficientnet-v2-imagenet21k-ft1k-l
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_l/classification/2>`__
   * - tensorflow-ic-efficientnet-v2-imagenet21k-ft1k-m
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_m/classification/2>`__
   * - tensorflow-ic-efficientnet-v2-imagenet21k-ft1k-s
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_s/classification/2>`__
   * - tensorflow-ic-efficientnet-v2-imagenet21k-ft1k-xl
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_xl/classification/2>`__
   * - tensorflow-ic-efficientnet-v2-imagenet21k-l
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_l/classification/2>`__
   * - tensorflow-ic-efficientnet-v2-imagenet21k-m
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_m/classification/2>`__
   * - tensorflow-ic-efficientnet-v2-imagenet21k-s
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_s/classification/2>`__
   * - tensorflow-ic-efficientnet-v2-imagenet21k-xl
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_xl/classification/2>`__
   * - tensorflow-ic-imagenet-inception-resnet-v2-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/5>`__
   * - tensorflow-ic-imagenet-inception-v1-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/inception_v1/classification/5>`__
   * - tensorflow-ic-imagenet-inception-v2-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/inception_v2/classification/5>`__
   * - tensorflow-ic-imagenet-inception-v3-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/inception_v3/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v1-025-128-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_128/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v1-025-160-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_160/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v1-025-192-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_192/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v1-025-224-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v1-050-128-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_128/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v1-050-160-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v1-050-192-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_192/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v1-050-224-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_050_224/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v1-075-128-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_128/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v1-075-160-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_160/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v1-075-192-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_192/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v1-075-224-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_075_224/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v1-100-128-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_128/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v1-100-160-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_160/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v1-100-192-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_192/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v1-100-224-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v2-035-128
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_035_128/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v2-035-160
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_035_160/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v2-035-192
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_035_192/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v2-035-224-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_035_224/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v2-035-96
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_035_96/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v2-050-128
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_050_128/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v2-050-160
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_050_160/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v2-050-192
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_050_192/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v2-050-224-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_050_224/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v2-050-96
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_050_96/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v2-075-128
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_075_128/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v2-075-160
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_075_160/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v2-075-192
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_075_192/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v2-075-224-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_075_224/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v2-075-96
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_075_96/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v2-100-160
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_100_160/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v2-100-192
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_100_192/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v2-100-224-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v2-100-96
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v2-130-224-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v2-140-224-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v3-large-075-224
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v3-large-100-224
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v3-small-075-224
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v3_small_075_224/classification/5>`__
   * - tensorflow-ic-imagenet-mobilenet-v3-small-100-224
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/classification/5>`__
   * - tensorflow-ic-imagenet-nasnet-large
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/nasnet_large/classification/5>`__
   * - tensorflow-ic-imagenet-nasnet-mobile
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/nasnet_mobile/classification/5>`__
   * - tensorflow-ic-imagenet-pnasnet-large
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/pnasnet_large/classification/5>`__
   * - tensorflow-ic-imagenet-resnet-v1-101-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v1_101/classification/5>`__
   * - tensorflow-ic-imagenet-resnet-v1-152-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v1_152/classification/5>`__
   * - tensorflow-ic-imagenet-resnet-v1-50-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v1_50/classification/5>`__
   * - tensorflow-ic-imagenet-resnet-v2-101-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v2_101/classification/5>`__
   * - tensorflow-ic-imagenet-resnet-v2-152-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v2_152/classification/5>`__
   * - tensorflow-ic-imagenet-resnet-v2-50-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/imagenet/resnet_v2_50/classification/5>`__
   * - tensorflow-ic-resnet-50-classification-1
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/tensorflow/resnet_50/classification/1>`__
   * - tensorflow-ic-swin-base-patch4-window12-384
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/swin_base_patch4_window12_384/1>`__
   * - tensorflow-ic-swin-base-patch4-window7-224
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/swin_base_patch4_window7_224/1>`__
   * - tensorflow-ic-swin-large-patch4-window12-384
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/swin_large_patch4_window12_384/1>`__
   * - tensorflow-ic-swin-large-patch4-window7-224
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/swin_large_patch4_window7_224/1>`__
   * - tensorflow-ic-swin-s3-base-224
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/swin_s3_base_224/1>`__
   * - tensorflow-ic-swin-s3-small-224
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/swin_s3_small_224/1>`__
   * - tensorflow-ic-swin-s3-tiny-224
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/swin_s3_tiny_224/1>`__
   * - tensorflow-ic-swin-small-patch4-window7-224
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/swin_small_patch4_window7_224/1>`__
   * - tensorflow-ic-swin-tiny-patch4-window7-224
     - True
     - 1.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/sayakpaul/swin_tiny_patch4_window7_224/1>`__
   * - tensorflow-ic-tf2-preview-inception-v3-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/tf2-preview/inception_v3/classification/4>`__
   * - tensorflow-ic-tf2-preview-mobilenet-v2-classification-4
     - True
     - 3.0.2
     - 2.80.0
     - `Tensorflow Hub <https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4>`__
