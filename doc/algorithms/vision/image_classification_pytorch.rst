###############################
Image Classification - PyTorch
###############################

This is a supervised image clasification algorithm which supports fine-tuning of many pre-trained models available in Pytorch Hub. The following
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
   * - pytorch-ic-alexnet
     - True
     - 2.2.4
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_alexnet/>`__
   * - pytorch-ic-densenet121
     - True
     - 2.2.4
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_densenet/>`__
   * - pytorch-ic-densenet161
     - True
     - 2.2.4
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_densenet/>`__
   * - pytorch-ic-densenet169
     - True
     - 2.2.4
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_densenet/>`__
   * - pytorch-ic-densenet201
     - True
     - 2.2.4
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_densenet/>`__
   * - pytorch-ic-googlenet
     - True
     - 2.2.4
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_googlenet/>`__
   * - pytorch-ic-mobilenet-v2
     - True
     - 2.2.4
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_mobilenet_v2/>`__
   * - pytorch-ic-resnet101
     - True
     - 2.2.4
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_resnet/>`__
   * - pytorch-ic-resnet152
     - True
     - 2.2.4
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_resnet/>`__
   * - pytorch-ic-resnet18
     - True
     - 2.2.4
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_resnet/>`__
   * - pytorch-ic-resnet34
     - True
     - 2.2.4
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_resnet/>`__
   * - pytorch-ic-resnet50
     - True
     - 2.2.4
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_resnet/>`__
   * - pytorch-ic-resnext101-32x8d
     - True
     - 2.2.4
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_resnext/>`__
   * - pytorch-ic-resnext50-32x4d
     - True
     - 2.2.4
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_resnext/>`__
   * - pytorch-ic-shufflenet-v2-x1-0
     - True
     - 2.2.4
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_shufflenet_v2/>`__
   * - pytorch-ic-squeezenet1-0
     - True
     - 2.2.4
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_squeezenet/>`__
   * - pytorch-ic-squeezenet1-1
     - True
     - 2.2.4
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_squeezenet/>`__
   * - pytorch-ic-vgg11
     - True
     - 2.2.4
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_vgg/>`__
   * - pytorch-ic-vgg11-bn
     - True
     - 2.2.4
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_vgg/>`__
   * - pytorch-ic-vgg13
     - True
     - 2.2.4
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_vgg/>`__
   * - pytorch-ic-vgg13-bn
     - True
     - 2.2.4
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_vgg/>`__
   * - pytorch-ic-vgg16
     - True
     - 2.2.4
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_vgg/>`__
   * - pytorch-ic-vgg16-bn
     - True
     - 2.2.4
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_vgg/>`__
   * - pytorch-ic-vgg19
     - True
     - 2.2.4
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_vgg/>`__
   * - pytorch-ic-vgg19-bn
     - True
     - 2.2.4
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_vgg/>`__
   * - pytorch-ic-wide-resnet101-2
     - True
     - 2.2.4
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_wide_resnet/>`__
   * - pytorch-ic-wide-resnet50-2
     - True
     - 2.2.4
     - 2.75.0
     - `Pytorch Hub <https://pytorch.org/hub/pytorch_vision_wide_resnet/>`__
