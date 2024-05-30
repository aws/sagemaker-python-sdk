###########################
Object Detection - PyTorch
###########################

This is a supervised object detection algorithm which supports fine-tuning of many pre-trained models available in Pytorch Hub. The following
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
   * - pytorch-od-nvidia-ssd
     - False
     - 2.0.0
     - 2.189.0
     - `Pytorch Hub <https://pytorch.org/hub/nvidia_deeplearningexamples_ssd/>`__
