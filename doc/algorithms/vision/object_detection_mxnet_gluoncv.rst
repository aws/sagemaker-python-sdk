##################################
Object Detection - MxNet GluonCV
##################################


The Amazon SageMaker Object Detection algorithm detects and classifies objects in images using a single deep neural network.
It is a supervised learning algorithm that takes images as input and identifies all instances of objects within the image scene.
The object is categorized into one of the classes in a specified collection with a confidence score that it belongs to the class.
Its location and scale in the image are indicated by a rectangular bounding box. It uses the `Single Shot multibox Detector (SSD) <https://arxiv.org/pdf/1512.02325.pdf>`__
framework and supports two base networks: `VGG <https://arxiv.org/pdf/1409.1556.pdf>`__ and `ResNet <https://arxiv.org/pdf/1603.05027.pdf>`__. The network can be trained from scratch,
or trained with models that have been pre-trained on the `ImageNet <https://image-net.org/>`__ dataset.

For a sample notebook that shows how to use the SageMaker Object Detection algorithm to train and host a model on the `Caltech Birds (CUB 200 2011) <http://www.vision.caltech.edu/visipedia/CUB-200-2011.html>`__
dataset using the Single Shot multibox Detector algorithm, see `Amazon SageMaker Object Detection for Bird Species <https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_amazon_algorithms/object_detection_birds/object_detection_birds.html>`__.
For instructions how to create and access Jupyter notebook instances that you can use to run the example in SageMaker, see `Use Amazon SageMaker Notebook Instances <https://docs.aws.amazon.com/sagemaker/latest/dg/nbi.html>`__.
Once you have created a notebook instance and opened it, select the SageMaker Examples tab to see a list of all the SageMaker samples. The object detection example notebook using the Object Detection
algorithm is located in the Introduction to Amazon Algorithms section. To open a notebook, click on its Use tab and select Create copy.

For detailed documentation, please refer to the `Sagemaker Object Detection Algorithm <https://docs.aws.amazon.com/sagemaker/latest/dg/object-detection.html>`__
