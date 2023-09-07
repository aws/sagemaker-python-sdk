#####################################
Semantic Segmentation - MxNet GluonCV
#####################################

The SageMaker semantic segmentation algorithm provides a fine-grained, pixel-level approach to developing computer vision applications.
It tags every pixel in an image with a class label from a predefined set of classes. Tagging is fundamental for understanding scenes, which is
critical to an increasing number of computer vision applications, such as self-driving vehicles, medical imaging diagnostics, and robot sensing.

For comparison, the `SageMaker Image Classification Algorithm <https://docs.aws.amazon.com/sagemaker/latest/dg/image-classification.html>`__ is a
supervised learning algorithm that analyzes only whole images, classifying them into one of multiple output categories. The
`Object Detection Algorithm <https://docs.aws.amazon.com/sagemaker/latest/dg/object-detection.html>`__ is a supervised learning algorithm that detects and
classifies all instances of an object in an image. It indicates the location and scale of each object in the image with a rectangular bounding box.

Because the semantic segmentation algorithm classifies every pixel in an image, it also provides information about the shapes of the objects contained in the image.
The segmentation output is represented as a grayscale image, called a segmentation mask. A segmentation mask is a grayscale image with the same shape as the input image.

The SageMaker semantic segmentation algorithm is built using the `MXNet Gluon framework and the Gluon CV toolkit <https://github.com/dmlc/gluon-cv>`__
. It provides you with a choice of three built-in algorithms to train a deep neural network. You can use the `Fully-Convolutional Network (FCN) algorithm <https://arxiv.org/abs/1605.06211>`__ ,
`Pyramid Scene Parsing (PSP) algorithm <https://arxiv.org/abs/1612.01105>`__, or `DeepLabV3 <https://arxiv.org/abs/1706.05587>`__.


Each of the three algorithms has two distinct components:

* The backbone (or encoder)—A network that produces reliable activation maps of features.

* The decoder—A network that constructs the segmentation mask from the encoded activation maps.

You also have a choice of backbones for the FCN, PSP, and DeepLabV3 algorithms: `ResNet50 or ResNet101 <https://arxiv.org/abs/1512.03385>`__.
These backbones include pretrained artifacts that were originally trained on the `ImageNet <http://www.image-net.org/>`__ classification task. You can fine-tune these backbones
for segmentation using your own data. Or, you can initialize and train these networks from scratch using only your own data. The decoders are never pretrained.

To deploy the trained model for inference, use the SageMaker hosting service. During inference, you can request the segmentation mask either as a
PNG image or as a set of probabilities for each class for each pixel. You can use these masks as part of a larger pipeline that includes additional downstream image processing or other applications.


For a sample Jupyter notebook that uses the SageMaker semantic segmentation algorithm to train a model and deploy it to perform inferences, see the
`Semantic Segmentation Example <https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_amazon_algorithms/semantic_segmentation_pascalvoc/semantic_segmentation_pascalvoc.html>`__. For instructions
on how to create and access Jupyter notebook instances that you can use to run the example in SageMaker, see `Use Amazon SageMaker Notebook Instances <https://docs.aws.amazon.com/sagemaker/latest/dg/nbi.html>`__.

To see a list of all of the SageMaker samples, create and open a notebook instance, and choose the SageMaker Examples tab. The example semantic segmentation notebooks are located under
Introduction to Amazon algorithms. To open a notebook, choose its Use tab, and choose Create copy.

For detailed documentation, please refer to the `Sagemaker Semantic Segmentation Algorithm <https://docs.aws.amazon.com/sagemaker/latest/dg/semantic-segmentation.html>`__
