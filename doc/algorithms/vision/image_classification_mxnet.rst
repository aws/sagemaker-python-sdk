#############################
Image Classification - MxNet
#############################

The Amazon SageMaker image classification algorithm is a supervised learning algorithm that supports multi-label classification. It takes an image as input and outputs one or more labels assigned to that image.
It uses a convolutional neural network that can be trained from scratch or trained using transfer learning when a large number of training images are not available.

The recommended input format for the Amazon SageMaker image classification algorithms is Apache MXNet `RecordIO <https://mxnet.apache.org/versions/1.9.1/api/faq/recordio.html>`__.
However, you can also use raw images in .jpg or .png format. Refer to `this discussion <https://mxnet.apache.org/versions/1.9.1/api/architecture/note_data_loading.html>`__ for a broad overview of efficient
data preparation and loading for machine learning systems.

For a sample notebook that uses the SageMaker image classification algorithm to train a model on the caltech-256 dataset and then to deploy it to perform inferences, see the
`End-to-End Multiclass Image Classification Example <https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_amazon_algorithms/imageclassification_caltech/Image-classification-fulltraining.html>`__.
For instructions how to create and access Jupyter notebook instances that you can use to run the example in SageMaker, see `Use Amazon SageMaker Notebook Instances <https://docs.aws.amazon.com/sagemaker/latest/dg/nbi.html>`__.
Once you have created a notebook instance and opened it, select the SageMaker Examples tab to see a list of all the SageMaker samples. The example image classification notebooks are located in the Introduction to Amazon
algorithms section. To open a notebook, click on its Use tab and select Create copy.

For detailed documentation, please refer to the `Sagemaker Image Classification Algorithm <https://docs.aws.amazon.com/sagemaker/latest/dg/image-classification.html>`__
