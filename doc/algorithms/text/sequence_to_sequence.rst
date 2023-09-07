#######################
Sequence-to-Sequence
#######################

Amazon SageMaker Sequence to Sequence is a supervised learning algorithm where the input is a sequence of tokens (for example, text, audio) and the output generated is another sequence of tokens. Example applications include: machine
translation (input a sentence from one language and predict what that sentence would be in another language), text summarization (input a longer string of words and predict a shorter string of words that is a summary), speech-to-text
(audio clips converted into output sentences in tokens). Recently, problems in this domain have been successfully modeled with deep neural networks that show a significant performance boost over previous methodologies. Amazon SageMaker
seq2seq uses Recurrent Neural Networks (RNNs) and Convolutional Neural Network (CNN) models with attention as encoder-decoder architectures.

For a sample notebook that shows how to use the SageMaker Sequence to Sequence algorithm to train a English-German translation model, see
`Machine Translation English-German Example Using SageMaker Seq2Seq <https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_amazon_algorithms/seq2seq_translation_en-de/SageMaker-Seq2Seq-Translation-English-German.html>`__.
For instructions how to create and access Jupyter notebook instances that you can use to run the example in SageMaker, see `Use Amazon SageMaker Notebook Instances <https://docs.aws.amazon.com/sagemaker/latest/dg/nbi.html>`__. Once you have
created a notebook instance and opened it, select the SageMaker Examples tab to see a list of all the SageMaker samples. The topic modeling example notebooks using the NTM algorithms are located in the Introduction to Amazon algorithms section.
To open a notebook, click on its Use tab and select Create copy.
