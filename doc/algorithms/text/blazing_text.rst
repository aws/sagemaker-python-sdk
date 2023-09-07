#############
Blazing Text
#############


The Amazon SageMaker BlazingText algorithm provides highly optimized implementations of the Word2vec and text classification algorithms. The Word2vec algorithm is useful for many downstream natural language processing (NLP)
tasks, such as sentiment analysis, named entity recognition, machine translation, etc. Text classification is an important task for applications that perform web searches, information retrieval, ranking, and document classification.

The Word2vec algorithm maps words to high-quality distributed vectors. The resulting vector representation of a word is called a word embedding. Words that are semantically similar correspond to vectors that are close together.
That way, word embeddings capture the semantic relationships between words.

Many natural language processing (NLP) applications learn word embeddings by training on large collections of documents. These pretrained vector representations provide information about semantics and word distributions that
typically improves the generalizability of other models that are later trained on a more limited amount of data. Most implementations of the Word2vec algorithm are not optimized for multi-core CPU architectures. This makes it
difficult to scale to large datasets.

With the BlazingText algorithm, you can scale to large datasets easily. Similar to Word2vec, it provides the Skip-gram and continuous bag-of-words (CBOW) training architectures. BlazingText's implementation of the supervised
multi-class, multi-label text classification algorithm extends the fastText text classifier to use GPU acceleration with custom `CUDA <https://docs.nvidia.com/cuda/index.html>`__

kernels. You can train a model on more than a billion words in a couple of minutes using a multi-core CPU or a GPU. And, you achieve performance on par with the state-of-the-art deep learning text classification algorithms.

The BlazingText algorithm is not parallelizable. For more information on parameters related to training, see `Docker Registry Paths for SageMaker Built-in Algorithms <https://docs.aws.amazon.com/en_us/sagemaker/latest/dg/sagemaker-algo-docker-registry-paths.html>`__.

For a sample notebook that uses the SageMaker BlazingText algorithm to train and deploy supervised binary and multiclass classification models, see
`Blazing Text classification on the DBPedia dataset <https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_amazon_algorithms/blazingtext_text_classification_dbpedia/blazingtext_text_classification_dbpedia.html>`__.
For instructions for creating and accessing Jupyter notebook instances that you can use to run the example in SageMaker, see `Use Amazon SageMaker Notebook Instances <https://docs.aws.amazon.com/sagemaker/latest/dg/nbi.html>`__.
After creating and opening a notebook instance, choose the SageMaker Examples tab to see a list of all the SageMaker examples. The topic modeling example notebooks that use the Blazing Text are located in the Introduction to Amazon
algorithms section. To open a notebook, choose its Use tab, then choose Create copy.
