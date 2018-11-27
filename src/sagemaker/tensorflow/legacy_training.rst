Preparing a Legacy Mode training script
=======================================

For Legacy Mode your TensorFlow training script must be a Python 2.7 source file. The SageMaker TensorFlow
Docker image uses this script by calling specifically-named functions from this script.

The training script **must contain** the following:

- Exactly one of the following:

  - ``model_fn``: defines the model that will be trained.
  - ``keras_model_fn``: defines the ``tf.keras`` model that will be trained.
  - ``estimator_fn``: defines the ``tf.estimator.Estimator`` that will train the model.

- ``train_input_fn``: preprocess and load training data.
- ``eval_input_fn``: preprocess and load evaluation data.

In addition, it may optionally contain:

- ``serving_input_fn``: Defines the features to be passed to the model during prediction.
    **Important:** this function is used only during training, but is required to deploy the model resulting from
    training to a SageMaker endpoint.

Creating a ``model_fn``
~~~~~~~~~~~~~~~~~~~~~~~

A ``model_fn`` is a function that contains all the logic to support training, evaluation,
and prediction. The basic skeleton for a ``model_fn`` looks like this:

.. code:: python

  def model_fn(features, labels, mode, hyperparameters):
    # Logic to do the following:
    # 1. Configure the model via TensorFlow operations
    # 2. Define the loss function for training/evaluation
    # 3. Define the training operation/optimizer
    # 4. Generate predictions
    # 5. Return predictions/loss/train_op/eval_metric_ops in EstimatorSpec object
    return EstimatorSpec(mode, predictions, loss, train_op, eval_metric_ops)

The ``model_fn`` must accept four positional arguments:

- ``features``: A dict containing the features passed to the model via ``train_input_fn``
  in **training** mode, via ``eval_input_fn`` in **evaluation** mode, and via ``serving_input_fn``
  in **predict** mode.
- ``labels``: A ``Tensor`` containing the labels passed to the model via ``train_input_fn``
  in **training** mode and ``eval_input_fn`` in **evaluation** mode. It will be empty for
  **predict** mode.
- ``mode``: One of the following ``tf.estimator.ModeKeys`` string values indicating the
  context in which the ``model_fn`` was invoked:
  - ``TRAIN``: the ``model_fn`` was invoked in **training** mode.
  - ``EVAL``: the ``model_fn`` was invoked in **evaluation** mode.
  - ``PREDICT``: the ``model_fn`` was invoked in **predict** mode.
- ``hyperparameters``: The hyperparameters passed to SageMaker TrainingJob that runs
  your TensorFlow training script. You can use this to pass hyperparameters to your
  training script.

The ``model_fn`` must return a ``tf.estimator.EstimatorSpec``.

Example of a complete ``model_fn``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

  def model_fn(features, labels, mode, hyperparameters):
    # Connect the first hidden layer to input layer
    # (features["x"]) with relu activation
    first_hidden_layer = Dense(10, activation='relu', name='first-layer')(features[INPUT_TENSOR_NAME])

    # Connect the second hidden layer to first hidden layer with relu
    second_hidden_layer = Dense(20, activation='relu')(first_hidden_layer)

    # Connect the output layer to second hidden layer (no activation fn)
    output_layer = Dense(1, activation='linear')(second_hidden_layer)

    # Reshape output layer to 1-dim Tensor to return predictions
    predictions = tf.reshape(output_layer, [-1])

    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions={"ages": predictions})

    # Calculate loss using mean squared error
    loss = tf.losses.mean_squared_error(labels, predictions)

    # Calculate root mean squared error as additional eval metric
    eval_metric_ops = {
        "rmse": tf.metrics.root_mean_squared_error(tf.cast(labels, tf.float64), predictions)
    }

    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=hyperparameters["learning_rate"])
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)

Distributed training
^^^^^^^^^^^^^^^^^^^^

When distributed training happens, a copy of the same neural network will be sent to
multiple training instances. Each instance will train with a batch of the dataset,
calculate loss and minimize the optimizer. One entire loop of this process is called training step.

A `global step <https://www.tensorflow.org/api_docs/python/tf/train/global_step>`_ is a global
counter shared between the instances. It is necessary for distributed training, so the optimizer
can keep track of the number of training steps across instances. The only change in the
previous complete ``model_fn`` to enable distributed training is to pass in the global
step into the ``optimizer.minimize`` function:

.. code:: python

  train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())

More information about distributed training can be find in talk from the TensorFlow Dev Summit 2017:
`Distributed TensorFlow <https://www.youtube.com/watch?time_continue=1&v=la_M6bCV91M>`_.


More details on how to create a ``model_fn`` can be find in `Constructing the model_fn <https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/docs_src/extend/estimators.md#constructing-the-model_fn-constructing-modelfn>`_.


Creating ``train_input_fn`` and ``eval_input_fn`` functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``train_input_fn`` is used to pass ``features`` and ``labels`` to the ``model_fn``
in **training** mode. The ``eval_input_fn`` is used to ``features`` and ``labels`` to the
``model_fn`` in **evaluation** mode.

The basic skeleton for the ``train_input_fn`` looks like this:

.. code:: python

  def train_input_fn(training_dir, hyperparameters):
    # Logic to the following:
    # 1. Reads the **training** dataset files located in training_dir
    # 2. Preprocess the dataset
    # 3. Return 1)  a dict of feature names to Tensors with
    # the corresponding feature data, and 2) a Tensor containing labels
    return features, labels

An ``eval_input_fn`` follows the same format:

.. code:: python

  def eval_input_fn(training_dir, hyperparameters):
    # Logic to the following:
    # 1. Reads the **evaluation** dataset files located in training_dir
    # 2. Preprocess the dataset
    # 3. Return 1)  a dict of feature names to Tensors with
    # the corresponding feature data, and 2) a Tensor containing labels
    return features, labels

**Note:** For TensorFlow 1.4 and 1.5, ``train_input_fn`` and ``eval_input_fn`` may also return a no-argument
function which returns the tuple ``(features, labels)``. This is no longer supported for TensorFlow 1.6 and up.

Example of a complete ``train_input_fn`` and ``eval_input_fn``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

  def train_input_fn(training_dir, hyperparameters):
    # invokes _input_fn with training dataset
    return _input_fn(training_dir, 'training_dataset.csv')

  def eval_input_fn(training_dir, hyperparameters):
    # invokes _input_fn with evaluation dataset
    return _input_fn(training_dir, 'evaluation_dataset.csv')

  def _input_fn(training_dir, training_filename):
      # reads the dataset using tf.dataset API
      training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
          filename=os.path.join(training_dir, training_filename), target_dtype=np.int, features_dtype=np.float32)

      # returns features x and labels y
      return tf.estimator.inputs.numpy_input_fn(
          x={INPUT_TENSOR_NAME: np.array(training_set.data)},
          y=np.array(training_set.target),
          num_epochs=None,
          shuffle=True)()


More details on how to create input functions can be find in `Building Input Functions with tf.estimator <https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/docs_src/get_started/input_fn.md#building-input-functions-with-tfestimator>`_.

Creating a ``serving_input_fn``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``serving_input_fn`` is used to define the shapes and types of the inputs the model accepts when
the model is exported for Tensorflow Serving. This function is optional if you only want to
train a model, but it is required if you want to create a SavedModel bundle that can be
deployed to a SageMaker endpoint.

``serving_input_fn`` is called at the end of model training and is **not** called during
inference. (If you'd like to preprocess inference data, please see
`Overriding input preprocessing with an input_fn <https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/tensorflow/deploying_python.rst#overriding-input-preprocessing-with-an-input_fn>`_ ).

The basic skeleton for the ``serving_input_fn`` looks like this:

.. code:: python

  def serving_input_fn(hyperparameters):
    # Logic to the following:
    # 1. Defines placeholders that TensorFlow serving will feed with inference requests
    # 2. Preprocess input data
    # 3. Returns a tf.estimator.export.ServingInputReceiver or tf.estimator.export.TensorServingInputReceiver,
    # which packages the placeholders and the resulting feature Tensors together.

**Note:** For TensorFlow 1.4 and 1.5, ``serving_input_fn`` may also return a no-argument function which returns a ``tf.estimator.export.ServingInputReceiver`` or``tf.estimator.export.TensorServingInputReceiver``. This is no longer supported for TensorFlow 1.6 and up.

Example of a complete ``serving_input_fn``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

  def serving_input_fn(hyperparameters):
      # defines the input placeholder
      tensor = tf.placeholder(tf.float32, shape=[1, 7])
      # returns the ServingInputReceiver object.
      return build_raw_serving_input_receiver_fn({INPUT_TENSOR_NAME: tensor})()

More details on how to create a `serving_input_fn` can be find in `Preparing serving inputs <https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/docs_src/programmers_guide/saved_model.md#preparing-serving-inputs>`_.

The complete example described above can find in `Abalone age predictor using layers notebook example <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/tensorflow_abalone_age_predictor_using_layers/tensorflow_abalone_age_predictor_using_layers.ipynb>`_.

More examples on how to create a TensorFlow training script can be find in the `Amazon SageMaker examples repository <https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-python-sdk>`_.

Support for pre-made ``tf.estimator`` and ``Keras`` models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to ``model_fn``, ``sagemaker.tensorflow.TensorFlow`` supports pre-canned ``tf.estimator``
and Keras models.

Using a pre-made ``tensorflow.estimator`` instead of a ``model_fn``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pre-canned estimators are machine learning estimators premade for general purpose problems.
``tf.estimator`` provides the following pre-canned estimators:

- `tf.estimator.LinearClassifier <https://www.tensorflow.org/api_docs/python/tf/estimator/LinearClassifier>`_: Constructs
  a linear classification model.
- `tf.estimator.LinearRegressor <https://www.tensorflow.org/api_docs/python/tf/estimator/LinearRegressor>`_: Constructs
  a linear regression model.
- `tf.estimator.DNNClassifier <https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier>`_: Constructs
  a neural network classification model.
- `tf.estimator.DNNRegressor <https://www.tensorflow.org/api_docs/python/tf/estimator/DNNRegressor>`_: Construct
  a neural network regression model.
- `tf.estimator.DNNLinearCombinedClassifier <https://www.tensorflow.org/api_docs/python/tf/estimator/DNNLinearCombinedClassifier>`_: Constructs
  a neural network and linear combined classification model.
- `tf.estimator.DNNLinearCombinedRegressor <https://www.tensorflow.org/api_docs/python/tf/estimator/DNNLinearCombinedRegressor>`_: Constructs
  a neural network and linear combined regression model.

To use a pre-canned ``tensorflow.estimator`` instead of creating a ``model_fn``, you need to write a ``estimator_fn``.
The base skeleton for the ``estimator_fn`` looks like this:

.. code:: python

  def estimator_fn(run_config, hyperparameters):
    # Logic to the following:
    # 1. Defines the features columns that will be the input of the estimator
    # 2. Returns an instance of a ``tensorflow.estimator`` passing in, the input run_config in the
    #    constructor.

Example of a complete ``estimator_fn``
''''''''''''''''''''''''''''''''''''''

.. code:: python

  def estimator_fn(run_config, hyperparameters):
      # Defines the features columns that will be the input of the estimator
      feature_columns = [tf.feature_column.numeric_column(INPUT_TENSOR_NAME, shape=[4])]
      # Returns the instance of estimator.
      return tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                        hidden_units=[10, 20, 10],
                                        n_classes=3,
                                        config=run_config)

More details on how to create a ``tensorflow.estimator`` can be find in `Creating Estimators in tf.estimator <https://www.tensorflow.org/extend/estimators>`_.

An example on how to create a TensorFlow training script with an ``estimator_fn`` can find in this `example <https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-python-sdk/tensorflow_iris_dnn_classifier_using_estimators>`_.


Using a ``Keras`` model instead of a ``model_fn``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``tf.keras`` is an full implementation inside TensorFlow of the Keras API. To use a ``tf.keras``
model for training instead of ``model_fn``, you need to write a ``keras_model_fn``. The base skeleton of
a ``keras_model_fn`` looks like this:

.. code:: python

  def keras_model_fn(hyperparameters):
      # Logic to do the following:
      # 1. Instantiate the Keras model
      # 2. Compile the Keras model
      return compiled_model


Example of a complete ``keras_model_fn``
''''''''''''''''''''''''''''''''''''''''

.. code:: python

  def keras_model_fn(hyperparameters):
    # Instantiate a Keras inception v3 model.
    keras_inception_v3 = tf.keras.applications.inception_v3.InceptionV3(weights=None)
    # Compile model with the optimizer, loss, and metrics you'd like to train with.
    keras_inception_v3.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),
                          loss='categorical_crossentropy', metric='accuracy')
    return keras_inception_v3


TensorFlow 1.4.0 support for Keras models is limited only for **single-node** training;
i.e. you must set ``train_instance_count=1`` when constructing your TensorFlow estimator.

More details on how to create a Keras model can be find in the `Keras documentation <https://keras.io/>`_.

Optional Hyperparameters
''''''''''''''''''''''''

These hyperparameters are used by TensorFlow to fine tune the training.
You need to add them inside the hyperparameters dictionary in the
``TensorFlow`` estimator constructor.

**All versions**

-  ``save_summary_steps (int)``: Save summaries every this many steps.
-  ``save_checkpoints_secs (int)``: Save checkpoints every this many seconds. Can not be specified with ``save_checkpoints_steps``.
-  ``save_checkpoints_steps (int)``: Save checkpoints every this many steps. Can not be specified with ``save_checkpoints_secs``.
-  ``keep_checkpoint_max (int)``: The maximum number of recent checkpoint files to keep. As new files are created, older files are deleted. If None or 0, all checkpoint files are kept. Defaults to 5 (that is, the 5 most recent checkpoint files are kept.)
-  ``keep_checkpoint_every_n_hours (int)``: Number of hours between each checkpoint to be saved. The default value of 10,000 hours effectively disables the feature.
-  ``log_step_count_steps (int)``: The frequency, in number of global steps, that the global step/sec will be logged during training.

**TensorFlow 1.6 and up**

- ``start_delay_secs (int)``: See docs for this parameter in `tf.estimator.EvalSpec <https://www.tensorflow.org/api_docs/python/tf/estimator/EvalSpec>`_.
- ``throttle_secs (int)``: See docs for this parameter in `tf.estimator.EvalSpec <https://www.tensorflow.org/api_docs/python/tf/estimator/EvalSpec>`_.

**TensorFlow 1.4 and 1.5**

-  ``eval_metrics (dict)``: ``dict`` of string, metric function. If `None`, default set is used. This should be ``None`` if the ``estimator`` is `tf.estimator.Estimator <https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator>`_. If metrics are provided they will be *appended* to the default set.
-  ``eval_delay_secs (int)``: Start evaluating after waiting for this many seconds.
-  ``continuous_eval_throttle_secs (int)``: Do not re-evaluate unless the last evaluation was started at least this many seconds ago.
-  ``min_eval_frequency (int)``: The minimum number of steps between evaluations. Of course, evaluation does not occur if no new snapshot is available, hence, this is the minimum. If 0, the evaluation will only happen after training. If None, defaults to 1000.
-  ``delay_workers_by_global_step (bool)``: if ``True`` delays training workers based on global step instead of time.
- ``train_steps_per_iteration (int)``: Perform this many (integer) number of train steps for each training-evaluation iteration. With a small value, the model will be evaluated more frequently with more checkpoints saved.

Calling fit
~~~~~~~~~~~

You start your training script by calling ``fit`` on a ``TensorFlow`` estimator. ``fit`` takes
both required and optional arguments.

Required argument
^^^^^^^^^^^^^^^^^

- ``inputs``: The S3 location(s) of datasets to be used for training. This can take one of two forms:

  - ``str``: An S3 URI, for example ``s3://my-bucket/my-training-data``, which indicates the dataset's location.
  - ``dict[str, str]``: A dictionary mapping channel names to S3 locations, for example ``{'train': 's3://my-bucket/my-training-data/train', 'test': 's3://my-bucket/my-training-data/test'}``
  - ``sagemaker.session.s3_input``: channel configuration for S3 data sources that can provide additional information as well as the path to the training dataset. See `the API docs <https://sagemaker.readthedocs.io/en/latest/session.html#sagemaker.session.s3_input>`_ for full details.

When the training job starts in SageMaker the container will download the dataset.
Both ``train_input_fn`` and ``eval_input_fn`` functions have a parameter called ``training_dir`` which
contains the directory inside the container where the dataset was saved into.
See `Creating train_input_fn and eval_input_fn functions`_.

Optional arguments
^^^^^^^^^^^^^^^^^^

-  ``wait (bool)``: Defaults to True, whether to block and wait for the
   training script to complete before returning.
   If set to False, it will return immediately, and can later be attached to.
-  ``logs (bool)``: Defaults to True, whether to show logs produced by training
   job in the Python session. Only meaningful when wait is True.
- ``run_tensorboard_locally (bool)``: Defaults to False. Executes TensorBoard in a different
  process with downloaded checkpoint information. Requires modules TensorBoard and AWS CLI.
  installed. Terminates TensorBoard when the execution ends. See `Running TensorBoard`_.
- ``job_name (str)``: Training job name. If not specified, the estimator generates a default job name,
  based on the training image name and current timestamp.

What happens when fit is called
'''''''''''''''''''''''''''''''

Calling ``fit`` starts a SageMaker training job. The training job will execute the following.

- Starts ``train_instance_count`` EC2 instances of the type ``train_instance_type``.
- On each instance, it will do the following steps:

  - starts a Docker container optimized for TensorFlow.
  - downloads the dataset.
  - setup up distributed training.
  - starts asynchronous training, executing the ``model_fn`` function defined in your script
    in **training** mode; i.e., ``features`` and ``labels`` are fed by a batch of the
    training dataset defined by ``train_input_fn``. See `Creating train_input_fn and eval_input_fn functions`_.

The training job finishes after the number of training steps reaches the value defined by
the ``TensorFlow`` estimator parameter ``training_steps`` is finished or when the training
job execution time reaches the ``TensorFlow`` estimator parameter ``train_max_run``.

When the training job finishes, a `TensorFlow serving <https://www.tensorflow.org/serving/serving_basic>`_
with the result of the training is generated and saved to the S3 location defined by
the ``TensorFlow`` estimator parameter ``output_path``.

If the ``wait=False`` flag is passed to ``fit``, then it will return immediately. The training job will continue running
asynchronously. At a later time, a Tensorflow Estimator can be obtained by attaching to the existing training job. If
the training job is not finished it will start showing the standard output of training and wait until it completes.
After attaching, the estimator can be deployed as usual.

.. code:: python

    tf_estimator.fit(your_input_data, wait=False)
    training_job_name = tf_estimator.latest_training_job.name

    # after some time, or in a separate python notebook, we can attach to it again.

    tf_estimator = TensorFlow.attach(training_job_name=training_job_name)


The evaluation process
''''''''''''''''''''''

During the training job, the first EC2 instance that is executing the training is named ``master``. All the other instances are called ``workers``.

All instances execute the training loop, feeding the ``model_fn`` with ``train_input_fn``.
Every ``min_eval_frequency`` steps (see `Optional Hyperparameters`_), the ``master`` instance
will execute the ``model_fn`` in **evaluation** mode; i.e., ``features`` and ``labels`` are
fed with the evaluation dataset defined by ``eval_input_fn``. See `Creating train_input_fn and eval_input_fn functions`_.

For more information on training and evaluation process, see `tf.estimator.train_and_evaluate <https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/estimator/training.py#L256>`_.

For more information on fit, see `SageMaker Python SDK Overview <#sagemaker-python-sdk-overview>`_.

TensorFlow Serving models
~~~~~~~~~~~~~~~~~~~~~~~~~

After your training job is complete in SageMaker and the ``fit`` call ends, the training job
will generate a `TensorFlow SavedModel <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md>`_
bundle ready for deployment. Your model will be available in S3 at the ``output_path`` location
that you specified when you created your ``sagemaker.tensorflow.TensorFlow`` estimator.

Restoring from checkpoints
~~~~~~~~~~~~~~~~~~~~~~~~~~

While your training job is executing, TensorFlow will generate checkpoints and save them in the S3
location defined by ``checkpoint_path`` parameter in the ``TensorFlow`` constructor.
These checkpoints can be used to restore a previous session or to evaluate the current training using ``TensorBoard``.

To restore a previous session, you just need to create a new ``sagemaker.tensorflow.TensorFlow``
estimator pointing to the previous checkpoint path:

.. code:: python

  previous_checkpoint_path = 's3://location/of/my/previous/generated/checkpoints'

  tf_estimator = TensorFlow('tf-train.py', role='SageMakerRole',
                            checkpoint_path=previous_checkpoint_path
                            training_steps=10000, evaluation_steps=100,
                            train_instance_count=1, train_instance_type='ml.p2.xlarge',
                            framework_version='1.10.0')
  tf_estimator.fit('s3://bucket/path/to/training/data')


Running TensorBoard
~~~~~~~~~~~~~~~~~~~

When the ``fit`` parameter ``run_tensorboard_locally`` is set ``True``, all the checkpoint data
located in ``checkpoint_path`` will be downloaded to a local temporary folder and a local
``TensorBoard`` application will be watching that temporary folder.
Every time a new checkpoint is created by the training job in the S3 bucket, ``fit`` will download that checkpoint to the same temporary folder and update ``TensorBoard``.

When the ``fit`` method starts the training, it will log the port that ``TensorBoard`` is using
to display metrics. The default port is **6006**, but another port can be chosen depending on
availability. The port number will increase until finds an available port. After that, the port
number will be printed in stdout.

It takes a few minutes to provision containers and start the training job. TensorBoard will start to display metrics shortly after that.

You can access TensorBoard locally at http://localhost:6006 or using your SakeMaker workspace at
`https*workspace_base_url*proxy/6006/ <proxy/6006/>`_ (TensorBoard will not work if you forget to put the slash,
'/', in end of the url). If TensorBoard started on a different port, adjust these URLs to match.

Note that TensorBoard is not supported when passing ``wait=False`` to ``fit``.
