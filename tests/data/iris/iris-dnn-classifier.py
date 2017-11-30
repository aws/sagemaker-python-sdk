import numpy as np
import os
import tensorflow as tf


def estimator_fn(run_config, hyperparameters):
    input_tensor_name = hyperparameters['input_tensor_name']
    feature_columns = [tf.feature_column.numeric_column(input_tensor_name, shape=[4])]
    return tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                      hidden_units=[10, 20, 10],
                                      n_classes=3,
                                      config=run_config)


def serving_input_fn(hyperparameters):
    input_tensor_name = hyperparameters['input_tensor_name']
    feature_spec = {input_tensor_name: tf.FixedLenFeature(dtype=tf.float32, shape=[4])}
    return tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)()


def train_input_fn(training_dir, hyperparameters):
    """Returns input function that would feed the model during training"""
    return _generate_input_fn(training_dir, 'iris_training.csv', hyperparameters)


def eval_input_fn(training_dir, hyperparameters):
    """Returns input function that would feed the model during evaluation"""
    return _generate_input_fn(training_dir, 'iris_test.csv', hyperparameters)


def _generate_input_fn(training_dir, training_filename, hyperparameters):
    input_tensor_name = hyperparameters['input_tensor_name']

    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=os.path.join(training_dir, training_filename),
        target_dtype=np.int,
        features_dtype=np.float32)

    return tf.estimator.inputs.numpy_input_fn(
        x={input_tensor_name: np.array(training_set.data)},
        y=np.array(training_set.target),
        num_epochs=None,
        shuffle=True)()
