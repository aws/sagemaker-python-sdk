from __future__ import absolute_import

import os
from datetime import timedelta

import airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.contrib.operators.sagemaker_training_operator import SageMakerTrainingOperator
from airflow.contrib.operators.sagemaker_transform_operator import SageMakerTransformOperator
import sagemaker
from sagemaker import Session
from sagemaker.tensorflow import TensorFlow
from sagemaker.workflow.airflow import training_config, transform_config_from_estimator
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist

default_args = {
    'owner': 'airflow',
    'start_date': airflow.utils.dates.days_ago(2),
    'provide_context': True,
    'retry_delay': timedelta(minutes=1)
}

dag = DAG('tensorflow_training_transform', default_args=default_args,
          schedule_interval='@once')

# Constants
role = 'SageMakerRole'


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# Convert MNIST data to TFRecords file format with Example protos
def convert_to(data_set, name, directory):
    """Converts a dataset to tfrecords."""
    images = data_set.images
    labels = data_set.labels
    num_examples = data_set.num_examples

    if images.shape[0] != num_examples:
        raise ValueError('Images size %d does not match label size %d.' %
                         (images.shape[0], num_examples))
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    filename = os.path.join(directory, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()


def prepare_data(**context):
    sagemaker_session = Session()

    # Download the training data
    data_sets = mnist.read_data_sets('data', dtype=tf.uint8, reshape=False, validation_size=5000)

    # Convert data format
    convert_to(data_sets.train, 'train', 'data')
    convert_to(data_sets.validation, 'validation', 'data')
    convert_to(data_sets.test, 'test', 'data')

    # Upload the training data
    training_data = sagemaker_session.upload_data(path='data', key_prefix='data/DEMO-mnist')

    # Use data that contains 1000 MNIST images in public SageMaker sample data bucket
    region = sagemaker_session.boto_region_name
    transform_data = "s3://sagemaker-sample-data-{}/batch-transform/mnist-1000-samples".format(region)

    # Store data URIs in XCOM
    return {
        'training_data': training_data,
        'transform_data': transform_data
    }


prepare = PythonOperator(
    task_id='prepare_data',
    python_callable=prepare_data,
    provide_context=True,
    retries=3,
    dag=dag)

training_data = "{{ ti.xcom_pull(task_ids='prepare_data')['training_data'] }}"

# You need to put the training script at ./scripts/tf_mnist.py in AIRFLOW_HOME.
mnist_estimator = TensorFlow(entry_point='tf_mnist.py',
                             role=role,
                             framework_version='1.11.0',
                             training_steps=1000,
                             evaluation_steps=100,
                             train_instance_count=2,
                             train_instance_type='ml.c4.xlarge')

train_config = training_config(estimator=mnist_estimator,
                               inputs=training_data)

tf_training = SageMakerTrainingOperator(
    task_id='tf_training',
    config=train_config,
    retries=3,
    dag=dag)

tf_training.set_upstream(prepare)

transform_data = "{{ ti.xcom_pull(task_ids='prepare_data')['transform_data'] }}"

trans_config = transform_config_from_estimator(estimator=mnist_estimator,
                                               task_id='tf_training',
                                               task_type='training',
                                               instance_count=1,
                                               instance_type='ml.m4.xlarge',
                                               data=transform_data,
                                               content_type='text/csv')

tf_transform = SageMakerTransformOperator(
    task_id='tf_transform',
    config=trans_config,
    retries=3,
    dag=dag)

tf_transform.set_upstream(tf_training)
