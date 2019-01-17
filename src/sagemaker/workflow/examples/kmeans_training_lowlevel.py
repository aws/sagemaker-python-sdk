from __future__ import absolute_import

import io
import gzip
import pickle
from datetime import timedelta
from time import gmtime, strftime
from urllib import request

import airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.contrib.operators.sagemaker_training_operator import SageMakerTrainingOperator
import boto3
from sagemaker.session import Session
from sagemaker.amazon.common import write_numpy_to_dense_tensor
from sagemaker.amazon.amazon_estimator import get_image_uri


default_args = {
    'owner': 'airflow',
    'start_date': airflow.utils.dates.days_ago(2),
    'provide_context': True,
    'retry_delay': timedelta(minutes=1)
}

dag = DAG('kmeans_training_lowlevel', default_args=default_args,
          schedule_interval='@once')

# Constants
role = 'my_sagemaker_role'
job_name = 'kmeans-lowlevel-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())


def prepare_resources(**context):
    bucket = Session().default_bucket()

    # Load the input dataset
    request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

    data_key = 'kmeans_lowlevel_example/data'
    data_location = 's3://{}/{}'.format(bucket, data_key)

    # Convert the training data into the format required by the SageMaker KMeans algorithm
    buf = io.BytesIO()
    write_numpy_to_dense_tensor(buf, train_set[0], train_set[1])
    buf.seek(0)

    # upload the input data
    boto3.resource('s3').Bucket(bucket).Object(data_key).upload_fileobj(buf)
    print('training data uploaded to: {}'.format(data_location))

    # get image and model output location
    image = get_image_uri(boto3.Session().region_name, 'kmeans')
    output_location = 's3://{}/kmeans_example/output'.format(bucket)

    # store necessary training resources in XCOM
    return {
        'data_location': data_location,
        'image': image,
        'output_location': output_location
    }


resources = PythonOperator(
    task_id='prepare_resources',
    python_callable=prepare_resources,
    provide_context=True,
    dag=dag
)

# build the training config
# See: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker.html#SageMaker.Client.create_training_job # noqa: E501
kmeans_training_config = {
    "AlgorithmSpecification": {
        "TrainingImage": "{{ task_instance.xcom_pull(task_ids='prepare_resources')['image'] }}",
        "TrainingInputMode": "File"
    },
    "RoleArn": role,
    "OutputDataConfig": {
        "S3OutputPath": "{{ task_instance.xcom_pull(task_ids='prepare_resources')['output_location'] }}"
    },
    "ResourceConfig": {
        "InstanceCount": 2,
        "InstanceType": "ml.c4.8xlarge",
        "VolumeSizeInGB": 50
    },
    "TrainingJobName": job_name,
    "HyperParameters": {
        "k": "10",
        "feature_dim": "784",
        "mini_batch_size": "500",
        "force_dense": "True"
    },
    "StoppingCondition": {
        "MaxRuntimeInSeconds": 60 * 60
    },
    "InputDataConfig": [
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": "{{ task_instance.xcom_pull(task_ids='prepare_resources')['data_location'] }}",
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
            "CompressionType": "None",
            "RecordWrapperType": "None"
        }
    ]
}

training = SageMakerTrainingOperator(
    task_id='kmeans_training',
    config=kmeans_training_config,
    retries=3,
    dag=dag)
training.set_upstream(resources)
