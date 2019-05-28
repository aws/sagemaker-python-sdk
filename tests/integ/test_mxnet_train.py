# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import os
import time

import numpy
import pytest

import tests.integ
from sagemaker.mxnet.estimator import MXNet
from sagemaker.mxnet.model import MXNetModel
from sagemaker.utils import sagemaker_timestamp
from tests.integ import DATA_DIR, PYTHON_VERSION, TRAINING_DEFAULT_TIMEOUT_MINUTES
from tests.integ.kms_utils import get_or_create_kms_key
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name


@pytest.fixture(scope='module')
def mxnet_training_job(sagemaker_session, mxnet_full_version):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        script_path = os.path.join(DATA_DIR, 'mxnet_mnist', 'mnist.py')
        data_path = os.path.join(DATA_DIR, 'mxnet_mnist')

        mx = MXNet(entry_point=script_path, role='SageMakerRole', framework_version=mxnet_full_version,
                   py_version=PYTHON_VERSION, train_instance_count=1, train_instance_type='ml.c4.xlarge',
                   sagemaker_session=sagemaker_session)

        train_input = mx.sagemaker_session.upload_data(path=os.path.join(data_path, 'train'),
                                                       key_prefix='integ-test-data/mxnet_mnist/train')
        test_input = mx.sagemaker_session.upload_data(path=os.path.join(data_path, 'test'),
                                                      key_prefix='integ-test-data/mxnet_mnist/test')

        mx.fit({'train': train_input, 'test': test_input})
        return mx.latest_training_job.name


@pytest.mark.canary_quick
@pytest.mark.regional_testing
def test_attach_deploy(mxnet_training_job, sagemaker_session):
    endpoint_name = 'test-mxnet-attach-deploy-{}'.format(sagemaker_timestamp())

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        estimator = MXNet.attach(mxnet_training_job, sagemaker_session=sagemaker_session)
        predictor = estimator.deploy(1, 'ml.m4.xlarge', endpoint_name=endpoint_name)
        data = numpy.zeros(shape=(1, 1, 28, 28))
        result = predictor.predict(data)
        assert result is not None


def test_deploy_model(mxnet_training_job, sagemaker_session, mxnet_full_version):
    endpoint_name = 'test-mxnet-deploy-model-{}'.format(sagemaker_timestamp())

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        desc = sagemaker_session.sagemaker_client.describe_training_job(TrainingJobName=mxnet_training_job)
        model_data = desc['ModelArtifacts']['S3ModelArtifacts']
        script_path = os.path.join(DATA_DIR, 'mxnet_mnist', 'mnist.py')
        model = MXNetModel(model_data, 'SageMakerRole', entry_point=script_path,
                           py_version=PYTHON_VERSION, sagemaker_session=sagemaker_session,
                           framework_version=mxnet_full_version)
        predictor = model.deploy(1, 'ml.m4.xlarge', endpoint_name=endpoint_name)

        data = numpy.zeros(shape=(1, 1, 28, 28))
        result = predictor.predict(data)
        assert result is not None

    predictor.delete_model()
    with pytest.raises(Exception) as exception:
        sagemaker_session.sagemaker_client.describe_model(ModelName=model.name)
        assert 'Could not find model' in str(exception.value)


def test_deploy_model_with_tags_and_kms(mxnet_training_job, sagemaker_session, mxnet_full_version):
    endpoint_name = 'test-mxnet-deploy-model-{}'.format(sagemaker_timestamp())

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        desc = sagemaker_session.sagemaker_client.describe_training_job(TrainingJobName=mxnet_training_job)
        model_data = desc['ModelArtifacts']['S3ModelArtifacts']
        script_path = os.path.join(DATA_DIR, 'mxnet_mnist', 'mnist.py')
        model = MXNetModel(model_data, 'SageMakerRole', entry_point=script_path,
                           py_version=PYTHON_VERSION, sagemaker_session=sagemaker_session,
                           framework_version=mxnet_full_version)

        tags = [{'Key': 'TagtestKey', 'Value': 'TagtestValue'}]
        kms_key_arn = get_or_create_kms_key(sagemaker_session)

        model.deploy(1, 'ml.m4.xlarge', endpoint_name=endpoint_name, tags=tags, kms_key=kms_key_arn)

        returned_model = sagemaker_session.describe_model(EndpointName=model.name)
        returned_model_tags = sagemaker_session.list_tags(ResourceArn=returned_model['ModelArn'])['Tags']

        endpoint = sagemaker_session.describe_endpoint(EndpointName=endpoint_name)
        endpoint_tags = sagemaker_session.list_tags(ResourceArn=endpoint['EndpointArn'])['Tags']

        endpoint_config = sagemaker_session.describe_endpoint_config(EndpointConfigName=endpoint['EndpointConfigName'])
        endpoint_config_tags = sagemaker_session.list_tags(ResourceArn=endpoint_config['EndpointConfigArn'])['Tags']

        production_variants = endpoint_config['ProductionVariants']

        assert returned_model_tags == tags
        assert endpoint_config_tags == tags
        assert endpoint_tags == tags
        assert production_variants[0]['InstanceType'] == 'ml.m4.xlarge'
        assert production_variants[0]['InitialInstanceCount'] == 1
        assert endpoint_config['KmsKeyId'] == kms_key_arn


def test_deploy_model_with_update_endpoint(mxnet_training_job, sagemaker_session, mxnet_full_version):
    endpoint_name = 'test-mxnet-deploy-model-{}'.format(sagemaker_timestamp())

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        desc = sagemaker_session.sagemaker_client.describe_training_job(TrainingJobName=mxnet_training_job)
        model_data = desc['ModelArtifacts']['S3ModelArtifacts']
        script_path = os.path.join(DATA_DIR, 'mxnet_mnist', 'mnist.py')
        model = MXNetModel(model_data, 'SageMakerRole', entry_point=script_path,
                           py_version=PYTHON_VERSION, sagemaker_session=sagemaker_session,
                           framework_version=mxnet_full_version)
        model.deploy(1, 'ml.t2.medium', endpoint_name=endpoint_name)
        old_endpoint = sagemaker_session.describe_endpoint(EndpointName=endpoint_name)
        old_config_name = old_endpoint['EndpointConfigName']

        model.deploy(1, 'ml.m4.xlarge', update_endpoint=True, endpoint_name=endpoint_name)
        new_endpoint = sagemaker_session.describe_endpoint(EndpointName=endpoint_name)['ProductionVariants']
        new_production_variants = new_endpoint['ProductionVariants']
        new_config_name = new_endpoint['EndpointConfigName']

        assert old_config_name != new_config_name
        assert new_production_variants['InstanceType'] == 'ml.m4.xlarge'
        assert new_production_variants['InitialInstanceCount'] == 1
        assert new_production_variants['AcceleratorType'] is None


def test_deploy_model_with_update_non_existing_endpoint(mxnet_training_job, sagemaker_session, mxnet_full_version):
    endpoint_name = 'test-mxnet-deploy-model-{}'.format(sagemaker_timestamp())
    expected_error_message = 'Endpoint with name "{}" does not exist; ' \
                             'please use an existing endpoint name'.format(endpoint_name)

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        desc = sagemaker_session.sagemaker_client.describe_training_job(TrainingJobName=mxnet_training_job)
        model_data = desc['ModelArtifacts']['S3ModelArtifacts']
        script_path = os.path.join(DATA_DIR, 'mxnet_mnist', 'mnist.py')
        model = MXNetModel(model_data, 'SageMakerRole', entry_point=script_path,
                           py_version=PYTHON_VERSION, sagemaker_session=sagemaker_session,
                           framework_version=mxnet_full_version)
        model.deploy(1, 'ml.t2.medium', endpoint_name=endpoint_name)
        sagemaker_session.describe_endpoint(EndpointName=endpoint_name)

        with pytest.raises(ValueError, message=expected_error_message):
            model.deploy(1, 'ml.m4.xlarge', update_endpoint=True, endpoint_name='non-existing-endpoint')


@pytest.mark.canary_quick
@pytest.mark.regional_testing
@pytest.mark.skipif(tests.integ.test_region() not in tests.integ.EI_SUPPORTED_REGIONS,
                    reason="EI isn't supported in that specific region.")
def test_deploy_model_with_accelerator(mxnet_training_job, sagemaker_session, ei_mxnet_full_version):
    endpoint_name = 'test-mxnet-deploy-model-ei-{}'.format(sagemaker_timestamp())

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        desc = sagemaker_session.sagemaker_client.describe_training_job(TrainingJobName=mxnet_training_job)
        model_data = desc['ModelArtifacts']['S3ModelArtifacts']
        script_path = os.path.join(DATA_DIR, 'mxnet_mnist', 'mnist.py')
        model = MXNetModel(model_data, 'SageMakerRole', entry_point=script_path,
                           framework_version=ei_mxnet_full_version, py_version=PYTHON_VERSION,
                           sagemaker_session=sagemaker_session)
        predictor = model.deploy(1, 'ml.m4.xlarge', endpoint_name=endpoint_name, accelerator_type='ml.eia1.medium')

        data = numpy.zeros(shape=(1, 1, 28, 28))
        result = predictor.predict(data)
        assert result is not None


def test_async_fit(sagemaker_session, mxnet_full_version):
    endpoint_name = 'test-mxnet-attach-deploy-{}'.format(sagemaker_timestamp())

    with timeout(minutes=5):
        script_path = os.path.join(DATA_DIR, 'mxnet_mnist', 'mnist.py')
        data_path = os.path.join(DATA_DIR, 'mxnet_mnist')

        mx = MXNet(entry_point=script_path, role='SageMakerRole', py_version=PYTHON_VERSION,
                   train_instance_count=1, train_instance_type='ml.c4.xlarge',
                   sagemaker_session=sagemaker_session, framework_version=mxnet_full_version,
                   distributions={'parameter_server': {'enabled': True}})

        train_input = mx.sagemaker_session.upload_data(path=os.path.join(data_path, 'train'),
                                                       key_prefix='integ-test-data/mxnet_mnist/train')
        test_input = mx.sagemaker_session.upload_data(path=os.path.join(data_path, 'test'),
                                                      key_prefix='integ-test-data/mxnet_mnist/test')

        mx.fit({'train': train_input, 'test': test_input}, wait=False)
        training_job_name = mx.latest_training_job.name

        print("Waiting to re-attach to the training job: %s" % training_job_name)
        time.sleep(20)

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        print("Re-attaching now to: %s" % training_job_name)
        estimator = MXNet.attach(training_job_name=training_job_name, sagemaker_session=sagemaker_session)
        predictor = estimator.deploy(1, 'ml.m4.xlarge', endpoint_name=endpoint_name)
        data = numpy.zeros(shape=(1, 1, 28, 28))
        result = predictor.predict(data)
        assert result is not None


def test_failed_training_job(sagemaker_session, mxnet_full_version):
    with timeout():
        script_path = os.path.join(DATA_DIR, 'mxnet_mnist', 'failure_script.py')

        mx = MXNet(entry_point=script_path, role='SageMakerRole', framework_version=mxnet_full_version,
                   py_version=PYTHON_VERSION, train_instance_count=1, train_instance_type='ml.c4.xlarge',
                   sagemaker_session=sagemaker_session)

        with pytest.raises(ValueError) as e:
            mx.fit()
        assert 'ExecuteUserScriptError' in str(e.value)
