# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import copy
import json

import pytest
from mock import Mock

from sagemaker import RealTimePredictor
from sagemaker.amazon.pca import PCA
from sagemaker.amazon.amazon_estimator import RecordSet
from sagemaker.estimator import Estimator
from sagemaker.tuner import _ParameterRange, ContinuousParameter, IntegerParameter, CategoricalParameter, \
    HyperparameterTuner, _TuningJob
from sagemaker.mxnet import MXNet
MODEL_DATA = "s3://bucket/model.tar.gz"

JOB_NAME = 'tuning_job'
REGION = 'us-west-2'
BUCKET_NAME = 'Some-Bucket'
ROLE = 'myrole'
IMAGE_NAME = 'image'

TRAIN_INSTANCE_COUNT = 1
TRAIN_INSTANCE_TYPE = 'ml.c4.xlarge'
NUM_COMPONENTS = 5

SCRIPT_NAME = 'my_script.py'
FRAMEWORK_VERSION = '1.0.0'

INPUTS = 's3://mybucket/train'
OBJECTIVE_METRIC_NAME = 'mock_metric'
HYPERPARAMETER_RANGES = {'validated': ContinuousParameter(0, 5),
                         'elizabeth': IntegerParameter(0, 5),
                         'blank': CategoricalParameter([0, 5])}
METRIC_DEFINTIONS = 'mock_metric_definitions'

TUNING_JOB_DETAILS = {
    'HyperParameterTuningJobConfig': {
        'ResourceLimits': {
            'MaxParallelTrainingJobs': 1,
            'MaxNumberOfTrainingJobs': 1
        },
        'HyperParameterTuningJobObjective': {
            'MetricName': OBJECTIVE_METRIC_NAME,
            'Type': 'Minimize'
        },
        'Strategy': 'Bayesian',
        'ParameterRanges': {
            'CategoricalParameterRanges': [],
            'ContinuousParameterRanges': [],
            'IntegerParameterRanges': [
                {
                    'MaxValue': '100',
                    'Name': 'mini_batch_size',
                    'MinValue': '10',
                },
            ]
        }
    },
    'HyperParameterTuningJobName': JOB_NAME,
    'TrainingJobDefinition': {
        'RoleArn': ROLE,
        'StaticHyperParameters': {
            'num_components': '1',
            '_tuning_objective_metric': 'train:throughput',
            'feature_dim': '784',
            'sagemaker_estimator_module': '"sagemaker.amazon.pca"',
            'sagemaker_estimator_class_name': '"PCA"',
        },
        'ResourceConfig': {
            'VolumeSizeInGB': 30,
            'InstanceType': 'ml.c4.xlarge',
            'InstanceCount': 1
        },
        'AlgorithmSpecification': {
            'TrainingImage': IMAGE_NAME,
            'TrainingInputMode': 'File',
            'MetricDefinitions': METRIC_DEFINTIONS,
        },
        'InputDataConfig': [
            {
                'ChannelName': 'train',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataDistributionType': 'ShardedByS3Key',
                        'S3Uri': INPUTS,
                        'S3DataType': 'ManifestFile'
                    }
                }
            }
        ],
        'StoppingCondition': {
            'MaxRuntimeInSeconds': 86400
        },
        'OutputDataConfig': {
            'S3OutputPath': BUCKET_NAME,
        }
    },
    'TrainingJobCounters': {
        'ClientError': 0,
        'Completed': 1,
        'InProgress': 0,
        'Fault': 0,
        'Stopped': 0
    },
    'HyperParameterTuningEndTime': 1526605831.0,
    'CreationTime': 1526605605.0,
    'HyperParameterTuningJobArn': 'arn:tuning_job',
}


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name='boto_session', region_name=REGION)
    sms = Mock(name='sagemaker_session', boto_session=boto_mock)
    sms.boto_region_name = REGION
    sms.default_bucket = Mock(name='default_bucket', return_value=BUCKET_NAME)
    sms.config = None
    return sms


@pytest.fixture()
def estimator(sagemaker_session):
    return Estimator(IMAGE_NAME, ROLE, TRAIN_INSTANCE_COUNT, TRAIN_INSTANCE_TYPE, output_path='s3://bucket/prefix',
                     sagemaker_session=sagemaker_session)


@pytest.fixture()
def tuner(estimator):
    return HyperparameterTuner(estimator, OBJECTIVE_METRIC_NAME, HYPERPARAMETER_RANGES, METRIC_DEFINTIONS)


def test_prepare_for_training(tuner):
    static_hyperparameters = {'validated': 1, 'another_one': 0}
    tuner.estimator.set_hyperparameters(**static_hyperparameters)
    tuner._prepare_for_training()

    assert tuner._current_job_name.startswith(IMAGE_NAME)

    assert len(tuner.static_hyperparameters) == 3
    assert tuner.static_hyperparameters['another_one'] == '0'

    class_name = json.dumps(tuner.estimator.__class__.__name__)
    assert tuner.static_hyperparameters['sagemaker_estimator_class_name'] == class_name
    module = json.dumps(tuner.estimator.__module__)
    assert tuner.static_hyperparameters['sagemaker_estimator_module'] == module


def test_prepare_for_training_with_job_name(tuner):
    static_hyperparameters = {'validated': 1, 'another_one': 0}
    tuner.estimator.set_hyperparameters(**static_hyperparameters)

    tuner._prepare_for_training(job_name='some-other-job-name')
    assert tuner._current_job_name == 'some-other-job-name'


def test_validate_parameter_ranges_number_validation_error(sagemaker_session):
    pca = PCA(ROLE, TRAIN_INSTANCE_COUNT, TRAIN_INSTANCE_TYPE, NUM_COMPONENTS,
              base_job_name='pca', sagemaker_session=sagemaker_session)

    invalid_hyperparameter_ranges = {'num_components': IntegerParameter(-1, 2)}

    with pytest.raises(ValueError) as e:
        HyperparameterTuner(estimator=pca, objective_metric_name=OBJECTIVE_METRIC_NAME,
                            hyperparameter_ranges=invalid_hyperparameter_ranges, metric_definitions=METRIC_DEFINTIONS)

    assert 'Value must be an integer greater than zero' in str(e)


def test_validate_parameter_ranges_string_value_validation_error(sagemaker_session):
    pca = PCA(ROLE, TRAIN_INSTANCE_COUNT, TRAIN_INSTANCE_TYPE, NUM_COMPONENTS,
              base_job_name='pca', sagemaker_session=sagemaker_session)

    invalid_hyperparameter_ranges = {'algorithm_mode': CategoricalParameter([0, 5])}

    with pytest.raises(ValueError) as e:
        HyperparameterTuner(estimator=pca, objective_metric_name=OBJECTIVE_METRIC_NAME,
                            hyperparameter_ranges=invalid_hyperparameter_ranges, metric_definitions=METRIC_DEFINTIONS)

    assert 'Value must be one of "regular" and "randomized"' in str(e)


def test_fit_pca(sagemaker_session, tuner):
    pca = PCA(ROLE, TRAIN_INSTANCE_COUNT, TRAIN_INSTANCE_TYPE, NUM_COMPONENTS,
              base_job_name='pca', sagemaker_session=sagemaker_session)

    pca.algorithm_mode = 'randomized'
    pca.subtract_mean = True
    pca.extra_components = 5

    tuner.estimator = pca

    tags = [{'Name': 'some-tag-without-a-value'}]
    tuner.tags = tags

    hyperparameter_ranges = {'num_components': IntegerParameter(2, 4),
                             'algorithm_mode': CategoricalParameter(['regular', 'randomized'])}
    tuner._hyperparameter_ranges = hyperparameter_ranges

    records = RecordSet(s3_data=INPUTS, num_records=1, feature_dim=1)
    tuner.fit(records, mini_batch_size=9999)

    _, _, tune_kwargs = sagemaker_session.tune.mock_calls[0]

    assert len(tune_kwargs['static_hyperparameters']) == 4
    assert tune_kwargs['static_hyperparameters']['extra_components'] == '5'
    assert len(tune_kwargs['parameter_ranges']['IntegerParameterRanges']) == 1
    assert tune_kwargs['job_name'].startswith('pca')
    assert tune_kwargs['tags'] == tags
    assert tuner.estimator.mini_batch_size == 9999


def test_attach_tuning_job_with_estimator_from_hyperparameters(sagemaker_session):
    job_details = copy.deepcopy(TUNING_JOB_DETAILS)
    sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(name='describe_tuning_job',
                                                                                  return_value=job_details)
    tuner = HyperparameterTuner.attach(JOB_NAME, sagemaker_session=sagemaker_session)

    assert tuner.latest_tuning_job.name == JOB_NAME
    assert tuner.objective_metric_name == OBJECTIVE_METRIC_NAME
    assert tuner.max_jobs == 1
    assert tuner.max_parallel_jobs == 1
    assert tuner.metric_definitions == METRIC_DEFINTIONS
    assert tuner.strategy == 'Bayesian'
    assert tuner.objective_type == 'Minimize'

    assert isinstance(tuner.estimator, PCA)
    assert tuner.estimator.role == ROLE
    assert tuner.estimator.train_instance_count == 1
    assert tuner.estimator.train_max_run == 24 * 60 * 60
    assert tuner.estimator.input_mode == 'File'
    assert tuner.estimator.output_path == BUCKET_NAME
    assert tuner.estimator.output_kms_key == ''

    assert '_tuning_objective_metric' not in tuner.estimator.hyperparameters()
    assert tuner.estimator.hyperparameters()['num_components'] == '1'


def test_attach_tuning_job_with_job_details(sagemaker_session):
    job_details = copy.deepcopy(TUNING_JOB_DETAILS)
    HyperparameterTuner.attach(JOB_NAME, sagemaker_session=sagemaker_session, job_details=job_details)
    sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job.assert_not_called


def test_attach_tuning_job_with_estimator_from_image(sagemaker_session):
    job_details = copy.deepcopy(TUNING_JOB_DETAILS)
    job_details['TrainingJobDefinition']['AlgorithmSpecification']['TrainingImage'] = '1111.amazonaws.com/pca:1'
    sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(name='describe_tuning_job',
                                                                                  return_value=job_details)

    tuner = HyperparameterTuner.attach(JOB_NAME, sagemaker_session=sagemaker_session)
    assert isinstance(tuner.estimator, PCA)


def test_attach_tuning_job_with_estimator_from_kwarg(sagemaker_session):
    job_details = copy.deepcopy(TUNING_JOB_DETAILS)
    sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(name='describe_tuning_job',
                                                                                  return_value=job_details)
    tuner = HyperparameterTuner.attach(JOB_NAME, sagemaker_session=sagemaker_session,
                                       estimator_cls='sagemaker.estimator.Estimator')
    assert isinstance(tuner.estimator, Estimator)


def test_attach_with_no_specified_estimator(sagemaker_session):
    job_details = copy.deepcopy(TUNING_JOB_DETAILS)
    del job_details['TrainingJobDefinition']['StaticHyperParameters']['sagemaker_estimator_module']
    del job_details['TrainingJobDefinition']['StaticHyperParameters']['sagemaker_estimator_class_name']
    sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(name='describe_tuning_job',
                                                                                  return_value=job_details)

    tuner = HyperparameterTuner.attach(JOB_NAME, sagemaker_session=sagemaker_session)
    assert isinstance(tuner.estimator, Estimator)


def test_serialize_parameter_ranges(tuner):
    hyperparameter_ranges = tuner.hyperparameter_ranges()

    for key, value in HYPERPARAMETER_RANGES.items():
        assert hyperparameter_ranges[value.__name__ + 'ParameterRanges'][0]['Name'] == key


def test_analytics(tuner):
    tuner.latest_tuning_job = _TuningJob(tuner.sagemaker_session, 'testjob')
    tuner_analytics = tuner.analytics()
    assert tuner_analytics is not None
    assert tuner_analytics.name.find('testjob') > -1


def test_serialize_categorical_ranges_for_frameworks(sagemaker_session, tuner):
    tuner.estimator = MXNet(entry_point=SCRIPT_NAME,
                            role=ROLE,
                            framework_version=FRAMEWORK_VERSION,
                            train_instance_count=TRAIN_INSTANCE_COUNT,
                            train_instance_type=TRAIN_INSTANCE_TYPE,
                            sagemaker_session=sagemaker_session)

    hyperparameter_ranges = tuner.hyperparameter_ranges()

    assert hyperparameter_ranges['CategoricalParameterRanges'][0]['Name'] == 'blank'
    assert hyperparameter_ranges['CategoricalParameterRanges'][0]['Values'] == ['"0"', '"5"']


def test_serialize_nonexistent_parameter_ranges(tuner):
    temp_hyperparameter_ranges = HYPERPARAMETER_RANGES.copy()
    parameter_type = temp_hyperparameter_ranges['validated'].__name__

    temp_hyperparameter_ranges['validated'] = None
    tuner._hyperparameter_ranges = temp_hyperparameter_ranges

    ranges = tuner.hyperparameter_ranges()
    assert len(ranges.keys()) == 3
    assert not ranges[parameter_type + 'ParameterRanges']


def test_stop_tuning_job(sagemaker_session, tuner):
    sagemaker_session.stop_tuning_job = Mock(name='stop_hyper_parameter_tuning_job')
    tuner.latest_tuning_job = _TuningJob(sagemaker_session, JOB_NAME)

    tuner.stop_tuning_job()

    sagemaker_session.stop_tuning_job.assert_called_once_with(name=JOB_NAME)


def test_stop_tuning_job_no_tuning_job(tuner):
    with pytest.raises(ValueError) as e:
            tuner.stop_tuning_job()
    assert 'No tuning job available' in str(e)


def test_best_tuning_job(tuner):
    tuning_job_description = {'BestTrainingJob': {'TrainingJobName': JOB_NAME}}

    tuner.estimator.sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(
        name='describe_hyper_parameter_tuning_job', return_value=tuning_job_description)

    tuner.latest_tuning_job = _TuningJob(tuner.estimator.sagemaker_session, JOB_NAME)
    best_training_job = tuner.best_training_job()

    assert best_training_job == JOB_NAME
    tuner.estimator.sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job.assert_called_once_with(
        HyperParameterTuningJobName=JOB_NAME)


def test_best_tuning_job_no_latest_job(tuner):
    with pytest.raises(Exception) as e:
        tuner.best_training_job()

    assert 'No tuning job available' in str(e)


def test_best_tuning_job_no_best_job(tuner):
    tuning_job_description = {'BestTrainingJob': {'Mock': None}}

    tuner.estimator.sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(
        name='describe_hyper_parameter_tuning_job', return_value=tuning_job_description)

    tuner.latest_tuning_job = _TuningJob(tuner.estimator.sagemaker_session, JOB_NAME)

    with pytest.raises(Exception) as e:
        tuner.best_training_job()

    tuner.estimator.sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job.assert_called_once_with(
        HyperParameterTuningJobName=JOB_NAME)
    assert 'Best training job not available for tuning job:' in str(e)


def test_deploy_default(tuner):
    returned_training_job_description = {
        'AlgorithmSpecification': {
            'TrainingInputMode': 'File',
            'TrainingImage': IMAGE_NAME
        },
        'HyperParameters': {
            'sagemaker_submit_directory': '"s3://some/sourcedir.tar.gz"',
            'checkpoint_path': '"s3://other/1508872349"',
            'sagemaker_program': '"iris-dnn-classifier.py"',
            'sagemaker_enable_cloudwatch_metrics': 'false',
            'sagemaker_container_log_level': '"logging.INFO"',
            'sagemaker_job_name': '"neo"',
            'training_steps': '100',
            '_tuning_objective_metric': 'Validation-accuracy',
        },

        'RoleArn': ROLE,
        'ResourceConfig': {
            'VolumeSizeInGB': 30,
            'InstanceCount': 1,
            'InstanceType': 'ml.c4.xlarge'
        },
        'StoppingCondition': {
            'MaxRuntimeInSeconds': 24 * 60 * 60
        },
        'TrainingJobName': 'neo',
        'TrainingJobStatus': 'Completed',
        'OutputDataConfig': {
            'KmsKeyId': '',
            'S3OutputPath': 's3://place/output/neo'
        },
        'TrainingJobOutput': {
            'S3TrainingJobOutput': 's3://here/output.tar.gz'
        },
        'ModelArtifacts': {
            'S3ModelArtifacts': MODEL_DATA
        }
    }
    tuning_job_description = {'BestTrainingJob': {'TrainingJobName': JOB_NAME}}

    tuner.estimator.sagemaker_session.sagemaker_client.describe_training_job = \
        Mock(name='describe_training_job', return_value=returned_training_job_description)
    tuner.estimator.sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(
        name='describe_hyper_parameter_tuning_job', return_value=tuning_job_description)
    tuner.estimator.sagemaker_session.log_for_jobs = Mock(name='log_for_jobs')

    tuner.latest_tuning_job = _TuningJob(tuner.estimator.sagemaker_session, JOB_NAME)
    predictor = tuner.deploy(TRAIN_INSTANCE_COUNT, TRAIN_INSTANCE_TYPE)

    tuner.estimator.sagemaker_session.create_model.assert_called_once()
    args = tuner.estimator.sagemaker_session.create_model.call_args[0]
    assert args[0].startswith(IMAGE_NAME)
    assert args[1] == ROLE
    assert args[2]['Image'] == IMAGE_NAME
    assert args[2]['ModelDataUrl'] == MODEL_DATA

    assert isinstance(predictor, RealTimePredictor)
    assert predictor.endpoint.startswith(JOB_NAME)
    assert predictor.sagemaker_session == tuner.estimator.sagemaker_session


def test_wait(tuner):
    tuner.latest_tuning_job = _TuningJob(tuner.estimator.sagemaker_session, JOB_NAME)
    tuner.estimator.sagemaker_session.wait_for_tuning_job = Mock(name='wait_for_tuning_job')

    tuner.wait()

    tuner.estimator.sagemaker_session.wait_for_tuning_job.assert_called_once_with(JOB_NAME)


def test_delete_endpoint(tuner):
    tuner.latest_tuning_job = _TuningJob(tuner.estimator.sagemaker_session, JOB_NAME)

    tuning_job_description = {'BestTrainingJob': {'TrainingJobName': JOB_NAME}}
    tuner.estimator.sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(
        name='describe_hyper_parameter_tuning_job', return_value=tuning_job_description)

    tuner.delete_endpoint()
    tuner.sagemaker_session.delete_endpoint.assert_called_with(JOB_NAME)


#################################################################################
# _ParameterRange Tests

def test_continuous_parameter():
    cont_param = ContinuousParameter(0.1, 1e-2)
    assert isinstance(cont_param, _ParameterRange)
    assert cont_param.__name__ is 'Continuous'


def test_continuous_parameter_ranges():
    cont_param = ContinuousParameter(0.1, 1e-2)
    ranges = cont_param.as_tuning_range('some')
    assert len(ranges.keys()) == 3
    assert ranges['Name'] == 'some'
    assert ranges['MinValue'] == '0.1'
    assert ranges['MaxValue'] == '0.01'


def test_integer_parameter():
    int_param = IntegerParameter(1, 2)
    assert isinstance(int_param, _ParameterRange)
    assert int_param.__name__ is 'Integer'


def test_integer_parameter_ranges():
    int_param = IntegerParameter(1, 2)
    ranges = int_param.as_tuning_range('some')
    assert len(ranges.keys()) == 3
    assert ranges['Name'] == 'some'
    assert ranges['MinValue'] == '1'
    assert ranges['MaxValue'] == '2'


def test_categorical_parameter_list():
    cat_param = CategoricalParameter(['a', 'z'])
    assert isinstance(cat_param, _ParameterRange)
    assert cat_param.__name__ is 'Categorical'


def test_categorical_parameter_list_ranges():
    cat_param = CategoricalParameter([1, 10])
    ranges = cat_param.as_tuning_range('some')
    assert len(ranges.keys()) == 2
    assert ranges['Name'] == 'some'
    assert ranges['Values'] == ['1', '10']


def test_categorical_parameter_value():
    cat_param = CategoricalParameter('a')
    assert isinstance(cat_param, _ParameterRange)


def test_categorical_parameter_value_ranges():
    cat_param = CategoricalParameter('a')
    ranges = cat_param.as_tuning_range('some')
    assert len(ranges.keys()) == 2
    assert ranges['Name'] == 'some'
    assert ranges['Values'] == ['a']


#################################################################################
# _TuningJob Tests

def test_start_new(tuner, sagemaker_session):
    tuning_job = _TuningJob(sagemaker_session, JOB_NAME)

    tuner.static_hyperparameters = {}
    started_tuning_job = tuning_job.start_new(tuner, INPUTS)

    assert started_tuning_job.sagemaker_session == sagemaker_session
    sagemaker_session.tune.assert_called_once()


def test_stop(sagemaker_session):
    tuning_job = _TuningJob(sagemaker_session, JOB_NAME)
    tuning_job.stop()

    sagemaker_session.stop_tuning_job.assert_called_once_with(name=JOB_NAME)


def test_tuning_job_wait(sagemaker_session):
    sagemaker_session.wait_for_tuning_job = Mock(name='wait_for_tuning_job')

    tuning_job = _TuningJob(sagemaker_session, JOB_NAME)
    tuning_job.wait()

    sagemaker_session.wait_for_tuning_job.assert_called_once_with(JOB_NAME)
