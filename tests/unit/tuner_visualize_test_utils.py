# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

TRIALS_DF_COLUMNS = [
    'criterion', 'max-depth', 'min-samples-leaf', 'min-weight-fraction-leaf', 'n-estimators', 'TrainingJobName',
    'TrainingJobStatus',
    'TrainingStartTime', 'TrainingEndTime', 'TrainingElapsedTimeSeconds', 'TuningJobName', 'valid-f1'
]

FULL_DF_COLUMNS = [
    'value', 'ts', 'label', 'rel_ts', 'TrainingJobName', 'criterion', 'max-depth', 'min-samples-leaf',
    'min-weight-fraction-leaf', 'n-estimators', 'TrainingJobStatus', 'TrainingStartTime', 'TrainingEndTime',
    'TrainingElapsedTimeSeconds', 'TuningJobName', 'valid-f1'
]


TRIALS_DF_TRAINING_JOB_NAMES = [
    'random-240712-1545-019-4ac17a84', 'random-240712-1545-021-fcd64dc1'
]

TRIALS_DF_TRAINING_JOB_STATUSES = ['Completed', 'Completed']

TUNING_JOB_NAME_1 = 'random-240712-1500'
TUNING_JOB_NAME_2 = 'bayesian-240712-1600'
TUNING_JOB_NAMES = [TUNING_JOB_NAME_1, TUNING_JOB_NAME_2]
TRIALS_DF_VALID_F1_VALUES = [0.950, 0.896]

FULL_DF_COLUMNS = ['value', 'ts', 'label', 'rel_ts', 'TrainingJobName', 'criterion', 'max-depth', 'min-samples-leaf',
                   'min-weight-fraction-leaf', 'n-estimators', 'TrainingJobStatus', 'TrainingStartTime',
                   'TrainingEndTime', 'TrainingElapsedTimeSeconds', 'TuningJobName', 'valid-f1']

TUNED_PARAMETERS = ['n-estimators', 'max-depth', 'min-samples-leaf', 'min-weight-fraction-leaf', 'criterion']
OBJECTIVE_NAME = 'valid-f1'

TRIALS_DF_DATA = {
    'criterion': ['gini', 'log_loss'],
    'max-depth': [18.0, 8.0],
    'min-samples-leaf': [3.0, 10.0],
    'min-weight-fraction-leaf': [0.011596, 0.062067],
    'n-estimators': [110.0, 18.0],
    'TrainingJobName': ['random-240712-1545-019-4ac17a84', 'random-240712-1545-021-fcd64dc1'],
    'TrainingJobStatus': ['Completed', 'Completed'],
    'TrainingStartTime': ['2024-07-12 17:55:59+02:00', '2024-07-12 17:56:50+02:00'],
    'TrainingEndTime': ['2024-07-12 17:56:43+02:00', '2024-07-12 17:57:29+02:00'],
    'TrainingElapsedTimeSeconds': [44.0, 39.0],
    'TuningJobName': TUNING_JOB_NAMES,
    'valid-f1': [0.950, 0.896]
}

FULL_DF_DATA = {
    'value': [0.951000, 0.950000],
    'ts': ['2024-07-12 15:56:00', '2024-07-12 15:56:00'],
    'label': ['valid-precision', 'valid-recall'],
    'rel_ts': ['1970-01-01 01:00:00', '1970-01-01 01:00:00'],
    'TrainingJobName': ['random-240712-1545-019-4ac17a84', 'random-240712-1545-019-4ac17a84'],
    'criterion': ['gini', 'gini'],
    'max-depth': [18.0, 18.0],
    'min-samples-leaf': [3.0, 3.0],
    'min-weight-fraction-leaf': [0.011596, 0.011596],
    'n-estimators': [110.0, 110.0],
    'TrainingJobStatus': ['Completed', 'Completed'],
    'TrainingStartTime': ['2024-07-12 17:55:59+02:00', '2024-07-12 17:55:59+02:00'],
    'TrainingEndTime': ['2024-07-12 17:56:43+02:00', '2024-07-12 17:56:43+02:00'],
    'TrainingElapsedTimeSeconds': [44.0, 45.0],
    'TuningJobName': ['random-240712-1545', 'random-240712-1545'],
    'valid-f1': [0.9500, 0.9500]
}

FILTERED_TUNING_JOB_DF_DATA = {
    'criterion': ['log_loss', 'gini'],
    'max-depth': [10.0, 16.0],
    'min-samples-leaf': [7.0, 2.0],
    'min-weight-fraction-leaf': [0.160910, 0.069803],
    'n-estimators': [67.0, 79.0],
    'TrainingJobName': ['random-240712-1545-050-c0b5c10a', 'random-240712-1545-049-2db2ec05'],
    'TrainingJobStatus': ['Completed', 'Completed'],
    'FinalObjectiveValue': [0.8190, 0.8910],
    'TrainingStartTime': ['2024-07-12 18:09:48+02:00', '2024-07-12 18:09:45+02:00'],
    'TrainingEndTime': ['2024-07-12 18:10:28+02:00', '2024-07-12 18:10:23+02:00'],
    'TrainingElapsedTimeSeconds': [40.0, 38.0],
    'TuningJobName': [TUNING_JOB_NAME_1, TUNING_JOB_NAME_2]
}

TUNING_RANGES = [
    {
        'Name': 'n-estimators',
        'MinValue': '1',
        'MaxValue': '200',
        'ScalingType': 'Auto'
    },
    {
        'Name': 'max-depth',
        'MinValue': '1',
        'MaxValue': '20',
        'ScalingType': 'Auto'
    },
    {
        'Name': 'min-samples-leaf',
        'MinValue': '1',
        'MaxValue': '10',
        'ScalingType': 'Auto'
    },
    {
        'Name': 'min-weight-fraction-leaf',
        'MinValue': '0.01',
        'MaxValue': '0.5',
        'ScalingType': 'Auto'
    },
    {
        'Name': 'criterion',
        'Values': ['"gini"', '"entropy"', '"log_loss"']
    }
]

TUNING_JOB_RESULT = {
    'HyperParameterTuningJobName': TUNING_JOB_NAME_1,
    'HyperParameterTuningJobConfig': {
        'Strategy': 'Random',
        'HyperParameterTuningJobObjective': {
            'Type': 'Maximize',
            'MetricName': 'valid-f1'
        }
    },
    'HyperParameterTuningJobStatus': 'Completed',
}
