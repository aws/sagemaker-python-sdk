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

import os

from mock.mock import Mock

from sagemaker.config import (
    SAGEMAKER,
    MONITORING_SCHEDULE,
    MONITORING_SCHEDULE_CONFIG,
    MONITORING_JOB_DEFINITION,
    MONITORING_OUTPUT_CONFIG,
    KMS_KEY_ID,
    MONITORING_RESOURCES,
    CLUSTER_CONFIG,
    VOLUME_KMS_KEY_ID,
    NETWORK_CONFIG,
    ENABLE_INTER_CONTAINER_TRAFFIC_ENCRYPTION,
    ENABLE_NETWORK_ISOLATION,
    VPC_CONFIG,
    SUBNETS,
    SECURITY_GROUP_IDS,
    ROLE_ARN,
    TAGS,
    KEY,
    VALUE,
    COMPILATION_JOB,
    OUTPUT_CONFIG,
    EDGE_PACKAGING_JOB,
    RESOURCE_KEY,
    ENDPOINT,
    ENDPOINT_CONFIG,
    DATA_CAPTURE_CONFIG,
    PRODUCTION_VARIANTS,
    AUTO_ML_JOB,
    AUTO_ML_JOB_V2,
    AUTO_ML_JOB_CONFIG,
    SECURITY_CONFIG,
    OUTPUT_DATA_CONFIG,
    MODEL_PACKAGE,
    VALIDATION_SPECIFICATION,
    VALIDATION_PROFILES,
    TRANSFORM_JOB_DEFINITION,
    TRANSFORM_OUTPUT,
    TRANSFORM_RESOURCES,
    VALIDATION_ROLE,
    FEATURE_GROUP,
    OFFLINE_STORE_CONFIG,
    S3_STORAGE_CONFIG,
    ONLINE_STORE_CONFIG,
    PROCESSING_JOB,
    PROCESSING_INPUTS,
    DATASET_DEFINITION,
    ATHENA_DATASET_DEFINITION,
    REDSHIFT_DATASET_DEFINITION,
    CLUSTER_ROLE_ARN,
    PROCESSING_OUTPUT_CONFIG,
    PROCESSING_RESOURCES,
    TRAINING_JOB,
    PROFILER_CONFIG,
    DISABLE_PROFILER,
    RESOURCE_CONFIG,
    TRANSFORM_JOB,
    EXECUTION_ROLE_ARN,
    MODEL,
    ASYNC_INFERENCE_CONFIG,
    SCHEMA_VERSION,
    PYTHON_SDK,
    MODULES,
    DEFAULT_S3_BUCKET,
    DEFAULT_S3_OBJECT_KEY_PREFIX,
    SESSION,
    ENVIRONMENT,
    CONTAINERS,
    PRIMARY_CONTAINER,
    INFERENCE_SPECIFICATION,
    ESTIMATOR,
    DEBUG_HOOK_CONFIG,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PY_VERSION = "py3"
DEFAULT_S3_BUCKET_NAME = "sagemaker-config-session-s3-bucket"
DEFAULT_S3_OBJECT_KEY_PREFIX_NAME = "test-prefix"

SAGEMAKER_CONFIG_SESSION = {
    SCHEMA_VERSION: "1.0",
    SAGEMAKER: {
        PYTHON_SDK: {
            MODULES: {
                SESSION: {
                    DEFAULT_S3_BUCKET: "sagemaker-config-session-s3-bucket",
                    DEFAULT_S3_OBJECT_KEY_PREFIX: "test-prefix",
                },
            },
        },
    },
}

SAGEMAKER_CONFIG_MONITORING_SCHEDULE = {
    SCHEMA_VERSION: "1.0",
    SAGEMAKER: {
        MONITORING_SCHEDULE: {
            MONITORING_SCHEDULE_CONFIG: {
                MONITORING_JOB_DEFINITION: {
                    ENVIRONMENT: {"configEnvVar1": "value1", "configEnvVar2": "value2"},
                    MONITORING_OUTPUT_CONFIG: {KMS_KEY_ID: "configKmsKeyId"},
                    MONITORING_RESOURCES: {
                        CLUSTER_CONFIG: {VOLUME_KMS_KEY_ID: "configVolumeKmsKeyId"},
                    },
                    NETWORK_CONFIG: {
                        ENABLE_INTER_CONTAINER_TRAFFIC_ENCRYPTION: False,
                        ENABLE_NETWORK_ISOLATION: True,
                        VPC_CONFIG: {
                            SUBNETS: ["subnets-123"],
                            SECURITY_GROUP_IDS: ["sg-123"],
                        },
                    },
                    ROLE_ARN: "arn:aws:iam::111111111111:role/ConfigRole",
                }
            },
            TAGS: [{KEY: "some-tag", VALUE: "value-for-tag"}],
        }
    },
}

SAGEMAKER_CONFIG_COMPILATION_JOB = {
    SCHEMA_VERSION: "1.0",
    SAGEMAKER: {
        COMPILATION_JOB: {
            OUTPUT_CONFIG: {KMS_KEY_ID: "TestKms"},
            ROLE_ARN: "arn:aws:iam::111111111111:role/ConfigRole",
            VPC_CONFIG: {SUBNETS: ["subnets-123"], SECURITY_GROUP_IDS: ["sg-123"]},
            TAGS: [{KEY: "some-tag", VALUE: "value-for-tag"}],
        },
    },
}

SAGEMAKER_CONFIG_EDGE_PACKAGING_JOB = {
    SCHEMA_VERSION: "1.0",
    SAGEMAKER: {
        EDGE_PACKAGING_JOB: {
            OUTPUT_CONFIG: {
                KMS_KEY_ID: "configKmsKeyId",
            },
            RESOURCE_KEY: "kmskeyid1",
            ROLE_ARN: "arn:aws:iam::111111111111:role/ConfigRole",
            TAGS: [{KEY: "some-tag", VALUE: "value-for-tag"}],
        },
    },
}

SAGEMAKER_CONFIG_ENDPOINT_CONFIG = {
    SCHEMA_VERSION: "1.0",
    SAGEMAKER: {
        ENDPOINT_CONFIG: {
            ASYNC_INFERENCE_CONFIG: {
                OUTPUT_CONFIG: {
                    KMS_KEY_ID: "testOutputKmsKeyId",
                }
            },
            DATA_CAPTURE_CONFIG: {
                KMS_KEY_ID: "testDataCaptureKmsKeyId",
            },
            KMS_KEY_ID: "ConfigKmsKeyId",
            PRODUCTION_VARIANTS: [
                {"CoreDumpConfig": {"KmsKeyId": "testCoreKmsKeyId"}},
                {"CoreDumpConfig": {"KmsKeyId": "testCoreKmsKeyId2"}},
            ],
            TAGS: [{KEY: "some-tag", VALUE: "value-for-tag"}],
        },
    },
}

SAGEMAKER_CONFIG_ENDPOINT = {
    SCHEMA_VERSION: "1.0",
    SAGEMAKER: {
        ENDPOINT: {
            TAGS: [{KEY: "some-tag", VALUE: "value-for-tag"}],
        }
    },
}

SAGEMAKER_CONFIG_ENDPOINT_ENDPOINT_CONFIG_COMBINED = {
    SCHEMA_VERSION: "1.0",
    SAGEMAKER: {
        ENDPOINT_CONFIG: {
            ASYNC_INFERENCE_CONFIG: {
                OUTPUT_CONFIG: {
                    KMS_KEY_ID: "testOutputKmsKeyId",
                }
            },
            DATA_CAPTURE_CONFIG: {
                KMS_KEY_ID: "testDataCaptureKmsKeyId",
            },
            KMS_KEY_ID: "ConfigKmsKeyId",
            PRODUCTION_VARIANTS: [
                {"CoreDumpConfig": {"KmsKeyId": "testCoreKmsKeyId"}},
                {"CoreDumpConfig": {"KmsKeyId": "testCoreKmsKeyId2"}},
            ],
            TAGS: [{KEY: "some-tag", VALUE: "value-for-tag"}],
        },
        ENDPOINT: {
            TAGS: [{KEY: "some-tag1", VALUE: "value-for-tag1"}],
        },
    },
}

SAGEMAKER_CONFIG_AUTO_ML = {
    SCHEMA_VERSION: "1.0",
    SAGEMAKER: {
        AUTO_ML_JOB: {
            AUTO_ML_JOB_CONFIG: {
                SECURITY_CONFIG: {
                    ENABLE_INTER_CONTAINER_TRAFFIC_ENCRYPTION: True,
                    VOLUME_KMS_KEY_ID: "TestKmsKeyId",
                    VPC_CONFIG: {
                        SUBNETS: ["subnets-123"],
                        SECURITY_GROUP_IDS: ["sg-123"],
                    },
                },
            },
            OUTPUT_DATA_CONFIG: {KMS_KEY_ID: "configKmsKeyId"},
            ROLE_ARN: "arn:aws:iam::111111111111:role/ConfigRole",
            TAGS: [{KEY: "some-tag", VALUE: "value-for-tag"}],
        },
    },
}

SAGEMAKER_CONFIG_AUTO_ML_V2 = {
    SCHEMA_VERSION: "1.0",
    SAGEMAKER: {
        AUTO_ML_JOB_V2: {
            SECURITY_CONFIG: {
                ENABLE_INTER_CONTAINER_TRAFFIC_ENCRYPTION: True,
                VOLUME_KMS_KEY_ID: "TestKmsKeyId",
                VPC_CONFIG: {
                    SUBNETS: ["subnets-123"],
                    SECURITY_GROUP_IDS: ["sg-123"],
                },
            },
            OUTPUT_DATA_CONFIG: {KMS_KEY_ID: "configKmsKeyId"},
            ROLE_ARN: "arn:aws:iam::111111111111:role/ConfigRole",
            TAGS: [{KEY: "some-tag", VALUE: "value-for-tag"}],
        },
    },
}

SAGEMAKER_CONFIG_MODEL_PACKAGE = {
    SCHEMA_VERSION: "1.0",
    SAGEMAKER: {
        MODEL_PACKAGE: {
            INFERENCE_SPECIFICATION: {
                CONTAINERS: [
                    {
                        ENVIRONMENT: {"configEnvVar1": "value1", "configEnvVar2": "value2"},
                    }
                ],
            },
            VALIDATION_SPECIFICATION: {
                VALIDATION_PROFILES: [
                    {
                        TRANSFORM_JOB_DEFINITION: {
                            ENVIRONMENT: {"configEnvVar1": "value1", "configEnvVar2": "value2"},
                            TRANSFORM_OUTPUT: {KMS_KEY_ID: "testKmsKeyId"},
                            TRANSFORM_RESOURCES: {VOLUME_KMS_KEY_ID: "testVolumeKmsKeyId"},
                        }
                    }
                ],
                VALIDATION_ROLE: "arn:aws:iam::111111111111:role/ConfigRole",
            },
            # TODO - does SDK not support tags for this API?
            # TAGS: EXAMPLE_TAGS,
        },
    },
}

SAGEMAKER_CONFIG_FEATURE_GROUP = {
    SCHEMA_VERSION: "1.0",
    SAGEMAKER: {
        FEATURE_GROUP: {
            OFFLINE_STORE_CONFIG: {
                S3_STORAGE_CONFIG: {
                    KMS_KEY_ID: "OfflineConfigKmsKeyId",
                }
            },
            ONLINE_STORE_CONFIG: {
                SECURITY_CONFIG: {
                    KMS_KEY_ID: "OnlineConfigKmsKeyId",
                }
            },
            ROLE_ARN: "arn:aws:iam::111111111111:role/ConfigRole",
            TAGS: [{KEY: "some-tag", VALUE: "value-for-tag"}],
        },
    },
}

SAGEMAKER_CONFIG_PROCESSING_JOB = {
    SCHEMA_VERSION: "1.0",
    SAGEMAKER: {
        PROCESSING_JOB: {
            ENVIRONMENT: {"configEnvVar1": "value1", "configEnvVar2": "value2"},
            NETWORK_CONFIG: {
                ENABLE_INTER_CONTAINER_TRAFFIC_ENCRYPTION: False,
                ENABLE_NETWORK_ISOLATION: True,
                VPC_CONFIG: {SUBNETS: ["subnets-123"], SECURITY_GROUP_IDS: ["sg-123"]},
            },
            PROCESSING_INPUTS: [
                {
                    DATASET_DEFINITION: {
                        ATHENA_DATASET_DEFINITION: {
                            KMS_KEY_ID: "AthenaKmsKeyId",
                        },
                        REDSHIFT_DATASET_DEFINITION: {
                            KMS_KEY_ID: "RedshiftKmsKeyId",
                            CLUSTER_ROLE_ARN: "arn:aws:iam::111111111111:role/ClusterRole",
                        },
                    },
                },
            ],
            PROCESSING_OUTPUT_CONFIG: {KMS_KEY_ID: "testKmsKeyId"},
            PROCESSING_RESOURCES: {
                CLUSTER_CONFIG: {
                    VOLUME_KMS_KEY_ID: "testVolumeKmsKeyId",
                },
            },
            ROLE_ARN: "arn:aws:iam::111111111111:role/ConfigRole",
            TAGS: [{KEY: "some-tag", VALUE: "value-for-tag"}],
        },
    },
}

SAGEMAKER_CONFIG_TRAINING_JOB = {
    SCHEMA_VERSION: "1.0",
    SAGEMAKER: {
        PYTHON_SDK: {
            MODULES: {
                ESTIMATOR: {
                    DEBUG_HOOK_CONFIG: False,
                },
            },
        },
        TRAINING_JOB: {
            ENABLE_INTER_CONTAINER_TRAFFIC_ENCRYPTION: True,
            ENABLE_NETWORK_ISOLATION: True,
            ENVIRONMENT: {"configEnvVar1": "value1", "configEnvVar2": "value2"},
            OUTPUT_DATA_CONFIG: {KMS_KEY_ID: "TestKms"},
            RESOURCE_CONFIG: {VOLUME_KMS_KEY_ID: "volumekey"},
            PROFILER_CONFIG: {DISABLE_PROFILER: False},
            ROLE_ARN: "arn:aws:iam::111111111111:role/ConfigRole",
            VPC_CONFIG: {SUBNETS: ["subnets-123"], SECURITY_GROUP_IDS: ["sg-123"]},
            TAGS: [{KEY: "some-tag", VALUE: "value-for-tag"}],
        },
    },
}

SAGEMAKER_CONFIG_TRAINING_JOB_WITH_DEBUG_HOOK_CONFIG_AS_FALSE = {
    SCHEMA_VERSION: "1.0",
    SAGEMAKER: {
        PYTHON_SDK: {
            MODULES: {
                ESTIMATOR: {
                    DEBUG_HOOK_CONFIG: False,
                },
            },
        },
    },
}

SAGEMAKER_CONFIG_TRAINING_JOB_WITH_DEBUG_HOOK_CONFIG_AS_TRUE = {
    SCHEMA_VERSION: "1.0",
    SAGEMAKER: {
        PYTHON_SDK: {
            MODULES: {
                ESTIMATOR: {
                    DEBUG_HOOK_CONFIG: True,
                },
            },
        },
    },
}

SAGEMAKER_CONFIG_TRANSFORM_JOB = {
    SCHEMA_VERSION: "1.0",
    SAGEMAKER: {
        TRANSFORM_JOB: {
            DATA_CAPTURE_CONFIG: {KMS_KEY_ID: "jobKmsKeyId"},
            ENVIRONMENT: {"configEnvVar1": "value1", "configEnvVar2": "value2"},
            TRANSFORM_OUTPUT: {KMS_KEY_ID: "outputKmsKeyId"},
            TRANSFORM_RESOURCES: {VOLUME_KMS_KEY_ID: "volumeKmsKeyId"},
            TAGS: [{KEY: "some-tag", VALUE: "value-for-tag"}],
        }
    },
}

SAGEMAKER_CONFIG_MODEL = {
    SCHEMA_VERSION: "1.0",
    SAGEMAKER: {
        MODEL: {
            CONTAINERS: [
                {
                    ENVIRONMENT: {"configEnvVar1": "value1", "configEnvVar2": "value2"},
                },
                {
                    ENVIRONMENT: {"configEnvVar1": "value1", "configEnvVar2": "value2"},
                },
            ],
            ENABLE_NETWORK_ISOLATION: True,
            EXECUTION_ROLE_ARN: "arn:aws:iam::111111111111:role/ConfigRole",
            VPC_CONFIG: {SUBNETS: ["subnets-123"], SECURITY_GROUP_IDS: ["sg-123"]},
            TAGS: [{KEY: "some-tag", VALUE: "value-for-tag"}],
        },
    },
}

SAGEMAKER_CONFIG_MODEL_WITH_PRIMARY_CONTAINER = {
    SCHEMA_VERSION: "1.0",
    SAGEMAKER: {
        MODEL: {
            ENABLE_NETWORK_ISOLATION: True,
            EXECUTION_ROLE_ARN: "arn:aws:iam::111111111111:role/ConfigRole",
            PRIMARY_CONTAINER: {
                ENVIRONMENT: {"configEnvVar1": "value1", "configEnvVar2": "value2"},
            },
            VPC_CONFIG: {SUBNETS: ["subnets-123"], SECURITY_GROUP_IDS: ["sg-123"]},
            TAGS: [{KEY: "some-tag", VALUE: "value-for-tag"}],
        },
    },
}


def _test_default_bucket_and_prefix_combinations(
    function_with_user_input=None,
    function_without_user_input=None,
    expected__without_user_input__with_default_bucket_and_default_prefix=None,
    expected__without_user_input__with_default_bucket_only=None,
    expected__with_user_input__with_default_bucket_and_prefix=None,
    expected__with_user_input__with_default_bucket_only=None,
    session_with_bucket_and_prefix=Mock(
        name="sagemaker_session",
        sagemaker_config={},
        default_bucket=Mock(name="default_bucket", return_value=DEFAULT_S3_BUCKET_NAME),
        default_bucket_prefix=DEFAULT_S3_OBJECT_KEY_PREFIX_NAME,
        config=None,
        settings=None,
    ),
    session_with_bucket_and_no_prefix=Mock(
        name="sagemaker_session",
        sagemaker_config={},
        default_bucket_prefix=None,
        default_bucket=Mock(name="default_bucket", return_value=DEFAULT_S3_BUCKET_NAME),
        config=None,
        settings=None,
    ),
):
    """
    Helper to test the different possible scenarios of how S3 params will be generated.

    Possible scenarios:
        1. User provided their own input, so (in most cases) there is no need to use default params
        2. User did not provide input. Session has a default_bucket_prefix set
        2. User did not provide input. Session does NOT have a default_bucket_prefix set
    """

    actual_values = []
    expected_values = []

    # With Default Bucket and Default Prefix
    if expected__without_user_input__with_default_bucket_and_default_prefix:
        actual_values.append(function_without_user_input(session_with_bucket_and_prefix))
        expected_values.append(expected__without_user_input__with_default_bucket_and_default_prefix)

    # With Default Bucket and no Default Prefix
    if expected__without_user_input__with_default_bucket_only:
        actual_values.append(function_without_user_input(session_with_bucket_and_no_prefix))
        expected_values.append(expected__without_user_input__with_default_bucket_only)

    # With user input & With Default Bucket and Default Prefix
    if expected__with_user_input__with_default_bucket_and_prefix:
        actual_values.append(function_with_user_input(session_with_bucket_and_prefix))
        expected_values.append(expected__with_user_input__with_default_bucket_and_prefix)

    # With user input & With Default Bucket and no Default Prefix
    if expected__with_user_input__with_default_bucket_only:
        actual_values.append(function_with_user_input(session_with_bucket_and_no_prefix))
        expected_values.append(expected__with_user_input__with_default_bucket_only)

    # It is better to put assert statements in the caller function rather than within here.
    # (If we put Asserts inside of this function, the info logged is not very debuggable. It just
    # says that the Assert failed, and doesn't show the difference.)
    return actual_values, expected_values
