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
    ENDPOINT_CONFIG,
    DATA_CAPTURE_CONFIG,
    PRODUCTION_VARIANTS,
    AUTO_ML_JOB,
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
    RESOURCE_CONFIG,
    TRANSFORM_JOB,
    EXECUTION_ROLE_ARN,
    MODEL,
    ASYNC_INFERENCE_CONFIG,
    SCHEMA_VERSION,
    PYTHON_SDK,
    MODULES,
    S3_BUCKET,
    SESSION,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PY_VERSION = "py3"

SAGEMAKER_CONFIG_SESSION = {
    SCHEMA_VERSION: "1.0",
    SAGEMAKER: {
        PYTHON_SDK: {
            MODULES: {
                SESSION: {
                    S3_BUCKET: "sagemaker-config-session-s3-bucket",
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
                    MONITORING_OUTPUT_CONFIG: {KMS_KEY_ID: "configKmsKeyId"},
                    MONITORING_RESOURCES: {
                        CLUSTER_CONFIG: {VOLUME_KMS_KEY_ID: "configVolumeKmsKeyId"},
                    },
                    NETWORK_CONFIG: {
                        ENABLE_INTER_CONTAINER_TRAFFIC_ENCRYPTION: False,
                        ENABLE_NETWORK_ISOLATION: True,
                        VPC_CONFIG: {SUBNETS: ["subnets-123"], SECURITY_GROUP_IDS: ["sg-123"]},
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

SAGEMAKER_CONFIG_AUTO_ML = {
    SCHEMA_VERSION: "1.0",
    SAGEMAKER: {
        AUTO_ML_JOB: {
            AUTO_ML_JOB_CONFIG: {
                SECURITY_CONFIG: {
                    ENABLE_INTER_CONTAINER_TRAFFIC_ENCRYPTION: True,
                    VOLUME_KMS_KEY_ID: "TestKmsKeyId",
                    VPC_CONFIG: {SUBNETS: ["subnets-123"], SECURITY_GROUP_IDS: ["sg-123"]},
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
            VALIDATION_SPECIFICATION: {
                VALIDATION_PROFILES: [
                    {
                        TRANSFORM_JOB_DEFINITION: {
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
        TRAINING_JOB: {
            ENABLE_INTER_CONTAINER_TRAFFIC_ENCRYPTION: True,
            ENABLE_NETWORK_ISOLATION: True,
            OUTPUT_DATA_CONFIG: {KMS_KEY_ID: "TestKms"},
            RESOURCE_CONFIG: {VOLUME_KMS_KEY_ID: "volumekey"},
            ROLE_ARN: "arn:aws:iam::111111111111:role/ConfigRole",
            VPC_CONFIG: {SUBNETS: ["subnets-123"], SECURITY_GROUP_IDS: ["sg-123"]},
            TAGS: [{KEY: "some-tag", VALUE: "value-for-tag"}],
        },
    },
}

SAGEMAKER_CONFIG_TRANSFORM_JOB = {
    SCHEMA_VERSION: "1.0",
    SAGEMAKER: {
        TRANSFORM_JOB: {
            DATA_CAPTURE_CONFIG: {KMS_KEY_ID: "jobKmsKeyId"},
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
            ENABLE_NETWORK_ISOLATION: True,
            EXECUTION_ROLE_ARN: "arn:aws:iam::111111111111:role/ConfigRole",
            VPC_CONFIG: {SUBNETS: ["subnets-123"], SECURITY_GROUP_IDS: ["sg-123"]},
            TAGS: [{KEY: "some-tag", VALUE: "value-for-tag"}],
        },
    },
}
