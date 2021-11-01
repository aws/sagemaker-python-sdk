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
"""Amazon SageMaker channel configurations for S3 data sources and file system data sources"""
from __future__ import absolute_import, print_function

import attr

FILE_SYSTEM_TYPES = ["FSxLustre", "EFS"]
FILE_SYSTEM_ACCESS_MODES = ["ro", "rw"]


class TrainingInput(object):
    """Amazon SageMaker channel configurations for S3 data sources.

    Attributes:
        config (dict[str, dict]): A SageMaker ``DataSource`` referencing
            a SageMaker ``S3DataSource``.
    """

    def __init__(
        self,
        s3_data,
        distribution=None,
        compression=None,
        content_type=None,
        record_wrapping=None,
        s3_data_type="S3Prefix",
        input_mode=None,
        attribute_names=None,
        target_attribute_name=None,
        shuffle_config=None,
    ):
        """Create a definition for input data used by an SageMaker training job.

        See AWS documentation on the ``CreateTrainingJob`` API for more details on the parameters.

        Args:
            s3_data (str): Defines the location of s3 data to train on.
            distribution (str): Valid values: 'FullyReplicated', 'ShardedByS3Key'
                (default: 'FullyReplicated').
            compression (str): Valid values: 'Gzip', None (default: None). This is used only in
                Pipe input mode.
            content_type (str): MIME type of the input data (default: None).
            record_wrapping (str): Valid values: 'RecordIO' (default: None).
            s3_data_type (str): Valid values: 'S3Prefix', 'ManifestFile', 'AugmentedManifestFile'.
                If 'S3Prefix', ``s3_data`` defines a prefix of s3 objects to train on.
                All objects with s3 keys beginning with ``s3_data`` will be used to train.
                If 'ManifestFile' or 'AugmentedManifestFile', then ``s3_data`` defines a
                single S3 manifest file or augmented manifest file (respectively),
                listing the S3 data to train on. Both the ManifestFile and
                AugmentedManifestFile formats are described in the SageMaker API documentation:
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_S3DataSource.html
            input_mode (str): Optional override for this channel's input mode (default: None).
                By default, channels will use the input mode defined on
                ``sagemaker.estimator.EstimatorBase.input_mode``, but they will ignore
                that setting if this parameter is set.

                    * None - Amazon SageMaker will use the input mode specified in the ``Estimator``
                    * 'File' - Amazon SageMaker copies the training dataset from the S3 location to
                        a local directory.
                    * 'Pipe' - Amazon SageMaker streams data directly from S3 to the container via
                        a Unix-named pipe.
                    * 'FastFile' - Amazon SageMaker streams data from S3 on demand instead of
                        downloading the entire dataset before training begins.

            attribute_names (list[str]): A list of one or more attribute names to use that are
                found in a specified AugmentedManifestFile.
            target_attribute_name (str): The name of the attribute will be predicted (classified)
                in a SageMaker AutoML job. It is required if the input is for SageMaker AutoML job.
            shuffle_config (sagemaker.inputs.ShuffleConfig): If specified this configuration enables
                shuffling on this channel. See the SageMaker API documentation for more info:
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_ShuffleConfig.html
        """
        self.config = {
            "DataSource": {"S3DataSource": {"S3DataType": s3_data_type, "S3Uri": s3_data}}
        }

        if not (target_attribute_name or distribution):
            distribution = "FullyReplicated"

        if distribution is not None:
            self.config["DataSource"]["S3DataSource"]["S3DataDistributionType"] = distribution

        if compression is not None:
            self.config["CompressionType"] = compression
        if content_type is not None:
            self.config["ContentType"] = content_type
        if record_wrapping is not None:
            self.config["RecordWrapperType"] = record_wrapping
        if input_mode is not None:
            self.config["InputMode"] = input_mode
        if attribute_names is not None:
            self.config["DataSource"]["S3DataSource"]["AttributeNames"] = attribute_names
        if target_attribute_name is not None:
            self.config["TargetAttributeName"] = target_attribute_name
        if shuffle_config is not None:
            self.config["ShuffleConfig"] = {"Seed": shuffle_config.seed}


class ShuffleConfig(object):
    """For configuring channel shuffling using a seed.

    For more detail, see the AWS documentation:
    https://docs.aws.amazon.com/sagemaker/latest/dg/API_ShuffleConfig.html
    """

    def __init__(self, seed):
        """Create a ShuffleConfig.

        Args:
            seed (long): the long value used to seed the shuffled sequence.
        """
        self.seed = seed


@attr.s
class CreateModelInput(object):
    """A class containing parameters which can be used to create a SageMaker Model

    Parameters:
        instance_type (str): type or EC2 instance will be used for model deployment.
        accelerator_type (str): elastic inference accelerator type.
    """

    instance_type: str = attr.ib(default=None)
    accelerator_type: str = attr.ib(default=None)


@attr.s
class CompilationInput(object):
    """Create a class containing all the parameters.

    It can be used when calling ``sagemaker.model.Model.compile_model()``

    Parameters:
        target_instance_type(str): Identifies the device that you want to
                run your model after compilation, for example: ml_c5. For allowed
                strings see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_OutputConfig.html.
        input_shape(str): Specifies the name and shape of the expected
                inputs for your trained model in json dictionary form, for
                example: {'data': [1,3,1024,1024]}, or {'var1': [1,1,28,28],
                'var2': [1,1,28,28]}
        output_path(str): Specifies where to store the compiled model
        framework (str, optional): The framework that is used to train the original
                model. Allowed values: 'mxnet', 'tensorflow', 'keras', 'pytorch',
                'onnx', 'xgboost' (default: None)
        framework_version (str, optional): The version of the framework (default: None)
        compile_max_run (int, optional): Timeout in seconds for compilation (default:
            15 * 60). After this amount of time Amazon SageMaker Neo
            terminates the compilation job regardless of its current status.
        tags (list[dict], optional): List of tags for labeling a compilation job.
            For more, see
            https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
        job_name (str, optional): The name of the compilation job (default: None)
        target_platform_os (str, optional): Target Platform OS, for example: 'LINUX'.
            (default: None)
            For allowed strings see
            https://docs.aws.amazon.com/sagemaker/latest/dg/API_OutputConfig.html.
            It can be used instead of target_instance_family.
        target_platform_arch (str, optional): Target Platform Architecture, for example: 'X86_64'.
            (default: None)
            For allowed strings see
            https://docs.aws.amazon.com/sagemaker/latest/dg/API_OutputConfig.html.
            It can be used instead of target_instance_family.
        target_platform_accelerator (str, optional): Target Platform Accelerator,
            for example: 'NVIDIA'. (default: None)
            For allowed strings see
            https://docs.aws.amazon.com/sagemaker/latest/dg/API_OutputConfig.html.
            It can be used instead of target_instance_family.
        compiler_options (dict, optional): Additional parameters for compiler. (default: None)
            Compiler Options are TargetPlatform / target_instance_family specific. See
            https://docs.aws.amazon.com/sagemaker/latest/dg/API_OutputConfig.html for details.
    """

    target_instance_type: str = attr.ib(default=None)
    input_shape: dict = attr.ib(factory=dict)
    output_path: str = attr.ib(default=None)
    framework: str = attr.ib(default=None)
    framework_version: str = attr.ib(default=None)
    compile_max_run: int = attr.ib(default=15 * 60)
    tags: list = attr.ib(factory=list)
    job_name: str = attr.ib(default=None)
    target_platform_os: str = attr.ib(default=None)
    target_platform_arch: str = attr.ib(default=None)
    target_platform_accelerator: str = attr.ib(default=None)
    compiler_options: dict = attr.ib(default=None)


@attr.s
class TransformInput(object):
    """Create a class containing all the parameters.

    It can be used when calling ``sagemaker.transformer.Transformer.transform()``
    """

    data: str = attr.ib()
    data_type: str = attr.ib(default="S3Prefix")
    content_type: str = attr.ib(default=None)
    compression_type: str = attr.ib(default=None)
    split_type: str = attr.ib(default=None)
    input_filter: str = attr.ib(default=None)
    output_filter: str = attr.ib(default=None)
    join_source: str = attr.ib(default=None)
    model_client_config: dict = attr.ib(default=None)


class FileSystemInput(object):
    """Amazon SageMaker channel configurations for file system data sources.

    Attributes:
        config (dict[str, dict]): A Sagemaker File System ``DataSource``.
    """

    def __init__(
        self,
        file_system_id,
        file_system_type,
        directory_path,
        file_system_access_mode="ro",
        content_type=None,
    ):
        """Create a new file system input used by an SageMaker training job.

        Args:
            file_system_id (str): An Amazon file system ID starting with 'fs-'.
            file_system_type (str): The type of file system used for the input.
                Valid values: 'EFS', 'FSxLustre'.
            directory_path (str): Absolute or normalized path to the root directory (mount point) in
                the file system.
                Reference: https://docs.aws.amazon.com/efs/latest/ug/mounting-fs.html and
                https://docs.aws.amazon.com/fsx/latest/LustreGuide/mount-fs-auto-mount-onreboot.html
            file_system_access_mode (str): Permissions for read and write.
                Valid values: 'ro' or 'rw'. Defaults to 'ro'.
        """

        if file_system_type not in FILE_SYSTEM_TYPES:
            raise ValueError(
                "Unrecognized file system type: %s. Valid values: %s."
                % (file_system_type, ", ".join(FILE_SYSTEM_TYPES))
            )

        if file_system_access_mode not in FILE_SYSTEM_ACCESS_MODES:
            raise ValueError(
                "Unrecognized file system access mode: %s. Valid values: %s."
                % (file_system_access_mode, ", ".join(FILE_SYSTEM_ACCESS_MODES))
            )

        self.config = {
            "DataSource": {
                "FileSystemDataSource": {
                    "FileSystemId": file_system_id,
                    "FileSystemType": file_system_type,
                    "DirectoryPath": directory_path,
                    "FileSystemAccessMode": file_system_access_mode,
                }
            }
        }

        if content_type:
            self.config["ContentType"] = content_type
