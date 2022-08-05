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

import json
import os
from random import getrandbits
from typing import List

from sagemaker import ModelMetrics, MetricsSource, FileSource, Predictor
from sagemaker.drift_check_baselines import DriftCheckBaselines
from sagemaker.instance_group import InstanceGroup
from sagemaker.metadata_properties import MetadataProperties
from sagemaker.model import FrameworkModel
from sagemaker.parameter import IntegerParameter
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.sparkml import SparkMLModel
from sagemaker.tensorflow import TensorFlow
from sagemaker.tuner import WarmStartConfig, WarmStartTypes
from mock import Mock, PropertyMock
from sagemaker import TrainingInput
from sagemaker.debugger import (
    Rule,
    DebuggerHookConfig,
    TensorBoardOutputConfig,
    ProfilerConfig,
    CollectionConfig,
)
from sagemaker.clarify import DataConfig
from sagemaker.network import NetworkConfig
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.amazon.amazon_estimator import RecordSet
from sagemaker.rl.estimator import RLToolkit, RLFramework
from sagemaker.pytorch import PyTorch
from sagemaker.workflow.entities import PipelineVariable
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import (
    ParameterString,
    ParameterInteger,
    ParameterFloat,
    ParameterBoolean,
)
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, TuningStep, TransformStep
from tests.unit import DATA_DIR

STR_VAL = "MyString"
ROLE = "DummyRole"
INSTANCE_TYPE = "ml.m5.xlarge"
BUCKET = "my-bucket"
REGION = "us-west-2"
IMAGE_URI = "fakeimage"
DUMMY_S3_SCRIPT_PATH = "s3://dummy-s3/dummy_script.py"
TENSORFLOW_PATH = os.path.join(DATA_DIR, "tfs/tfs-test-entrypoint-and-dependencies")
TENSORFLOW_ENTRY_POINT = os.path.join(TENSORFLOW_PATH, "inference.py")
GIT_REPO = "https://github.com/aws/sagemaker-python-sdk.git"
BRANCH = "test-branch-git-config"
COMMIT = "ae15c9d7d5b97ea95ea451e4662ee43da3401d73"

DEFAULT_VALUE = "default_value"
CLAZZ_ARGS = "clazz_args"
FUNC_ARGS = "func_args"
REQUIRED = "required"
OPTIONAL = "optional"
COMMON = "common"
INIT = "init"
TYPE = "type"


class PipelineVariableEncoder(json.JSONEncoder):
    """The Json Encoder for PipelineVariable"""

    def default(self, obj):
        """To return a serializable object for the input object if it's a PipelineVariable

        or call the base implementation if it's not

        Args:
            obj (object): The input object to be handled.
        """
        if isinstance(obj, PipelineVariable):
            return obj.expr
        return json.JSONEncoder.default(self, obj)


class MockProperties(PipelineVariable):
    """A mock object or Pipeline Properties"""

    def __init__(
        self,
        step_name: str,
        path: str = None,
        shape_name: str = None,
        shape_names: List[str] = None,
        service_name: str = "sagemaker",
    ):
        """Initialize a MockProperties object"""
        self.step_name = step_name
        self.path = path

    @property
    def expr(self):
        """The 'Get' expression dict for a `Properties`."""
        return {"Get": f"Steps.{self.step_name}.Outcome"}

    @property
    def _referenced_steps(self) -> List[str]:
        """List of step names that this function depends on."""
        return [self.step_name]


def _generate_mock_pipeline_session():
    """Generate mock pipeline session"""
    client_mock = Mock()
    client_mock._client_config.user_agent = (
        "Boto3/1.14.24 Python/3.8.5 Linux/5.4.0-42-generic Botocore/1.17.24 Resource"
    )
    client_mock.describe_algorithm.return_value = {
        "TrainingSpecification": {
            "TrainingChannels": [
                {
                    "SupportedContentTypes": ["application/json"],
                    "SupportedInputModes": ["File"],
                    "Name": "train",
                }
            ],
            "SupportedTrainingInstanceTypes": ["ml.m5.xlarge", "ml.m4.xlarge"],
            "SupportedHyperParameters": [
                {
                    "Name": "MyKey",
                    "Type": "FreeText",
                }
            ],
        },
        "AlgorithmName": "algo-name",
    }

    role_mock = Mock()
    type(role_mock).arn = PropertyMock(return_value=ROLE)
    resource_mock = Mock()
    resource_mock.Role.return_value = role_mock
    session_mock = Mock(region_name=REGION)
    session_mock.resource.return_value = resource_mock
    session_mock.client.return_value = client_mock

    return PipelineSession(
        boto_session=session_mock,
        sagemaker_client=client_mock,
        default_bucket=BUCKET,
    )


def _generate_all_pipeline_vars() -> dict:
    """Generate a dic with all kinds of Pipeline variables"""
    # Parameter
    ppl_param_str = ParameterString(name="MyString")
    ppl_param_int = ParameterInteger(name="MyInt")
    ppl_param_float = ParameterFloat(name="MyFloat")
    ppl_param_bool = ParameterBoolean(name="MyBool")

    # Function
    ppl_join = Join(on=" ", values=[ppl_param_int, ppl_param_float, 1, "test"])
    property_file = PropertyFile(
        name="name",
        output_name="result",
        path="output",
    )
    ppl_json_get = JsonGet(
        step_name="my-step",
        property_file=property_file,
        json_path="my-json-path",
    )

    # Properties
    ppl_prop = MockProperties(step_name="MyPreStep")

    # Execution Variables
    ppl_exe_var = ExecutionVariables.PIPELINE_NAME

    return dict(
        str=[
            (
                ppl_param_str,
                dict(origin=ppl_param_str.expr, to_string=ppl_param_str.to_string().expr),
            ),
            (ppl_join, dict(origin=ppl_join.expr, to_string=ppl_join.to_string().expr)),
            (ppl_json_get, dict(origin=ppl_json_get.expr, to_string=ppl_json_get.to_string().expr)),
            (ppl_prop, dict(origin=ppl_prop.expr, to_string=ppl_prop.to_string().expr)),
            (ppl_exe_var, dict(origin=ppl_exe_var.expr, to_string=ppl_exe_var.to_string().expr)),
        ],
        int=[
            (
                ppl_param_int,
                dict(origin=ppl_param_int.expr, to_string=ppl_param_int.to_string().expr),
            ),
            (ppl_json_get, dict(origin=ppl_json_get.expr, to_string=ppl_json_get.to_string().expr)),
            (ppl_prop, dict(origin=ppl_prop.expr, to_string=ppl_prop.to_string().expr)),
        ],
        float=[
            (
                ppl_param_float,
                dict(origin=ppl_param_float.expr, to_string=ppl_param_float.to_string().expr),
            ),
            (ppl_json_get, dict(origin=ppl_json_get.expr, to_string=ppl_json_get.to_string().expr)),
            (
                ppl_prop,
                dict(origin=ppl_prop.expr, to_string=ppl_prop.to_string().expr),
            ),
        ],
        bool=[
            (
                ppl_param_bool,
                dict(origin=ppl_param_bool.expr, to_string=ppl_param_bool.to_string().expr),
            ),
            (ppl_json_get, dict(origin=ppl_json_get.expr, to_string=ppl_json_get.to_string().expr)),
            (
                ppl_prop,
                dict(origin=ppl_prop.expr, to_string=ppl_prop.to_string().expr),
            ),
        ],
    )


# TODO: we should remove the _IS_TRUE_TMP and replace its usages with IS_TRUE
# As currently the `instance_groups` does not work well with some estimator subclasses,
# we temporarily hard code it to False which disables the instance_groups
_IS_TRUE_TMP = False
IS_TRUE = bool(getrandbits(1))
PIPELINE_SESSION = _generate_mock_pipeline_session()
PIPELINE_VARIABLES = _generate_all_pipeline_vars()

# TODO: need to recursively assign with Pipeline Variable in later changes
FIXED_ARGUMENTS = dict(
    common=dict(
        role=ROLE,
        sagemaker_session=PIPELINE_SESSION,
        source_dir=f"s3://{BUCKET}/source",
        entry_point=TENSORFLOW_ENTRY_POINT,
        dependencies=[os.path.join(TENSORFLOW_PATH, "dependency.py")],
        code_location=f"s3://{BUCKET}/code",
        predictor_cls=Predictor,
        model_metrics=ModelMetrics(
            model_statistics=MetricsSource(
                content_type=ParameterString(name="model_statistics_content_type"),
                s3_uri=ParameterString(name="model_statistics_s3_uri"),
                content_digest=ParameterString(name="model_statistics_content_digest"),
            )
        ),
        metadata_properties=MetadataProperties(
            commit_id=ParameterString(name="meta_properties_commit_id"),
            repository=ParameterString(name="meta_properties_repository"),
            generated_by=ParameterString(name="meta_properties_generated_by"),
            project_id=ParameterString(name="meta_properties_project_id"),
        ),
        drift_check_baselines=DriftCheckBaselines(
            model_constraints=MetricsSource(
                content_type=ParameterString(name="drift_constraints_content_type"),
                s3_uri=ParameterString(name="drift_constraints_s3_uri"),
                content_digest=ParameterString(name="drift_constraints_content_digest"),
            ),
            bias_config_file=FileSource(
                content_type=ParameterString(name="drift_bias_content_type"),
                s3_uri=ParameterString(name="drift_bias_s3_uri"),
                content_digest=ParameterString(name="drift_bias_content_digest"),
            ),
        ),
        model_package_name="my-model-pkg" if IS_TRUE else None,
        model_package_group_name="my-model-pkg-group" if not IS_TRUE else None,
        inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
        transform_instances=["ml.t2.medium", "ml.m5.xlarge"],
        content_types=["application/json"],
        response_types=["application/json"],
    ),
    processor=dict(
        instance_type=INSTANCE_TYPE,
        estimator_cls=PyTorch,
        code=f"s3://{BUCKET}/code",
        spark_event_logs_s3_uri=f"s3://{BUCKET}/my-spark-output-path",
        framework_version="1.8",
        network_config=NetworkConfig(
            subnets=[ParameterString(name="nw_cfg_subnets")],
            security_group_ids=[ParameterString(name="nw_cfg_security_group_ids")],
            enable_network_isolation=ParameterBoolean(name="nw_cfg_enable_nw_isolation"),
            encrypt_inter_container_traffic=ParameterBoolean(
                name="nw_cfg_encrypt_inter_container_traffic"
            ),
        ),
        inputs=[
            ProcessingInput(
                source=ParameterString(name="proc_input_source"),
                destination=ParameterString(name="proc_input_dest"),
                s3_data_type=ParameterString(name="proc_input_s3_data_type"),
                app_managed=ParameterBoolean(name="proc_input_app_managed"),
            ),
        ],
        outputs=[
            ProcessingOutput(
                source=ParameterString(name="proc_output_source"),
                destination=ParameterString(name="proc_output_dest"),
                app_managed=ParameterBoolean(name="proc_output_app_managed"),
            ),
        ],
        data_config=DataConfig(
            s3_data_input_path=ParameterString(name="clarify_processor_input"),
            s3_output_path=ParameterString(name="clarify_processor_output"),
            s3_analysis_config_output_path="s3://analysis_config_output_path",
        ),
        data_bias_config=DataConfig(
            s3_data_input_path=ParameterString(name="clarify_processor_input"),
            s3_output_path=ParameterString(name="clarify_processor_output"),
            s3_analysis_config_output_path="s3://analysis_config_output_path",
        ),
    ),
    estimator=dict(
        image_uri_region="us-west-2",
        input_mode="File",
        records=RecordSet(
            s3_data=ParameterString(name="records_s3_data"),
            num_records=1000,
            feature_dim=128,
            s3_data_type=ParameterString(name="records_s3_data_type"),
            channel=ParameterString(name="records_channel"),
        ),
        disable_profiler=False,
        vector_dim=128,
        enc_dim=128,
        momentum=1e-6,
        beta_1=1e-4,
        beta_2=1e-4,
        mini_batch_size=1000,
        dropout=0.25,
        num_classes=10,
        mlp_dim=512,
        mlp_activation="relu",
        output_layer="softmax",
        comparator_list="hadamard,concat,abs_diff",
        token_embedding_storage_type="dense",
        enc0_network="bilstm",
        enc1_network="bilstm",
        enc0_token_embedding_dim=256,
        enc1_token_embedding_dim=256,
        enc0_vocab_size=512,
        enc1_vocab_size=512,
        bias_init_method="normal",
        factors_init_method="normal",
        predictor_type="regressor",
        linear_init_method="uniform",
        toolkit=RLToolkit.RAY,
        toolkit_version="1.6.0",
        framework=RLFramework.PYTORCH,
        algorithm_mode="regular",
        num_topics=6,
        k=6,
        init_method="kmeans++",
        local_init_method="kmeans++",
        eval_metrics="mds,ssd",
        tol=1e-4,
        dimension_reduction_type="sign",
        index_type="faiss.Flat",
        faiss_index_ivf_nlists="auto",
        index_metric="COSINE",
        binary_classifier_model_selection_criteria="f1",
        positive_example_weight_mult="auto",
        loss="logistic",
        target_recall=0.1,
        target_precision=0.8,
        early_stopping_tolerance=1e-4,
        encoder_layers_activation="relu",
        optimizer="adam",
        tolerance=1e-4,
        rescale_gradient=1e-2,
        weight_decay=1e-6,
        learning_rate=1e-4,
        num_trees=50,
        source_dir=f"s3://{BUCKET}/source",
        entry_point=os.path.join(TENSORFLOW_PATH, "inference.py"),
        dependencies=[os.path.join(TENSORFLOW_PATH, "dependency.py")],
        code_location=f"s3://{BUCKET}/code",
        output_path=f"s3://{BUCKET}/output",
        model_uri=f"s3://{BUCKET}/model",
        py_version="py2",
        framework_version="2.1.1",
        rules=[
            Rule.custom(
                name="CustomeRule",
                image_uri=ParameterString(name="rules_image_uri"),
                instance_type=ParameterString(name="rules_instance_type"),
                volume_size_in_gb=ParameterInteger(name="rules_volume_size"),
                source="path/to/my_custom_rule.py",
                rule_to_invoke=ParameterString(name="rules_to_invoke"),
                container_local_output_path=ParameterString(name="rules_local_output"),
                s3_output_path=ParameterString(name="rules_to_s3_output_path"),
                other_trials_s3_input_paths=[ParameterString(name="rules_other_s3_input")],
                rule_parameters={"threshold": ParameterString(name="rules_param")},
                collections_to_save=[
                    CollectionConfig(
                        name=ParameterString(name="rules_collections_name"),
                        parameters={"key1": ParameterString(name="rules_collections_param")},
                    )
                ],
            )
        ],
        debugger_hook_config=DebuggerHookConfig(
            s3_output_path=ParameterString(name="debugger_hook_s3_output"),
            container_local_output_path=ParameterString(name="debugger_container_output"),
            hook_parameters={"key1": ParameterString(name="debugger_hook_param")},
            collection_configs=[
                CollectionConfig(
                    name=ParameterString(name="debugger_collections_name"),
                    parameters={"key1": ParameterString(name="debugger_collections_param")},
                )
            ],
        ),
        tensorboard_output_config=TensorBoardOutputConfig(
            s3_output_path=ParameterString(name="tensorboard_s3_output"),
            container_local_output_path=ParameterString(name="tensorboard_container_output"),
        ),
        profiler_config=ProfilerConfig(
            s3_output_path=ParameterString(name="profile_config_s3_output_path"),
            system_monitor_interval_millis=ParameterInteger(name="profile_config_system_monitor"),
        ),
        inputs={
            "train": TrainingInput(
                s3_data=ParameterString(name="train_inputs_s3_data"),
                distribution=ParameterString(name="train_inputs_distribution"),
                compression=ParameterString(name="train_inputs_compression"),
                content_type=ParameterString(name="train_inputs_content_type"),
                record_wrapping=ParameterString(name="train_inputs_record_wrapping"),
                s3_data_type=ParameterString(name="train_inputs_s3_data_type"),
                input_mode=ParameterString(name="train_inputs_input_mode"),
                attribute_names=[ParameterString(name="train_inputs_attribute_name")],
                target_attribute_name=ParameterString(name="train_inputs_target_attr_name"),
                instance_groups=[ParameterString(name="train_inputs_instance_groups")],
            ),
        },
        instance_groups=[
            InstanceGroup(
                instance_group_name=ParameterString(name="instance_group_name"),
                # hard code the instance_type here because InstanceGroup.instance_type
                # would be used to retrieve image_uri if image_uri is not presented
                # and currently the test mechanism does not support skip the test case
                # relating to bonded parameters in composite variables (i.e. the InstanceGroup)
                # TODO: we should support skip testing on bonded parameters in composite vars
                instance_type="ml.m5.xlarge",
                instance_count=ParameterString(name="instance_group_instance_count"),
            ),
        ]
        if _IS_TRUE_TMP
        else None,
        instance_type="ml.m5.xlarge" if not _IS_TRUE_TMP else None,
        instance_count=1 if not _IS_TRUE_TMP else None,
        distribution={} if not _IS_TRUE_TMP else None,
    ),
    transformer=dict(
        instance_type=INSTANCE_TYPE,
        data=f"s3://{BUCKET}/data",
    ),
    tuner=dict(
        instance_type=INSTANCE_TYPE,
        estimator=TensorFlow(
            entry_point=TENSORFLOW_ENTRY_POINT,
            role=ROLE,
            framework_version="2.1.1",
            py_version="py2",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=PIPELINE_SESSION,
            enable_sagemaker_metrics=True,
            max_retry_attempts=3,
            hyperparameters={"static-hp": "hp1", "train_size": "1280"},
        ),
        hyperparameter_ranges={
            "batch-size": IntegerParameter(
                min_value=ParameterInteger(name="hyper_range_min_value"),
                max_value=ParameterInteger(name="hyper_range_max_value"),
                scaling_type=ParameterString(name="hyper_range_scaling_type"),
            ),
        },
        warm_start_config=WarmStartConfig(
            warm_start_type=WarmStartTypes.IDENTICAL_DATA_AND_ALGORITHM,
            parents={ParameterString(name="warm_start_cfg_parent")},
        ),
        estimator_name="estimator-1",
        inputs={
            "estimator-1": TrainingInput(s3_data=ParameterString(name="inputs_estimator_1")),
        },
        include_cls_metadata={"estimator-1": IS_TRUE},
    ),
    model=dict(
        instance_type=INSTANCE_TYPE,
        serverless_inference_config=ServerlessInferenceConfig(),
        framework_version="1.11.0",
        py_version="py3",
        accelerator_type="ml.eia2.xlarge",
    ),
    pipelinemodel=dict(
        instance_type=INSTANCE_TYPE,
        models=[
            SparkMLModel(
                name="MySparkMLModel",
                model_data=f"s3://{BUCKET}",
                role=ROLE,
                sagemaker_session=PIPELINE_SESSION,
                env={"SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT": "text/csv"},
            ),
            FrameworkModel(
                image_uri=IMAGE_URI,
                model_data=f"s3://{BUCKET}/model.tar.gz",
                role=ROLE,
                sagemaker_session=PIPELINE_SESSION,
                entry_point=f"{DATA_DIR}/dummy_script.py",
                name="modelName",
                vpc_config={"Subnets": ["abc", "def"], "SecurityGroupIds": ["123", "456"]},
            ),
        ],
    ),
)
STEP_CLASS = dict(
    processor=ProcessingStep,
    estimator=TrainingStep,
    transformer=TransformStep,
    tuner=TuningStep,
    model=ModelStep,
    pipelinemodel=ModelStep,
)

# A dict for class __init__ parameters which are not used in
# request dict generated by a specific target function but are used in another
# For example, as for Model class constructor, the vpc_config is used only in
# model.create and is ignored in model.register
# Thus when testing on model.register we should skip replacing vpc_config
# with pipeline variables
CLASS_PARAMS_EXCLUDED_IN_TARGET_FUNC = dict(
    model=dict(
        register=dict(
            common={"vpc_config", "enable_network_isolation"},
        ),
    ),
    pipelinemodel=dict(
        register=dict(
            common={"vpc_config", "enable_network_isolation"},
        ),
    ),
)
# A dict for base class __init__ parameters which are not used in
# some specific subclasses.
# For example, TensorFlowModel uses **kwarg for duplicate parameters
# in base class (FrameworkModel/Model) but it ignores the "image_config"
# in target functions.
BASE_CLASS_PARAMS_EXCLUDED_IN_SUB_CLASS = dict(
    model=dict(
        TensorFlowModel={"image_config"},
        SKLearnModel={"image_config"},
        PyTorchModel={"image_config"},
        XGBoostModel={"image_config"},
        ChainerModel={"image_config"},
        HuggingFaceModel={"image_config"},
        MXNetModel={"image_config"},
        KNNModel={"image_uri"},  # it's overridden in __init__ with a fixed value
        SparkMLModel={"image_uri"},  # it's overridden in __init__ with a fixed value
        KMeansModel={"image_uri"},  # it's overridden in __init__ with a fixed value
        PCAModel={"image_uri"},  # it's overridden in __init__ with a fixed value
        LDAModel={"image_uri"},  # it's overridden in __init__ with a fixed value
        NTMModel={"image_uri"},  # it's overridden in __init__ with a fixed value
        Object2VecModel={"image_uri"},  # it's overridden in __init__ with a fixed value
        FactorizationMachinesModel={"image_uri"},  # it's overridden in __init__ with a fixed value
        IPInsightsModel={"image_uri"},  # it's overridden in __init__ with a fixed value
        RandomCutForestModel={"image_uri"},  # it's overridden in __init__ with a fixed value
        LinearLearnerModel={"image_uri"},  # it's overridden in __init__ with a fixed value
        MultiDataModel={
            "model_data",  # model_data is overridden in __init__ with model_data_prefix
            "image_config",
            "container_log_level",  # it's simply ignored
        },
    ),
    estimator=dict(
        AlgorithmEstimator={  # its kwargs is ignored so base class parameters are ignored
            "enable_network_isolation",
            "container_log_level",
            "checkpoint_s3_uri",
            "checkpoint_local_path",
            "enable_sagemaker_metrics",
            "environment",
            "max_retry_attempts",
            "source_dir",
            "entry_point",
        },
        SKLearn={
            "instance_count",
            "instance_type",
        },
    ),
)
# A dict to keep the optional arguments which should not be set to None
# in the test iteration according to the logic specific to the subclass.
PARAMS_SHOULD_NOT_BE_NONE = dict(
    estimator=dict(
        init=dict(
            # TODO: we should remove the three instance_ parameters here
            # For mutually exclusive parameters: instance group
            # vs instance count/instance type, if any side is set to None during iteration,
            # the other side should get a not None value, instead of listing them here
            # and force them to be not None
            common={"instance_count", "instance_type", "instance_groups"},
            LDA={"mini_batch_size"},
        )
    ),
    model=dict(
        register=dict(
            common={},
            Model={"model_data"},
            HuggingFaceModel={"model_data"},
        ),
        create=dict(
            common={},
            Model={"role"},
            SparkMLModel={"role"},
            MultiDataModel={"role"},
        ),
    ),
)
# A dict for parameters which should not be replaced with pipeline variables
# since they are bonded with other parameters with None value. For example:
# Case 1: if outputs (a parameter in FrameworkProcessor.run) is None,
# output_kms_key (a parameter in constructor) is omitted
# so don't need to replace it with pipeline variables
# Case 2: if image_uri is None, instance_type is not allowed to be pipeline variables,
# otherwise, the the class can fail to be initiated
UNSET_PARAM_BONDED_WITH_NONE = dict(
    processor=dict(
        init=dict(
            common=dict(instance_type={"image_uri"}),
        ),
        run=dict(
            common=dict(output_kms_key={"outputs"}),
        ),
    ),
    estimator=dict(
        init=dict(
            common=dict(
                # entry_point can only be parameterized when source_dir is given
                # if source_dir is None, entry_point should be skipped to parameterize
                entry_point={"source_dir"},
                instance_type={"image_uri"},
            ),
        ),
        fit=dict(
            common=dict(
                subnets={"security_group_ids"},
                security_group_ids={"subnets"},
                model_channel_name={"model_uri"},
                checkpoint_local_path={"checkpoint_s3_uri"},
                instance_type={"image_uri"},
            ),
        ),
    ),
    model=dict(
        register=dict(
            common=dict(
                env={"model_package_group_name"},
                image_config={"model_package_group_name"},
                model_server_workers={"model_package_group_name"},
                container_log_level={"model_package_group_name"},
                framework={"model_package_group_name"},
                framework_version={"model_package_group_name"},
                nearest_model_name={"model_package_group_name"},
                data_input_configuration={"model_package_group_name"},
            )
        ),
    ),
    pipelinemodel=dict(
        register=dict(
            common=dict(
                image_uri={
                    # model_package_name and model_package_group_name are mutual exclusive.
                    # If model_package_group_name is not None, image_uri will be ignored
                    "model_package_name"
                },
                framework={"model_package_group_name"},
                framework_version={"model_package_group_name"},
                nearest_model_name={"model_package_group_name"},
                data_input_configuration={"model_package_group_name"},
            ),
        ),
    ),
)

# A dict for parameters which should not be replaced with pipeline variables
# since they are bonded with other parameters with not None value. For example:
# 1. for any model subclass, if model_package_name is not None, model_package_group_name should be None
# and should skip to be replaced with a pipeline variable
# 2. for MultiDataModel, if if model is given, its kwargs including container_log_level will be ignored
# Note: for any mutual exclusive parameters (e.g. model_package_name, model_package_group_name),
# we can add an entry for each of them.
UNSET_PARAM_BONDED_WITH_NOT_NONE = dict(
    model=dict(
        register=dict(
            common=dict(
                model_package_name={"model_package_group_name"},
                model_package_group_name={"model_package_name"},
            ),
        ),
    ),
    pipelinemodel=dict(
        register=dict(
            common=dict(
                model_package_name={"model_package_group_name"},
                model_package_group_name={"model_package_name"},
            ),
        ),
    ),
    estimator=dict(
        init=dict(
            common=dict(
                entry_point={"enable_network_isolation"},
                source_dir={"enable_network_isolation"},
            ),
            TensorFlow=dict(
                image_uri={"compiler_config"},
                compiler_config={"image_uri"},
            ),
            HuggingFace=dict(
                image_uri={"compiler_config"},
                compiler_config={"image_uri"},
            ),
        ),
        fit=dict(
            common=dict(
                instance_count={"instance_groups"},
                instance_type={"instance_groups"},
            ),
        ),
    ),
)


# A dict for parameters that should not be set to None since they are bonded with
# other parameters with None value. For example:
# if image_uri is None in TensorFlow, py_version should not be None
# since it's used as substitute argument to retrieve image_uri.
SET_PARAM_BONDED_WITH_NONE = dict(
    estimator=dict(
        init=dict(
            common=dict(),
            TensorFlow=dict(
                py_version={"image_uri"},
                framework_version={"image_uri"},
            ),
            HuggingFace=dict(
                transformers_version={"image_uri"},
                tensorflow_version={"pytorch_version"},
            ),
        )
    ),
    model=dict(
        register=dict(
            common=dict(
                inference_instances={"model_package_group_name"},
                transform_instances={"model_package_group_name"},
            ),
        )
    ),
    pipelinemodel=dict(
        register=dict(
            common=dict(
                inference_instances={"model_package_group_name"},
                transform_instances={"model_package_group_name"},
            ),
        )
    ),
)

# A dict for parameters that should not be set to None since they are bonded with
# other parameters with not None value. Thus we can skip it. For example:
# dimension_reduction_target should not be none when dimension_reduction_type is set
SET_PARAM_BONDED_WITH_NOT_NONE = dict(
    estimator=dict(
        init=dict(
            common=dict(),
            KNN=dict(dimension_reduction_target={"dimension_reduction_type"}),
        ),
    ),
)
