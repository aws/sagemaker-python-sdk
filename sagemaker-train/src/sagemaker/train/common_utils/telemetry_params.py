"""Shared telemetry param lists for sagemaker-train classes."""
from sagemaker.core.telemetry.telemetry_logging import TelemetryParamType

# Common params for SFT, DPO, RLVR, RLAIF trainers
BASE_TRAINER_TELEMETRY_PARAMS = [
    ("_model_name", TelemetryParamType.ATTR_VALUE),
    ("training_type", TelemetryParamType.ATTR_VALUE),
    ("networking", TelemetryParamType.ATTR_EXISTS),
    ("kms_key_id", TelemetryParamType.ATTR_EXISTS),
    ("mlflow_resource_arn", TelemetryParamType.ATTR_EXISTS),
    ("stopping_condition", TelemetryParamType.ATTR_EXISTS),
    ("validation_dataset", TelemetryParamType.KWARG_EXISTS),
    ("wait", TelemetryParamType.KWARG_EXISTS),
]

# Common params for all evaluators
BASE_EVALUATOR_TELEMETRY_PARAMS = [
    ("_base_model_name", TelemetryParamType.ATTR_VALUE),
    ("compute", TelemetryParamType.ATTR_TYPE),
    ("networking", TelemetryParamType.ATTR_EXISTS),
    ("kms_key_id", TelemetryParamType.ATTR_EXISTS),
    ("mlflow_resource_arn", TelemetryParamType.ATTR_EXISTS),
]
