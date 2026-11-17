"""Request wiring for managed pipeline fields on Pipeline and PipelineExecution."""

from unittest.mock import MagicMock, patch

from sagemaker.core.resources import Base, Pipeline, PipelineExecution

MANAGED_PIPELINE_ARN = "arn:aws:sagemaker:us-west-2:aws:pipeline/aws-system-pipeline-demo"
ROLE_ARN = "arn:aws:iam::123456789012:role/ExecutionRole"


def _list_pipelines(managed_by=None):
    client = MagicMock()
    client.list_pipelines.return_value = {"PipelineSummaries": []}
    kwargs = {} if managed_by is None else {"managed_by": managed_by}
    with patch.object(Base, "get_sagemaker_client", return_value=client):
        list(Pipeline.get_all(**kwargs))
    return client.list_pipelines.call_args.kwargs


def test_get_all_sends_managed_by():
    assert _list_pipelines(managed_by="AWS")["ManagedBy"] == "AWS"


def test_get_all_omits_managed_by_when_unset():
    assert "ManagedBy" not in _list_pipelines()


def _start_execution(**fields):
    client = MagicMock()
    execution = PipelineExecution(
        pipeline_execution_arn=f"{MANAGED_PIPELINE_ARN}/execution/abc", **fields
    )
    with patch.object(PipelineExecution, "_get_client", return_value=client):
        execution.start(pipeline_name=MANAGED_PIPELINE_ARN, client_request_token="token")
    return client.start_pipeline_execution.call_args.kwargs


def test_start_sends_role_arn():
    assert _start_execution(role_arn=ROLE_ARN)["RoleArn"] == ROLE_ARN


def test_start_omits_role_arn_when_unset():
    assert "RoleArn" not in _start_execution()


def test_get_all_describes_listed_managed_pipeline_by_arn():
    # The name alone resolves in the caller's account, where a managed pipeline does not
    # exist, so iterating get_all results must describe each pipeline by its ARN.
    client = MagicMock()
    client.list_pipelines.return_value = {
        "PipelineSummaries": [
            {"PipelineArn": MANAGED_PIPELINE_ARN, "PipelineName": "aws-system-pipeline-demo"}
        ]
    }
    client.describe_pipeline.return_value = {
        "PipelineArn": MANAGED_PIPELINE_ARN,
        "PipelineName": "aws-system-pipeline-demo",
    }
    with patch.object(Base, "get_sagemaker_client", return_value=client):
        pipelines = list(Pipeline.get_all(managed_by="AWS"))
    client.describe_pipeline.assert_called_once_with(PipelineName=MANAGED_PIPELINE_ARN)
    assert [p.pipeline_arn for p in pipelines] == [MANAGED_PIPELINE_ARN]


def test_unfiltered_get_all_describes_returned_managed_pipeline_by_arn():
    # Without managed_by, the service can return managed and customer pipelines together.
    # Each must be described by its ARN, and ManagedBy must not be sent.
    customer_arn = "arn:aws:sagemaker:us-west-2:123456789012:pipeline/my-pipeline"
    client = MagicMock()
    client.list_pipelines.return_value = {
        "PipelineSummaries": [
            {"PipelineArn": MANAGED_PIPELINE_ARN, "PipelineName": "aws-system-pipeline-demo"},
            {"PipelineArn": customer_arn, "PipelineName": "my-pipeline"},
        ]
    }
    client.describe_pipeline.side_effect = lambda PipelineName: {
        "PipelineArn": PipelineName,
        "PipelineName": PipelineName.rsplit("/", 1)[-1],
    }
    with patch.object(Base, "get_sagemaker_client", return_value=client):
        pipelines = list(Pipeline.get_all())
    assert "ManagedBy" not in client.list_pipelines.call_args.kwargs
    assert [c.kwargs["PipelineName"] for c in client.describe_pipeline.call_args_list] == [
        MANAGED_PIPELINE_ARN,
        customer_arn,
    ]
    assert [p.pipeline_arn for p in pipelines] == [MANAGED_PIPELINE_ARN, customer_arn]


def _refresh(**fields):
    client = MagicMock()
    client.describe_pipeline.return_value = {}
    pipeline = Pipeline.model_construct(**fields)
    with patch.object(Pipeline, "_get_client", return_value=client):
        pipeline.refresh()
    return client.describe_pipeline.call_args.kwargs


def test_refresh_uses_arn_when_assigned():
    assert (
        _refresh(pipeline_name="aws-system-pipeline-demo", pipeline_arn=MANAGED_PIPELINE_ARN)[
            "PipelineName"
        ]
        == MANAGED_PIPELINE_ARN
    )


def test_refresh_falls_back_to_name_without_arn():
    from sagemaker.core.utils.utils import Unassigned

    assert (
        _refresh(pipeline_name="my-pipeline", pipeline_arn=Unassigned())["PipelineName"]
        == "my-pipeline"
    )
