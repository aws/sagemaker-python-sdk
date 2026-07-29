"""MTRL (Multi-Turn RL) SageMaker Pipelines templates.

Canonical templates for the MTRL Agentic Eval pipeline contract:
see Quip "MTRL Agentic Eval — SM Pipeline Template". Three template
shapes are exported:

* ``MTRL_TEMPLATE_BASE_MODEL_ONLY``   — evaluate a base model only.
* ``MTRL_TEMPLATE_FINE_TUNED_ONLY``   — evaluate a fine-tuned model only.
* ``MTRL_TEMPLATE``                   — base vs fine-tuned side-by-side.

Contract summary
----------------
* Step ``Type`` = ``"Job"``; ``Arguments.JobCategory = "AgentRFTEvaluation"``;
  ``Arguments.JobConfigSchemaVersion = "1.0.0"``.
* Input channel name = ``"evaluation"``.
* ``MlflowConfig`` lives inside
  ``JobConfigDocument.OutputDataConfig.MlflowConfig`` (not top-level).
* Distinct ``MlflowRunName`` per eval step: ``base-model-eval`` vs
  ``fine-tuned-model-eval`` — both runs land in the same experiment for
  side-by-side comparison.
* ``VpcConfig`` is at the step ``Arguments`` level (not inside the
  ``JobConfigDocument``).
* Hyperparameters (``eval_group_size``, ``sampling_temperature``,
  ``top_p``, ``max_tokens``, ``pass_k_values``, ``success_threshold``)
  are emitted as string values under ``EvaluationConfig.HyperParameters``.
* Lineage DAG:
  ``CreateEvaluationAction → (Evaluate…) → AssociateLineage``;
  ``AssociateLineage`` reads ``ServiceOutput.MlflowDetails.*`` from the
  eval step(s) via ``Get`` expressions.

Template context keys (all three templates consume the same context shape;
each template omits keys it does not need):

    pipeline_name, role_arn, base_model_arn, agent_arn, agent_qualifier,
    dataset_uri, s3_output_path, mlflow_resource_arn,
    mlflow_experiment_name, eval_group_size,
    sampling_temperature, top_p, max_tokens, pass_k_values,
    success_threshold, model_package_group_arn, source_model_package_arn,
    action_arn_prefix, dataset_artifact_arn, kms_key_arn, vpc_config,
    vpc_security_group_ids, vpc_subnets, tags
"""

from __future__ import absolute_import

# Canonical constants for the MTRL pipeline step contract.
_MTRL_STEP_TYPE = "Job"
_MTRL_JOB_CATEGORY = "AgentRFTEvaluation"
_MTRL_JOB_CONFIG_SCHEMA_VERSION = "1.0.0"

# --------------------------------------------------------------------------
# Shared template fragments (assembled into the three exported templates).
# --------------------------------------------------------------------------

# Lineage: CreateEvaluationAction. ``source_type_clause`` is either
# ``"SourceType": "Model"`` (base-only) or ``"SourceType": "ModelPackage"``.
# ``source_uri_expr`` is a Jinja expression producing the SourceUri value.
def _create_eval_action_step(source_uri_expr: str, source_type: str) -> str:
    return (
        '        {\n'
        '            "Name": "CreateEvaluationAction",\n'
        '            "Type": "Lineage",\n'
        '            "Arguments": {\n'
        '                "Actions": [\n'
        '                    {\n'
        '                        "ActionName": { "Get": "Execution.PipelineExecutionId" },\n'
        '                        "ActionType": "Evaluation",\n'
        '                        "Source": {\n'
        f'                            "SourceUri": {source_uri_expr},\n'
        f'                            "SourceType": "{source_type}"\n'
        '                        },\n'
        '                        "Properties": {\n'
        '                            "PipelineExecutionArn": { "Get": "Execution.PipelineExecutionArn" },\n'
        '                            "PipelineName": "{{ pipeline_name }}"\n'
        '                        }\n'
        '                    }\n'
        '                ],\n'
        '                "Contexts": [\n'
        '                    {\n'
        '                        "ContextName": { "Get": "Execution.PipelineExecutionId" },\n'
        '                        "ContextType": "PipelineExecution",\n'
        '                        "Source": { "SourceUri": { "Get": "Execution.PipelineExecutionArn" } }\n'
        '                    }\n'
        '                ],\n'
        '                "Associations": [\n'
        '                    {\n'
        '                        "Source": { "Name": { "Get": "Execution.PipelineExecutionId" }, "Type": "Action" },\n'
        '                        "Destination": { "Name": { "Get": "Execution.PipelineExecutionId" }, "Type": "Context" },\n'
        '                        "AssociationType": "ContributedTo"\n'
        '                    }{% if dataset_artifact_arn %},\n'
        '                    {\n'
        '                        "Source": { "Arn": "{{ dataset_artifact_arn }}" },\n'
        '                        "Destination": {\n'
        '                            "Arn": { "Std:Join": { "On": "/", "Values": [\n'
        '                                "{{ action_arn_prefix }}",\n'
        '                                { "Get": "Execution.PipelineExecutionId" }\n'
        '                            ] } }\n'
        '                        },\n'
        '                        "AssociationType": "ContributedTo"\n'
        '                    }{% endif %}\n'
        '                ]\n'
        '            }\n'
        '        }'
    )



# Eval step: emits one ``Job``-typed step (base or fine-tuned). The caller
# supplies the step name, the mlflow run name, an optional ModelPackageConfig
# block, and the DependsOn step name.
def _eval_step(step_name: str, mlflow_run_name: str, include_mpc: bool, depends_on: str) -> str:
    """Build a Job step that takes a pre-stringified JobConfigDocument.

    The template expects ``job_config_document_str`` (for base-model steps)
    or ``job_config_document_ft_str`` (for fine-tuned steps) in the Jinja2
    context — both are JSON **strings** (double-encoded), not dicts.
    """
    # Pick the context variable name based on whether this is a fine-tuned step.
    doc_var = "job_config_document_ft_str" if include_mpc else "job_config_document_str"
    return (
        '        {\n'
        f'            "Name": "{step_name}",\n'
        '            "Type": "Job",\n'
        f'            "DependsOn": ["{depends_on}"],\n'
        '            "Arguments": {\n'
        '                "JobCategory": "AgentRFTEvaluation",\n'
        '                "RoleArn": "{{ role_arn }}",\n'
        '                "JobConfigSchemaVersion": "1.0.0",\n'
        f'                "JobConfigDocument": {{{{ {doc_var} | tojson }}}}'
        '{% if vpc_config %},\n'
        '                "VpcConfig": {\n'
        '                    "SecurityGroupIds": {{ vpc_security_group_ids | tojson }},\n'
        '                    "Subnets": {{ vpc_subnets | tojson }}\n'
        '                }{% endif %}{% if tags %},\n'
        '                "Tags": {{ tags | tojson }}{% endif %}\n'
        '            }\n'
        '        }'
    )



# Lineage: AssociateLineage. ``artifact_names`` is a list of ``(label, run_step)``
# tuples — one per eval step — used to build artifact entries and associations.
def _associate_lineage_step(artifact_entries, depends_on: str) -> str:
    artifacts = []
    associations = []
    for label, run_step in artifact_entries:
        artifacts.append(
            '                    {\n'
            '                        "ArtifactName": { "Std:Join": { "On": "-", "Values": [\n'
            '                            { "Get": "Execution.PipelineExecutionId" },\n'
            f'                            "{label}"\n'
            '                        ] } },\n'
            '                        "ArtifactType": "EvaluationReport",\n'
            f'                        "Source": {{ "SourceUri": {{ "Get": "Steps.{run_step}.JobConfigDocument.ServiceOutput.MlflowDetails.RunId" }} }},\n'
            '                        "Properties": {\n'
            f'                            "MlflowExperimentId": {{ "Get": "Steps.{run_step}.JobConfigDocument.ServiceOutput.MlflowDetails.ExperimentId" }},\n'
            f'                            "MlflowRunName": {{ "Get": "Steps.{run_step}.JobConfigDocument.ServiceOutput.MlflowDetails.RunName" }}\n'
            '                        }\n'
            '                    }'
        )
        associations.append(
            '                    {\n'
            '                        "Source": {\n'
            '                            "Name": { "Std:Join": { "On": "-", "Values": [\n'
            '                                { "Get": "Execution.PipelineExecutionId" },\n'
            f'                                "{label}"\n'
            '                            ] } },\n'
            '                            "Type": "Artifact"\n'
            '                        },\n'
            '                        "Destination": {\n'
            '                            "Arn": { "Std:Join": { "On": "/", "Values": [\n'
            '                                "{{ action_arn_prefix }}",\n'
            '                                { "Get": "Execution.PipelineExecutionId" }\n'
            '                            ] } }\n'
            '                        },\n'
            '                        "AssociationType": "ContributedTo"\n'
            '                    }'
        )
    return (
        '        {\n'
        '            "Name": "AssociateLineage",\n'
        '            "Type": "Lineage",\n'
        f'            "DependsOn": ["{depends_on}"],\n'
        '            "Arguments": {\n'
        '                "Artifacts": [\n'
        + ',\n'.join(artifacts) + '\n'
        '                ],\n'
        '                "Associations": [\n'
        + ',\n'.join(associations) + '\n'
        '                ]\n'
        '            }\n'
        '        }'
    )


def _pipeline(steps) -> str:
    """Wrap a list of rendered step strings into a full pipeline definition."""
    return (
        '{\n'
        '    "Version": "2020-12-01",\n'
        '    "Metadata": {},\n'
        '    "Parameters": [],\n'
        '    "Steps": [\n'
        + ',\n'.join(steps) + '\n'
        '    ]\n'
        '}'
    )


# --------------------------------------------------------------------------
# Exported templates.
# --------------------------------------------------------------------------

# Base model only: evaluate a base JumpStart / hub model without any fine-tuned
# comparison. DAG: CreateEvaluationAction → EvaluateBaseModel → AssociateLineage.
MTRL_TEMPLATE_BASE_MODEL_ONLY = _pipeline([
    _create_eval_action_step(source_uri_expr='"{{ base_model_arn }}"', source_type="Model"),
    _eval_step(step_name="EvaluateBaseModel", mlflow_run_name="base-model-eval",
               include_mpc=False, depends_on="CreateEvaluationAction"),
    _associate_lineage_step(
        artifact_entries=[("base-eval-report", "EvaluateBaseModel")],
        depends_on="EvaluateBaseModel",
    ),
])


# Fine-tuned model only: evaluate a fine-tuned model without a base comparison.
# DAG: CreateEvaluationAction → EvaluateFineTunedModel → AssociateLineage.
MTRL_TEMPLATE_FINE_TUNED_ONLY = _pipeline([
    _create_eval_action_step(
        source_uri_expr='"{{ source_model_package_arn }}"',
        source_type="ModelPackage",
    ),
    _eval_step(step_name="EvaluateFineTunedModel", mlflow_run_name="fine-tuned-model-eval",
               include_mpc=True, depends_on="CreateEvaluationAction"),
    _associate_lineage_step(
        artifact_entries=[("fine-tuned-eval-report", "EvaluateFineTunedModel")],
        depends_on="EvaluateFineTunedModel",
    ),
])


# Comparison: evaluate both the base model and the fine-tuned model in a single
# pipeline. Both eval steps share the same MLflow experiment but use distinct
# run names (``base-model-eval`` / ``fine-tuned-model-eval``). DAG:
# CreateEvaluationAction → EvaluateBaseModel → EvaluateFineTunedModel → AssociateLineage.
MTRL_TEMPLATE = _pipeline([
    _create_eval_action_step(
        source_uri_expr='"{{ source_model_package_arn }}"',
        source_type="ModelPackage",
    ),
    _eval_step(step_name="EvaluateBaseModel", mlflow_run_name="base-model-eval",
               include_mpc=False, depends_on="CreateEvaluationAction"),
    _eval_step(step_name="EvaluateFineTunedModel", mlflow_run_name="fine-tuned-model-eval",
               include_mpc=True, depends_on="EvaluateBaseModel"),
    _associate_lineage_step(
        artifact_entries=[
            ("base-eval-report", "EvaluateBaseModel"),
            ("fine-tuned-eval-report", "EvaluateFineTunedModel"),
        ],
        depends_on="EvaluateFineTunedModel",
    ),
])


__all__ = [
    "MTRL_TEMPLATE_BASE_MODEL_ONLY",
    "MTRL_TEMPLATE_FINE_TUNED_ONLY",
    "MTRL_TEMPLATE",
]
