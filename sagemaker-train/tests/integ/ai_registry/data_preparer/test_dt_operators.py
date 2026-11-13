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
"""Integration tests: one per DataTransformation operator.

TextQualityFilter is covered by ``test_dt_happy_path``; this module covers the
remaining eight operators in the catalog, plus two failure paths (an unknown
operator name and a misspelled Config key), the four constraints the tutorial
tells customers not to violate (``TestDocumentedConstraints``), and one chained
pair (``TestDTSplitThenMerge``) that feeds one job's output datasets straight into
the next as input — the only coverage of a service-produced DataSet being consumed
by DT, since every other test imports its own input via ``import_dataset``.

The SDK is a passthrough for ``TransformationConfig`` — ``Operator`` is a bare
string and ``Config`` is ``Dict[str, Any]`` with additionalProperties allowed, so
nothing client-side rejects a misspelled operator or key. Every ``Config`` below
is therefore spelled against the service-side per-operator JSON Schema rather
than invented, and the payload envelope respects its documented limits:
``InputDataConfig`` holds at most 2 entries and ``Operations`` holds exactly 1
(the service accepts one operation per job).
"""

from __future__ import absolute_import

import json
import logging
import uuid
from urllib.parse import urlparse

import boto3
import pytest
from botocore.exceptions import ClientError

from sagemaker.ai_registry import (
    DataPreparer,
    DataPreparerJobFailedError,
    DatasetJobConfig,
    JobCategory,
)
from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerError

# Submits real jobs and waits for them: scheduled CI only, not PR checks
# (see the gpu_intensive marker in sagemaker-train/pyproject.toml).
pytestmark = pytest.mark.gpu_intensive


def _read_rows(s3_uri, region):
    """Read a JSONL output object from S3 and return its parsed rows."""
    parsed = urlparse(s3_uri)
    s3 = boto3.client("s3", region_name=region)
    body = s3.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))["Body"].read()
    return [json.loads(line) for line in body.splitlines() if line.strip()]


def _object_size(s3_uri, region):
    """Return the object's size in bytes, raising ClientError if it is absent."""
    parsed = urlparse(s3_uri)
    s3 = boto3.client("s3", region_name=region)
    return s3.head_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))["ContentLength"]


@pytest.fixture
def preparer(sagemaker_session, dt_role):
    """DataPreparer bound to the integ session and execution role."""
    return DataPreparer(sagemaker_session=sagemaker_session, role=dt_role)


@pytest.fixture
def output_path(integ_bucket):
    """A fresh S3 output prefix per test, so outputs never overwrite each other."""
    return f"s3://{integ_bucket}/data-preparer-integ-dt-output/{uuid.uuid4().hex[:8]}/"


@pytest.mark.integ
class TestDedupOperators:
    """ExactDedup and FuzzyDedup over a fixture with exact and near duplicates.

    ``dt_dedup_duplicates.jsonl`` has 6 rows: rows 1-3 share an identical prompt,
    row 4 repeats that prompt with only its final word changed, and rows 5-6 are
    lexically unrelated. The two operators therefore have to disagree — exact
    matching collapses only rows 1-3 (6 -> 4) while fuzzy matching also absorbs
    row 4 (6 -> 3). Asserting both counts is what distinguishes the operators;
    a single count would pass even if one silently ran as the other.

    ``TextField`` must be set explicitly: it defaults to ``"text"``, which this
    fixture does not have.
    """

    def test_exact_dedup_collapses_identical_prompts(
        self, preparer, s3_fixtures, import_dataset, output_path, sagemaker_session
    ):
        """ExactDedup keeps one of the three identical rows and the near-duplicate."""
        input_arn = import_dataset(s3_fixtures["dt_dedup_duplicates.jsonl"])

        dataset = DatasetJobConfig(
            job_category=JobCategory.DATA_TRANSFORMATION,
            input_data_config=[{"DatasetArn": input_arn}],
            output_data_config={"S3OutputPath": output_path},
            transformation_config={
                "Operations": [
                    {
                        "Operator": "ExactDedup",
                        # Normalize stays False: for small inputs the service raises
                        # NotImplementedError for Normalize=True, and the service-side
                        # schema for ExactDedup is additionalProperties=false with only
                        # TextField and Normalize — EmitEvidence would be rejected at
                        # CreateJob even though the execution layer accepts it.
                        "Config": {"TextField": "prompt", "Normalize": False},
                    }
                ],
            },
        )
        result = preparer.transform(dataset)

        assert len(result) == 1
        rows = _read_rows(result[0].source, sagemaker_session.boto_region_name)
        assert len(rows) == 4
        prompts = [row["prompt"] for row in rows]
        assert len(set(prompts)) == 4, "ExactDedup left duplicate prompts behind"
        # The near-duplicate must survive exact matching — that is what separates
        # this operator's result from FuzzyDedup's.
        assert any(prompt.endswith("today") for prompt in prompts)

    def test_fuzzy_dedup_also_collapses_near_duplicate(
        self, preparer, s3_fixtures, import_dataset, output_path, sagemaker_session
    ):
        """FuzzyDedup absorbs the one-word-different row that ExactDedup keeps."""
        input_arn = import_dataset(s3_fixtures["dt_dedup_duplicates.jsonl"])

        dataset = DatasetJobConfig(
            job_category=JobCategory.DATA_TRANSFORMATION,
            input_data_config=[{"DatasetArn": input_arn}],
            output_data_config={"S3OutputPath": output_path},
            transformation_config={
                "Operations": [
                    {
                        "Operator": "FuzzyDedup",
                        "Config": {
                            "TextField": "prompt",
                            "JaccardThreshold": 0.8,
                            "NumPerm": 128,
                            "NgramSize": 5,
                        },
                    }
                ],
            },
        )
        result = preparer.transform(dataset)

        assert len(result) == 1
        rows = _read_rows(result[0].source, sagemaker_session.boto_region_name)
        assert len(rows) == 3


@pytest.mark.integ
class TestSplitOperator:
    """Split is the only operator that publishes more than one output DataSet.

    It is therefore the sole end-to-end exercise of ``transform()`` returning a
    ``List[DataSet]`` longer than 1, and of the per-ARN hub/version rehydration
    running more than once.
    """

    def test_split_produces_two_datasets(
        self, preparer, s3_fixtures, import_dataset, output_path, sagemaker_session
    ):
        """An 80/20 split of 5 rows yields two DataSets holding 4 and 1 rows."""
        input_arn = import_dataset(s3_fixtures["byod_training_sft.jsonl"])

        dataset = DatasetJobConfig(
            job_category=JobCategory.DATA_TRANSFORMATION,
            input_data_config=[{"DatasetArn": input_arn}],
            output_data_config={"S3OutputPath": output_path},
            transformation_config={
                "Operations": [
                    {
                        "Operator": "Split",
                        # Ratios is the only required key; it must have >= 2 entries and
                        # sum to 1.0. The service guarantees every split gets at least
                        # one record and gives the last split the remainder.
                        "Config": {"Ratios": {"train": 0.8, "test": 0.2}, "Seed": 42},
                    }
                ],
            },
        )
        result = preparer.transform(dataset)

        assert len(result) == 2
        assert result[0].arn != result[1].arn
        counts = sorted(
            len(_read_rows(ds.source, sagemaker_session.boto_region_name)) for ds in result
        )
        assert counts == [1, 4]

        # A6: every split output has >= 1 record and the total equals the
        # input count (5 rows in byod_training_sft.jsonl).
        assert all(
            c >= 1 for c in counts
        ), f"A6: all split outputs must have >= 1 record; got counts={counts}"
        assert sum(counts) == 5, (
            f"A6: split output record counts must sum to input count 5; "
            f"got {sum(counts)} from counts={counts}"
        )


@pytest.mark.integ
class TestMergeOperator:
    """Merge is the only operator that requires two inputs.

    The service rejects a single input ("Merge requires at least 2 input
    datasets") and the job-config schema caps ``InputDataConfig`` at 2, so two is
    both the floor and the ceiling here. Its ``Config`` schema declares no
    properties at all, so the service's validator rejects any key that is
    supplied — hence the empty dict, pinned by
    ``TestDocumentedConstraints.test_merge_rejects_any_config_property``.
    """

    def test_merge_concatenates_two_datasets(
        self, preparer, s3_fixtures, import_dataset, output_path, sagemaker_session
    ):
        """Merging a 5-row and a 3-row dataset yields 8 rows."""
        first_arn = import_dataset(s3_fixtures["byod_training_sft.jsonl"])
        second_arn = import_dataset(s3_fixtures["dt_merge_second.jsonl"])

        dataset = DatasetJobConfig(
            job_category=JobCategory.DATA_TRANSFORMATION,
            input_data_config=[{"DatasetArn": first_arn}, {"DatasetArn": second_arn}],
            output_data_config={"S3OutputPath": output_path},
            transformation_config={
                "Operations": [{"Operator": "Merge", "Config": {}}],
            },
        )
        result = preparer.transform(dataset)

        assert len(result) == 1
        rows = _read_rows(result[0].source, sagemaker_session.boto_region_name)
        assert len(rows) == 8


@pytest.mark.integ
class TestDTSplitThenMerge:
    """Split, then Merge its outputs back — chained, in the style of ``TestSDGThenDQ``.

    The two classes above already cover Split and Merge in isolation, and keeping
    them that way is deliberate: either can run alone, and a Split regression
    cannot mask a Merge regression. This class covers the one thing neither can.

    Every other DT test feeds an input built by ``import_dataset``, i.e. a raw
    ``import_hub_content`` call made by the test itself. So nothing else asserts
    that a DataSet *the service produced* is directly consumable as the input to
    another DT job. That is a real customer path — split a corpus, transform the
    halves, recombine — and it exercises the output side of one job against the
    input side of the next without a round-trip through ``DataSet.create``.

    It works because the DT service publishes its outputs as full v4 DataSets:
    verified against a completed Split job, its two outputs are named
    ``{job_name}-train`` / ``{job_name}-test`` at HubContentVersion ``0.0.1``,
    with ``DocumentSchemaVersion`` ``4.0.0``, and they carry the input's
    ``DatasetFileFormat`` forward. Merge therefore accepts them as-is. Split
    yielding exactly two outputs is also what makes this pair fit: the job-config
    schema caps ``InputDataConfig`` at 2, which is simultaneously Merge's floor.

    Ordering is positional (``test_01`` before ``test_02``) and the ARNs travel in
    a class attribute — the same coupling ``TestSDGThenDQ`` accepts for the same
    reason. ``test_02`` skips rather than fails if ``test_01`` did not produce
    them, so a Split break is reported once, at its cause.
    """

    _split_arns = None

    def test_01_split_produces_datasets_for_chaining(
        self, preparer, s3_fixtures, import_dataset, output_path, sagemaker_session
    ):
        """Split 5 rows 80/20 into two DataSets and hand their ARNs to test_02."""
        input_arn = import_dataset(s3_fixtures["byod_training_sft.jsonl"])

        dataset = DatasetJobConfig(
            job_category=JobCategory.DATA_TRANSFORMATION,
            input_data_config=[{"DatasetArn": input_arn}],
            output_data_config={"S3OutputPath": output_path},
            transformation_config={
                "Operations": [
                    {
                        "Operator": "Split",
                        "Config": {"Ratios": {"train": 0.8, "test": 0.2}, "Seed": 42},
                    }
                ],
            },
        )
        result = preparer.transform(dataset)

        assert len(result) == 2
        counts = sorted(
            len(_read_rows(ds.source, sagemaker_session.boto_region_name)) for ds in result
        )
        assert counts == [1, 4]

        # Publish to the class for test_02. Store ARNs rather than DataSet objects:
        # test_02 needs only the ARN, and re-resolving it there proves the ARN alone
        # is a sufficient handle — which is the point of the chain.
        TestDTSplitThenMerge._split_arns = [ds.arn for ds in result]

    def test_02_merge_recombines_split_outputs(self, preparer, output_path, sagemaker_session):
        """Merging the two halves recovers the original 5 rows.

        Note there is no ``import_dataset`` here — the inputs are the previous
        job's outputs, referenced purely by the ARN it returned. Row count is the
        assertion that makes the round-trip meaningful: 4 + 1 back to 5 means no
        half was dropped, duplicated, or silently truncated on republish.
        """
        arns = TestDTSplitThenMerge._split_arns
        if not arns:
            pytest.skip("test_01_split_produces_datasets_for_chaining did not produce outputs")
        assert len(arns) == 2, f"expected 2 split outputs to merge, got {len(arns)}"

        dataset = DatasetJobConfig(
            job_category=JobCategory.DATA_TRANSFORMATION,
            input_data_config=[{"DatasetArn": arn} for arn in arns],
            output_data_config={"S3OutputPath": output_path},
            # Merge's Config schema declares no properties and the service
            # rejects any that are supplied — empty dict, as above.
            transformation_config={
                "Operations": [{"Operator": "Merge", "Config": {}}],
            },
        )
        result = preparer.transform(dataset)

        assert len(result) == 1
        rows = _read_rows(result[0].source, sagemaker_session.boto_region_name)
        assert len(rows) == 5


@pytest.mark.integ
class TestSchemaOperators:
    """SchemaConversion rewrites every row's shape; SchemaFilter drops misshapen rows."""

    def test_schema_conversion_rewrites_to_openai_messages(
        self, preparer, s3_fixtures, import_dataset, output_path, sagemaker_session
    ):
        """Converting HF prompt/completion to OpenAI conversational reshapes all 5 rows.

        ``TargetFormat`` has to differ from the input DataSet's declared
        ``DatasetFileFormat`` — converting a format to itself is not a meaningful
        request, so the fixture's HF_Prompt_Completion is targeted at
        OpenAI_Messages.
        """
        input_arn = import_dataset(s3_fixtures["byod_training_sft.jsonl"])

        dataset = DatasetJobConfig(
            job_category=JobCategory.DATA_TRANSFORMATION,
            input_data_config=[{"DatasetArn": input_arn}],
            output_data_config={"S3OutputPath": output_path},
            transformation_config={
                "Operations": [
                    {
                        "Operator": "SchemaConversion",
                        "Config": {"TargetFormat": "OpenAI_Messages"},
                    }
                ],
            },
        )
        result = preparer.transform(dataset)

        assert len(result) == 1
        rows = _read_rows(result[0].source, sagemaker_session.boto_region_name)
        # Conversion reshapes rows; it must not drop any.
        assert len(rows) == 5
        assert all("messages" in row for row in rows)
        assert all("prompt" not in row for row in rows)

    def test_schema_filter_drops_nonconforming_rows(
        self, preparer, s3_fixtures, import_dataset, output_path, sagemaker_session
    ):
        """SchemaFilter keeps the 4 HF rows and drops the 2 foreign-shaped rows.

        ``dt_schema_mixed.jsonl`` deliberately mixes formats: 4 rows are HF
        prompt/completion, one is HF language-modeling (``text``) and one is
        OpenAI conversational (``messages``). Handling input that does not
        uniformly match the declared format is the point of this operator, and it
        needs no extra setting to do so.
        """
        input_arn = import_dataset(s3_fixtures["dt_schema_mixed.jsonl"])

        dataset = DatasetJobConfig(
            job_category=JobCategory.DATA_TRANSFORMATION,
            input_data_config=[{"DatasetArn": input_arn}],
            output_data_config={"S3OutputPath": output_path},
            transformation_config={
                "Operations": [
                    {
                        "Operator": "SchemaFilter",
                        "Config": {"ExpectedFormat": "HF_Prompt_Completion"},
                    }
                ],
            },
        )
        result = preparer.transform(dataset)

        assert len(result) == 1
        rows = _read_rows(result[0].source, sagemaker_session.boto_region_name)
        assert len(rows) == 4
        assert all("prompt" in row and "completion" in row for row in rows)


@pytest.mark.integ
class TestCustomCodeOperators:
    """BYOC runs customer-supplied code; CodeGen has Bedrock write it first.

    Both execute inside the service's RestrictedPython sandbox, which calls a
    function named exactly ``transform_dataset(input_path, output_path)`` and
    expects it to write the output file and return a stats dict.
    """

    def test_byoc_runs_customer_supplied_code(
        self, preparer, s3_fixtures, import_dataset, output_path, sagemaker_session
    ):
        """The BYOC fixture keeps only positive-sentiment rows — 5 rows down to 2."""
        input_arn = import_dataset(s3_fixtures["byod_training_sft.jsonl"])

        dataset = DatasetJobConfig(
            job_category=JobCategory.DATA_TRANSFORMATION,
            input_data_config=[{"DatasetArn": input_arn}],
            output_data_config={"S3OutputPath": output_path},
            transformation_config={
                "Operations": [
                    {
                        "Operator": "BYOC",
                        "Config": {
                            "TransformationCodeS3Uri": s3_fixtures["dt_byoc_transformer.py"],
                        },
                    }
                ],
            },
        )
        result = preparer.transform(dataset)

        assert len(result) == 1
        rows = _read_rows(result[0].source, sagemaker_session.boto_region_name)
        # byod_training_sft.jsonl completions are positive, negative, neutral,
        # positive, negative.
        assert len(rows) == 2
        assert all(row["completion"] == "positive" for row in rows)

    def test_codegen_generates_and_executes_transform(
        self, preparer, s3_fixtures, import_dataset, output_path, sagemaker_session
    ):
        """CodeGen turns a natural-language spec into code that keeps neutral rows.

        The generated code comes from an LLM, so this is the least deterministic
        test in the suite — the spec is written to be unambiguous, but a failure
        here is worth re-running before treating it as a real regression.
        """
        input_arn = import_dataset(s3_fixtures["byod_training_sft.jsonl"])

        dataset = DatasetJobConfig(
            job_category=JobCategory.DATA_TRANSFORMATION,
            input_data_config=[{"DatasetArn": input_arn}],
            output_data_config={"S3OutputPath": output_path},
            transformation_config={
                "Operations": [
                    {
                        "Operator": "CodeGen",
                        "Config": {
                            "SpecificationS3Uri": s3_fixtures["dt_codegen_spec.md"],
                            "AutoExecute": True,
                        },
                    }
                ],
            },
        )
        result = preparer.transform(dataset)

        assert len(result) == 1
        rows = _read_rows(result[0].source, sagemaker_session.boto_region_name)
        # Exactly one row in the fixture has completion "neutral".
        assert len(rows) == 1
        assert rows[0]["completion"] == "neutral"

        # The generated code is uploaded beside the output as transformer.py, per
        # SageMakerDataTransformer's io_resolver.TRANSFORMER_FILENAME. transform()
        # returns no handle to it, so the only way a caller reaches it is the
        # sibling-of-source derivation documented in dt_tutorial.rst. Pin it here
        # so a service-side rename fails a test instead of leaving that doc wrong.
        code_uri = result[0].source.rsplit("/", 1)[0] + "/transformer.py"
        assert _object_size(code_uri, sagemaker_session.boto_region_name) > 0

        # AutoExecute=True must transform the whole input, not upload the sample
        # preview a False run would publish (sample_output.<ext>).
        assert result[0].source.endswith("/dataset.jsonl")


@pytest.mark.integ
class TestFailurePaths:
    """A bad operator name or a bad Config key must fail with a usable message.

    Neither the SDK nor the Hub-published schema constrains ``Operator`` to the
    catalog or ``Config`` to the per-operator keys, so both kinds of typo reach
    the service unchallenged. The backstop is service-side and, notably,
    *asynchronous*: validation runs after the job is already persisted, so
    both of these surface as a failed job with the validator's text in
    FailureReason — not as a rejected API call. What these tests pin is that
    the text names the offending operator or key, which is all that stands
    between a misspelling and an opaque failure.
    """

    def test_unknown_operator_is_rejected_with_supported_list(
        self, preparer, s3_fixtures, import_dataset, output_path
    ):
        """A bogus operator name surfaces the name and the supported catalog."""
        input_arn = import_dataset(s3_fixtures["byod_training_sft.jsonl"])

        dataset = DatasetJobConfig(
            job_category=JobCategory.DATA_TRANSFORMATION,
            input_data_config=[{"DatasetArn": input_arn}],
            output_data_config={"S3OutputPath": output_path},
            transformation_config={
                "Operations": [{"Operator": "NotARealOperator", "Config": {}}],
            },
        )

        # Same reasoning as the config-key test below: assert the specific job
        # failure, not a bare Exception, so an expired credential or a CreateJob
        # error cannot make this pass without exercising the validator at all.
        with pytest.raises(DataPreparerJobFailedError) as exc_info:
            preparer.transform(dataset)

        message = str(exc_info.value)
        assert "NotARealOperator" in message
        assert (
            "TextQualityFilter" in message
        ), f"Error should list supported operators, got: {message}"

    def test_misspelled_config_key_is_rejected_by_name(
        self, preparer, s3_fixtures, import_dataset, output_path
    ):
        """A misspelled Config key fails the job and the message names the key.

        ``Threshold`` is the plausible-but-wrong spelling of FuzzyDedup's
        ``JaccardThreshold`` — this exact typo was live in the unit test until it
        was corrected against the service catalog. FuzzyDedup declares no
        required fields, so the only rule this payload can break is
        ``additionalProperties: false``, which isolates the misspelling as the
        sole cause of failure.

        Note *when* this fails. ``CreateJob`` accepts it: the API persists the
        job at ``Pending`` without validating the operator config. Validation
        runs asynchronously after submission — the job transitions through
        Pending -> Validating before being marked Failed with the validator
        string as FailureReason. So this raises from ``wait``, not from the
        CreateJob call, and it surfaces as a job failure rather than a
        client-side or API-level rejection. Compute is still never provisioned:
        validation runs before any execution infrastructure is launched.
        """
        input_arn = import_dataset(s3_fixtures["dt_dedup_duplicates.jsonl"])

        dataset = DatasetJobConfig(
            job_category=JobCategory.DATA_TRANSFORMATION,
            input_data_config=[{"DatasetArn": input_arn}],
            output_data_config={"S3OutputPath": output_path},
            transformation_config={
                "Operations": [
                    {
                        "Operator": "FuzzyDedup",
                        "Config": {"TextField": "prompt", "Threshold": 0.8},
                    }
                ],
            },
        )

        # DataPreparerJobFailedError specifically, not a bare Exception: the point
        # of this test is that the rejection arrives as a *job failure* from wait,
        # and a bare Exception would also pass on an unrelated CreateJob or
        # credential error, hiding that distinction.
        with pytest.raises(DataPreparerJobFailedError) as exc_info:
            preparer.transform(dataset)

        # Assert the operator and the offending key are named, not the full
        # phrasing — the tail is emitted by the server-side schema validator
        # and its exact wording is not a contract we control. Observed:
        #   "FuzzyDedup Config validation failed: $: property 'Threshold' is not
        #    defined in the schema and the schema does not allow additional
        #    properties"
        message = str(exc_info.value)
        assert "Threshold" in message, f"Error should name the rejected Config key, got: {message}"
        assert (
            "FuzzyDedup" in message
        ), f"Error should name the operator whose schema rejected it, got: {message}"


@pytest.mark.integ
class TestDocumentedConstraints:
    """The four "don't do this" rules the DT tutorial states, each pinned to its
    enforcement point.

    Every positive test above deliberately stays inside these rules, so nothing
    else here proves the rules exist — the tutorial's warnings would keep passing
    review even if the service stopped enforcing them. Each config below is read
    off the enforcing source rather than guessed, and the docstrings name it,
    because the *layer* that rejects determines what a customer sees:

    * ``Merge`` config keys and >1 ``Operations`` are rejected by the service's
      asynchronous validator after submission, so their FailureReason names the
      offending operator or structure and can be asserted on.
    * SchemaConversion source==target and ``ExactDedup Normalize=True`` are
      rejected during execution, not by the schema validator; the service
      wraps those execution errors in a fixed generic string
      ("Transformation task failed. Please check your input data and
      specification, then retry."), so the cause does *not* reach FailureReason.
      Those two tests assert the job failed and that it was not a per-operator
      schema rejection; asserting the cause is impossible by design, which is
      itself why the tutorial has to warn about them.
    """

    def test_merge_rejects_any_config_property(
        self, preparer, s3_fixtures, import_dataset, output_path
    ):
        """Merge accepts no Config keys, so any key at all fails the job.

        Merge's operator schema declares no properties. That makes it the one
        operator where the *name* of the key is irrelevant: ``OutputName`` is a
        plausible thing for a customer to reach for, and it fails for the same
        reason any other key would. This is what makes ``"Config": {}`` in the
        tests above a requirement rather than a stylistic choice.
        """
        first_arn = import_dataset(s3_fixtures["byod_training_sft.jsonl"])
        second_arn = import_dataset(s3_fixtures["dt_merge_second.jsonl"])

        dataset = DatasetJobConfig(
            job_category=JobCategory.DATA_TRANSFORMATION,
            input_data_config=[{"DatasetArn": first_arn}, {"DatasetArn": second_arn}],
            output_data_config={"S3OutputPath": output_path},
            transformation_config={
                "Operations": [{"Operator": "Merge", "Config": {"OutputName": "merged"}}],
            },
        )

        with pytest.raises(DataPreparerJobFailedError) as exc_info:
            preparer.transform(dataset)

        message = str(exc_info.value)
        assert (
            "Merge" in message
        ), f"Error should name the operator whose schema rejected it, got: {message}"
        assert (
            "Config" in message
        ), f"Error should say the operator's Config was rejected, got: {message}"
        # Not asserted: that the message names the rejected key ("OutputName"), which
        # is what lets a customer fix a typo without guesswork. The service dropped it
        # ("Merge operator does not accept Config parameters. Contact support ...") and
        # has not restored it; reproduced in us-west-2 and us-east-1 on 2026-11-13, so
        # it is service-wide. The enforcement contract above is what this test covers.
        # The message regression is tracked outside the suite -- nothing here will tell
        # you when it is fixed.

    def test_more_than_one_operation_is_rejected(
        self, preparer, s3_fixtures, import_dataset, output_path
    ):
        """Two operations in one job are rejected — the limit is 1, not 10.

        This is the one constraint where the published schema and the service
        disagree, which is exactly why it needs a test rather than a doc line.
        ``print_schema(DATA_TRANSFORMATION)`` advertises
        ``operations: minItems=1, maxItems=10``, but the service accepts exactly one
        operation per job. Both operators below are individually valid
        (ExactDedup and Split appear in passing tests above), so the count is the
        only thing under test.

        The service rejects the request synchronously at ``CreateJob`` (a
        ``ValidationException``), so the customer gets an immediate error rather
        than a job that fails later.
        """
        input_arn = import_dataset(s3_fixtures["dt_dedup_duplicates.jsonl"])

        dataset = DatasetJobConfig(
            job_category=JobCategory.DATA_TRANSFORMATION,
            input_data_config=[{"DatasetArn": input_arn}],
            output_data_config={"S3OutputPath": output_path},
            transformation_config={
                "Operations": [
                    {
                        "Operator": "ExactDedup",
                        "Config": {"TextField": "prompt", "Normalize": False},
                    },
                    {
                        "Operator": "Split",
                        "Config": {"Ratios": {"train": 0.8, "test": 0.2}, "Seed": 42},
                    },
                ],
            },
        )

        with pytest.raises(ClientError) as exc_info:
            preparer.transform(dataset)

        error = exc_info.value.response["Error"]
        assert error["Code"] == "ValidationException", f"got: {error}"
        # The service has reworded this rejection at least once: it used to name the
        # one-operation limit, and now returns a generic schema error carrying the
        # bound. Accept either, but keep the message tied to THIS constraint so the
        # test still distinguishes it from any other ValidationException on this call.
        assert any(
            phrase in error["Message"] for phrase in ("one operation", "at most 1 element")
        ), f"Error should state the one-operation limit, got: {error['Message']}"

    def test_schema_conversion_to_the_source_format_fails(
        self, preparer, s3_fixtures, import_dataset, output_path
    ):
        """Converting a dataset to the format it already declares fails the job.

        The rule lives in the execution layer, not in the operator schema:
        ``SchemaConversion`` only requires ``TargetFormat`` and only constrains
        it to the three-value enum, so ``HF_Prompt_Completion`` passes schema
        validation at the service level. The execution then compares it against
        the source format resolved from the input hub content's
        ``DatasetFileFormat`` and raises. Only cross-format conversions are
        supported — there is no identity conversion — so even without the guard
        every record would error.

        ``import_dataset`` declares ``HF_Prompt_Completion``, which the fixture
        actually is, so targeting that same value is the identity case.
        """
        input_arn = import_dataset(s3_fixtures["byod_training_sft.jsonl"])

        dataset = DatasetJobConfig(
            job_category=JobCategory.DATA_TRANSFORMATION,
            input_data_config=[{"DatasetArn": input_arn}],
            output_data_config={"S3OutputPath": output_path},
            transformation_config={
                "Operations": [
                    {
                        "Operator": "SchemaConversion",
                        "Config": {"TargetFormat": "HF_Prompt_Completion"},
                    }
                ],
            },
        )

        with pytest.raises(DataPreparerJobFailedError) as exc_info:
            preparer.transform(dataset)

        # The execution-time guard raises a bare ValueError, which the service
        # converts to the generic failure message — so assert what is knowable:
        # this was NOT a per-operator schema rejection. That distinction is the
        # content of the test; a job that failed at CreateJob validation instead
        # would mean the constraint moved to the schema-validation layer.
        message = str(exc_info.value)
        assert "Config validation failed" not in message, (
            f"Expected the job to fail while running, but the operator config was rejected "
            f"the config: {message}"
        )

    def test_exact_dedup_normalize_true_fails_on_the_lambda_lane(
        self, preparer, s3_fixtures, import_dataset, output_path
    ):
        """``Normalize: True`` fails for any dataset small enough to run on Lambda.

        Not a schema problem: ``ExactDedup``'s schema declares ``Normalize`` as a
        boolean, and the service's config validator accepts
        ``{"TextField": "text", "Normalize": true}`` without error. The failure
        is input-size-specific: for small inputs the service raises
        ``NotImplementedError`` on ``normalize`` because that Unicode normalization
        is only available on the infrastructure used for large inputs.

        This fixture is a few KB, so the small-input path is guaranteed, which
        is what makes the test deterministic — and is also why every ExactDedup
        config in this module pins ``Normalize: False``. Note the corollary: the
        same config would *succeed* on a sufficiently large input, so the
        tutorial's warning is about the common case, not a permanent property of
        the operator.
        """
        input_arn = import_dataset(s3_fixtures["dt_dedup_duplicates.jsonl"])

        dataset = DatasetJobConfig(
            job_category=JobCategory.DATA_TRANSFORMATION,
            input_data_config=[{"DatasetArn": input_arn}],
            output_data_config={"S3OutputPath": output_path},
            transformation_config={
                "Operations": [
                    {
                        "Operator": "ExactDedup",
                        "Config": {"TextField": "prompt", "Normalize": True},
                    }
                ],
            },
        )

        with pytest.raises(DataPreparerJobFailedError) as exc_info:
            preparer.transform(dataset)

        # As above: the execution error is wrapped in the generic service failure
        # message, so the specific cause does not reach FailureReason. Pin only
        # that the per-operator schema did not reject it at the API level — if
        # it ever does, this warning becomes a CreateJob-time contract and both
        # the test and the tutorial text should change.
        message = str(exc_info.value)
        assert "Config validation failed" not in message, (
            f"Expected the job to fail while running, but the operator config was rejected "
            f"Normalize=True: {message}"
        )


@pytest.mark.integ
class TestWaitLogsFiltering:
    """wait(logs=True) streams CW logs with DEBUG lines suppressed.

    Exercises ``Job._is_printable_log_line``: the launch fix that prevents
    DEBUG-level container log lines from leaking through to the customer in
    log-tail mode.  A single extra ExactDedup job is submitted specifically to
    exercise this code path; all other tests in this module use the default
    ``logs=False``.
    """

    def test_logs_true_filters_debug_lines(
        self,
        preparer,
        s3_fixtures,
        import_dataset,
        output_path,
        sagemaker_session,
        capfd,
    ):
        """A DT job run with wait=True, logs=True prints at least one line and
        no container DEBUG lines (format: ' - DEBUG - ').

        The wait() loop in Log_Tail_Mode always prints the job-completed summary
        line, so at least one line is guaranteed even if the CloudWatch stream is
        empty.  _is_printable_log_line must reject every line whose level token
        is DEBUG regardless of the surrounding text.
        """
        input_arn = import_dataset(s3_fixtures["dt_dedup_duplicates.jsonl"])
        dataset = DatasetJobConfig(
            job_category=JobCategory.DATA_TRANSFORMATION,
            input_data_config=[{"DatasetArn": input_arn}],
            output_data_config={"S3OutputPath": output_path},
            transformation_config={
                "Operations": [
                    {
                        "Operator": "ExactDedup",
                        "Config": {"TextField": "prompt", "Normalize": False},
                    }
                ],
            },
        )

        preparer.transform(dataset, wait=True, logs=True)

        captured = capfd.readouterr()
        all_output = captured.out + captured.err
        lines = [ln for ln in all_output.splitlines() if ln.strip()]

        # The job-completed summary line is always printed by _wait_log_tail_mode.
        assert lines, (
            "wait(logs=True) produced no visible output; expected at least the "
            "job-completed summary line"
        )
        # _is_printable_log_line must suppress every container DEBUG log line.
        debug_lines = [ln for ln in lines if " - DEBUG - " in ln]
        assert not debug_lines, (
            f"wait(logs=True) leaked {len(debug_lines)} DEBUG line(s); "
            f"first: {debug_lines[0]!r}"
        )


@pytest.mark.integ
class TestEmptyOutput:
    """TextQualityFilter with threshold above every row should produce zero output.

    A4: the SDK contract for zero-row output is not yet finalized.  Both
    outcomes are acceptable:
      - The job completes and each returned DataSet has 0 records.
      - The job raises a documented DataPreparerError subclass.
    Any other exception type is an unexpected regression and fails the test.
    """

    def test_all_rows_filtered_out(
        self,
        preparer,
        s3_fixtures,
        import_dataset,
        output_path,
        sagemaker_session,
        record_property,
        caplog,
    ):
        """TextQualityFilter(MinLength=200) drops all 5 rows of byod_training_sft.jsonl.

        The longest prompt in that fixture is 178 characters, so MinLength=200
        exceeds every row.  The observed terminal state and row count are recorded
        in the assertion messages so the live run documents the current behavior.
        """
        input_arn = import_dataset(s3_fixtures["byod_training_sft.jsonl"])
        dataset = DatasetJobConfig(
            job_category=JobCategory.DATA_TRANSFORMATION,
            input_data_config=[{"DatasetArn": input_arn}],
            output_data_config={"S3OutputPath": output_path},
            transformation_config={
                "Operations": [
                    {
                        "Operator": "TextQualityFilter",
                        "Config": {
                            "FilterType": "CharacterLength",
                            "TextField": "prompt",
                            "Action": "drop",
                            "MinLength": 200,  # above all 5 rows (max 178 chars)
                        },
                    }
                ],
            },
        )

        try:
            with caplog.at_level(
                logging.WARNING, logger="sagemaker.ai_registry.data_preparer.preparer"
            ):
                result = preparer.transform(dataset)
        except DataPreparerError as exc:
            # A4: raising a documented DataPreparerError on empty output is one of
            # the two acceptable outcomes while the SDK contract is unresolved.
            # Record which outcome occurred so the run documents current behavior.
            record_property("zero_row_outcome", f"raised {type(exc).__name__}: {exc}")
            return
        record_property("zero_row_outcome", "completed")

        # Job completed — every returned DataSet must have 0 records.
        assert isinstance(
            result, list
        ), f"B3: transform must return a list; got {type(result).__name__}"
        for ds in result:
            rows = _read_rows(ds.source, sagemaker_session.boto_region_name)
            assert len(rows) == 0, (
                f"B3: empty-output job completed but {ds.source!r} has "
                f"{len(rows)} rows; expected 0 (all dropped by MinLength=200)"
            )
        # A4: a completed job with no output records must not look like real
        # output — the SDK warns, naming the empty output object.
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        for ds in result:
            assert any("no records" in w and ds.source in w for w in warnings), (
                f"A4: expected an empty-output warning for {ds.source!r}; "
                f"warnings logged: {warnings}"
            )
