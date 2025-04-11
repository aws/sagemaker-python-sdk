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
import io
import numpy as np
import pytest
import logging
from six.moves.urllib.parse import urlparse
from typing import Tuple
from sagemaker import s3
from sagemaker.estimator import Estimator
import sagemaker
from sagemaker.session import Session
import sagemaker.amazon.common as smac
from sagemaker.image_uris import retrieve
from sagemaker.utils import unique_name_from_base
from sagemaker.model_card import (
    ModelCard,
    ModelOverview,
    TrainingDetails,
    ModelCardStatusEnum,
    ModelPackage,
    BusinessDetails,
    IntendedUses,
    schema_constraints,
)
from sagemaker.model_card.model_card import ModelApprovalStatusEnum
from tests.integ import (
    MODEL_CARD_DEFAULT_TIMEOUT_MINUTES,
    DATA_DIR,
)
from tests.integ.timeout import timeout, timeout_and_delete_model_by_name
from tests.integ.retry import retries


ROLE = "SageMakerRole"


# intended uses arguments
PURPOSE_OF_MODEL = "mock model for testing"
INTENDED_USES = "this model card is used for development testing"
FACTORS_AFFECTING_MODEL_EFFICIENCY = "a bad factor"
RISK_RATING = schema_constraints.RiskRatingEnum.LOW
EXPLANATIONS_FOR_RISK_RATING = "ramdomly the first example"

# business details arguments
BUSINESS_PROBLEM = "mock model for business problem testing"
BUSINESS_STAKEHOLDERS = "business stakeholders testing"
LINE_OF_BUSINESS = "how many business models"


@pytest.fixture(scope="module", name="training_job")
def training_job_fixture(
    sagemaker_session: Session,
    cpu_instance_type: str,
):
    """Training job fixture used for the creation of models and model packages."""
    with timeout(minutes=MODEL_CARD_DEFAULT_TIMEOUT_MINUTES):
        # upload data
        raw_data = (
            (0.5, 0),
            (0.75, 0),
            (1.0, 0),
            (1.25, 0),
            (1.50, 0),
            (1.75, 0),
            (2.0, 0),
            (2.25, 1),
            (2.5, 0),
            (2.75, 1),
            (3.0, 0),
            (3.25, 1),
            (3.5, 0),
            (4.0, 1),
            (4.25, 1),
            (4.5, 1),
            (4.75, 1),
            (5.0, 1),
            (5.5, 1),
        )
        training_data = np.array(raw_data).astype("float32")
        labels = training_data[:, 1]

        bucket = sagemaker_session.default_bucket()
        prefix = "integ-test-data/model-card/binary-classifier"

        buf = io.BytesIO()
        smac.write_numpy_to_dense_tensor(buf, training_data, labels)
        buf.seek(0)

        sagemaker_session.boto_session.resource(
            "s3", region_name=sagemaker_session.boto_region_name
        ).Bucket(bucket).Object(os.path.join(prefix, "train")).upload_fileobj(buf)

        # train model
        s3_train_data = f"s3://{bucket}/{prefix}/train"
        output_location = f"s3://{bucket}/{prefix}/output"
        container = retrieve("linear-learner", sagemaker_session.boto_session.region_name)
        estimator = sagemaker.estimator.Estimator(
            container,
            role=ROLE,
            instance_count=1,
            instance_type=cpu_instance_type,
            output_path=output_location,
            sagemaker_session=sagemaker_session,
        )
        estimator.set_hyperparameters(
            feature_dim=2, mini_batch_size=10, predictor_type="binary_classifier"
        )
        estimator.fit({"train": s3_train_data})

        return estimator


@pytest.fixture(scope="module", name="binary_classifier")
def binary_classifier_fixture(
    sagemaker_session: Session,
    training_job: Estimator,
):
    """Manage the model required for the model card integration test."""
    model_name = unique_name_from_base("integ-test-binary-classifier-endpoint")
    estimator = training_job
    with timeout_and_delete_model_by_name(
        model_name=model_name,
        sagemaker_session=sagemaker_session,
        minutes=MODEL_CARD_DEFAULT_TIMEOUT_MINUTES,
    ):
        model = estimator.create_model(name=model_name)
        container_def = model.prepare_container_def()
        sagemaker_session.create_model(model_name, ROLE, container_def)

        # Yield to run the integration tests
        yield model_name, estimator.latest_training_job.name

        # Cleanup resources
        sagemaker_session.delete_model(model_name)

    # Validate resource cleanup
    with pytest.raises(Exception) as exception:
        sagemaker_session.sagemaker_client.describe_model(ModelName=model_name)
        assert "Could not find model" in str(exception.value)


@pytest.fixture(scope="module", name="evaluation_data")
def evaluation_data_fixture(sagemaker_session: Session):
    """Manage the evaluation metrics report for model packages"""
    bucket = sagemaker_session.default_bucket()
    prefix = "integ-test-data/model-card/evaluation"
    model_quality_data_path = os.path.join(
        DATA_DIR, "model_card", "evaluation_metrics", "model_monitor_model_quality_regression.json"
    )
    bias_data_path = os.path.join(DATA_DIR, "model_card", "evaluation_metrics", "clarify_bias.json")

    # s3 path to upload the evaluation reports
    base_s3_path = f"s3://{bucket}/{prefix}"

    # upload local evaluation report file to s3 for testing evaluation job auto-discovery
    model_quality_s3_uri = _upload_data(model_quality_data_path, base_s3_path, sagemaker_session)
    bias_s3_uri = _upload_data(bias_data_path, base_s3_path, sagemaker_session)

    yield model_quality_s3_uri, bias_s3_uri

    # clean up resources
    _delete_data(model_quality_s3_uri, sagemaker_session)
    _delete_data(bias_s3_uri, sagemaker_session)


@pytest.fixture(scope="module", name="model_package")
def model_package_fixture(
    sagemaker_session: Session,
    training_job: Estimator,
    evaluation_data: Tuple[str, str],
):
    """Manage model package groups and model packages for the model card integration test"""
    model_package_group_name = unique_name_from_base("test-model-package-group")
    training_job_name = training_job.latest_training_job.name
    model_quality_s3_uri, bias_s3_uri = evaluation_data
    with timeout(minutes=MODEL_CARD_DEFAULT_TIMEOUT_MINUTES):
        sagemaker_session.sagemaker_client.create_model_package_group(
            ModelPackageGroupName=model_package_group_name
        )

        training_job = sagemaker_session.sagemaker_client.describe_training_job(
            TrainingJobName=training_job_name
        )

        model_data_url = training_job["ModelArtifacts"]["S3ModelArtifacts"]
        image = training_job["AlgorithmSpecification"]["TrainingImage"]

        create_model_package_input_dict = {
            "ModelPackageGroupName": model_package_group_name,
            "ModelPackageDescription": "Test model package registered for integ test",
            "ModelApprovalStatus": ModelApprovalStatusEnum.PENDING_MANUAL_APPROVAL,
            "InferenceSpecification": {
                "Containers": [{"Image": image, "ModelDataUrl": model_data_url}],
                "SupportedContentTypes": ["text/csv"],
                "SupportedResponseMIMETypes": ["text/csv"],
            },
        }

        model_pkg1 = sagemaker_session.sagemaker_client.create_model_package(
            **create_model_package_input_dict
        )

        # creating another model package with ModelMetrics(evaluation data)
        model_metrics_data = {
            "ModelMetrics": {
                "ModelQuality": {
                    "Statistics": {
                        "ContentType": "application/json",
                        "S3Uri": model_quality_s3_uri,
                    }
                },
                "Bias": {"Report": {"ContentType": "application/json", "S3Uri": bias_s3_uri}},
                "Explainability": {},
            }
        }

        create_model_package_input_dict.update(model_metrics_data)
        model_pkg2 = sagemaker_session.sagemaker_client.create_model_package(
            **create_model_package_input_dict
        )

        # Yield to run the integration tests
        yield model_pkg1["ModelPackageArn"], model_pkg2["ModelPackageArn"]

        # clean up resources
        sagemaker_session.sagemaker_client.delete_model_package(
            ModelPackageName=model_pkg1["ModelPackageArn"]
        )
        sagemaker_session.sagemaker_client.delete_model_package(
            ModelPackageName=model_pkg2["ModelPackageArn"]
        )
        sagemaker_session.sagemaker_client.delete_model_package_group(
            ModelPackageGroupName=model_package_group_name
        )

    # Validate resource cleanup
    with pytest.raises(Exception) as exception:
        sagemaker_session.sagemaker_client.describe_model_package(
            ModelPackageName=model_pkg1["ModelPackageArn"]
        )
        assert "does not exist" in str(exception.value)

        sagemaker_session.sagemaker_client.describe_model_package(
            ModelPackageName=model_pkg2["ModelPackageArn"]
        )
        assert "does not exist" in str(exception.value)

        sagemaker_session.sagemaker_client.describe_model_package_group(
            ModelPackageGroupName=model_package_group_name
        )
        assert "does not exist" in str(exception.value)


@pytest.fixture(scope="module", name="business_details")
def business_details_fixture():
    """Example of business details instance."""
    test_example = BusinessDetails(
        business_problem=BUSINESS_PROBLEM,
        business_stakeholders=BUSINESS_STAKEHOLDERS,
        line_of_business=LINE_OF_BUSINESS,
    )
    return test_example


@pytest.fixture(scope="module", name="intended_uses")
def intended_uses_fixture():
    """Example of intended uses instance."""
    test_example = IntendedUses(
        purpose_of_model=PURPOSE_OF_MODEL,
        intended_uses=INTENDED_USES,
        factors_affecting_model_efficiency=FACTORS_AFFECTING_MODEL_EFFICIENCY,
        risk_rating=RISK_RATING,
        explanations_for_risk_rating=EXPLANATIONS_FOR_RISK_RATING,
    )
    return test_example


@pytest.mark.flaky(reruns=5, reruns_delay=2)
def test_model_card_create_read_update_and_delete(
    sagemaker_session,
    binary_classifier,
):
    model_name, training_job_name = binary_classifier

    with timeout(minutes=MODEL_CARD_DEFAULT_TIMEOUT_MINUTES):
        model_card_name = unique_name_from_base("model-card")

        model_overview = ModelOverview.from_model_name(
            model_name=model_name,
            sagemaker_session=sagemaker_session,
        )
        assert model_overview.model_id

        training_details1 = TrainingDetails.from_training_job_name(
            training_job_name=training_job_name,
            sagemaker_session=sagemaker_session,
        )
        assert training_details1.training_job_details.training_arn
        training_details2 = TrainingDetails.from_model_overview(
            model_overview=model_overview,
            sagemaker_session=sagemaker_session,
        )
        assert (
            training_details1.training_job_details.training_arn
            == training_details2.training_job_details.training_arn
        )
        assert (
            training_details1.training_job_details.training_environment.container_image[0]
            == training_details2.training_job_details.training_environment.container_image[0]
        )
        assert len(training_details1.training_job_details.training_metrics) == len(
            training_details2.training_job_details.training_metrics
        )

        card = ModelCard(
            name=model_card_name,
            status=ModelCardStatusEnum.DRAFT,
            model_overview=model_overview,
            training_details=training_details1,
            sagemaker_session=sagemaker_session,
        )
        card.create()
        assert card.arn

        new_model_description = "the model card is updated."
        card.model_overview.model_description = new_model_description
        card.update()
        assert len(card.get_version_history()) == 2

        card_copy = ModelCard.load(
            name=model_card_name,
            sagemaker_session=sagemaker_session,
        )
        assert card_copy.arn == card.arn
        assert card_copy.model_overview.model_description == new_model_description

        # export job
        bucket = sagemaker_session.default_bucket()
        prefix = "integ-test-data/model-card"
        s3_output_path = f"s3://{bucket}/{prefix}/export"
        pdf_s3_url = card.export_pdf(
            export_job_name=f"export-{model_card_name}", s3_output_path=s3_output_path
        )
        parsed_url = urlparse(pdf_s3_url)
        pdf_bucket = parsed_url.netloc
        pdf_key = parsed_url.path.lstrip("/")
        region = sagemaker_session.boto_region_name
        s3 = sagemaker_session.boto_session.client("s3", region_name=region)
        assert s3.list_objects_v2(Bucket=pdf_bucket, Prefix=pdf_key)["KeyCount"] == 1

        # list export jobs
        assert len(card.list_export_jobs()["ModelCardExportJobSummaries"]) == 1

        # clean resources
        s3.delete_object(Bucket=pdf_bucket, Key=pdf_key)
        card.delete()

    # Validate resource cleanup
    with pytest.raises(Exception) as exception:
        sagemaker_session.sagemaker_client.describe_model_card(ModelCardName=model_card_name)
        assert "does not exist" in str(exception.value)

        s3.get_object(Bucket=pdf_bucket, Key=pdf_key)
        assert "The specified key does not exist" in str(exception.value)


def test_model_card_create_read_and_delete_with_model_package(
    sagemaker_session,
    business_details,
    intended_uses,
    model_package,
):
    model_package_arn1, model_package_arn2 = model_package
    model_card_name1 = unique_name_from_base("model-card1")
    model_card_name2 = unique_name_from_base("model-card2")

    with timeout(minutes=MODEL_CARD_DEFAULT_TIMEOUT_MINUTES):
        model_package_details1 = ModelPackage.from_model_package_arn(
            model_package_arn=model_package_arn1, sagemaker_session=sagemaker_session
        )

        assert model_package_details1.model_package_arn == model_package_arn1
        assert (
            model_package_details1.model_approval_status
            == ModelApprovalStatusEnum.PENDING_MANUAL_APPROVAL
        )

        # create first model card with business details and intended uses
        card1 = ModelCard(
            name=model_card_name1,
            status=ModelCardStatusEnum.DRAFT,
            business_details=business_details,
            intended_uses=intended_uses,
            model_package_details=model_package_details1,
            sagemaker_session=sagemaker_session,
        )

        # validate that there is an auto discovered training details
        assert card1.training_details.training_job_details is not None

        card1.create()
        assert card1.arn

        # create second model card with model package that has evaluation details and carried over information
        model_package_details2 = ModelPackage.from_model_package_arn(
            model_package_arn=model_package_arn2, sagemaker_session=sagemaker_session
        )

        try:
            # Wait for the new model card (card1) to be indexed in the Search
            for _ in retries(
                max_retry_count=5,
                exception_message_prefix="Waiting for Search to index model card(card1)",
                seconds_to_sleep=3,
            ):
                card2 = ModelCard(
                    name=model_card_name2,
                    status=ModelCardStatusEnum.DRAFT,
                    model_package_details=model_package_details2,
                    sagemaker_session=sagemaker_session,
                )

                if card2.business_details is not None:
                    break
        except Exception as error:
            logging.error(error)
            # Delete card1 if it was not indexed after 5 retries as a clean up process
            card1.delete()

        # validate that there are auto discovered training details and evaluation details
        assert card2.training_details.training_job_details is not None
        assert card2.evaluation_details is not None

        # validate the carry over information like business details and intended_uses
        assert card2.business_details.business_problem == business_details.business_problem
        assert card2.intended_uses.purpose_of_model == intended_uses.purpose_of_model

        card2.create()
        assert card2.arn

        # cleanup resources
        card1.delete()
        card2.delete()

    # Validate resource cleanup
    with pytest.raises(Exception) as exception:
        sagemaker_session.sagemaker_client.describe_model_card(ModelCardName=model_card_name1)
        assert "does not exist" in str(exception.value)

        sagemaker_session.sagemaker_client.describe_model_card(ModelCardName=model_card_name2)
        assert "does not exist" in str(exception.value)


def _upload_data(data_local_path, s3_data_path, sagemaker_session):
    """Upload data to S3
    Args:
        data_local_path (str): File path to the local evaluation report file.
        s3_data_path (str): S3 prefix to store the evaluation report file.
        sagemaker_session (:class:`~sagemaker.session.Session`):
            Session object which manages interactions with Amazon SageMaker and
            any other AWS services needed. If not specified, the processor creates
            one using the default AWS configuration chain.
    Returns:
        The S3 uri of the uploaded data.
    """
    return s3.S3Uploader.upload(
        local_path=data_local_path,
        desired_s3_uri=s3_data_path,
        sagemaker_session=sagemaker_session,
    )


def _delete_data(s3_uri, sagemaker_session):
    """Delete the uploaded data from S3

    Args:
        s3_data_path (str): S3 path of the file.
        sagemaker_session (:class:`~sagemaker.session.Session`):
            Session object which manages interactions with Amazon SageMaker and
            any other AWS services needed. If not specified, the processor creates
            one using the default AWS configuration chain.
    """
    parsed_url = urlparse(s3_uri)
    pdf_bucket = parsed_url.netloc
    pdf_key = parsed_url.path.lstrip("/")
    region = sagemaker_session.boto_region_name
    s3 = sagemaker_session.boto_session.client("s3", region_name=region)

    # delete file from S3
    s3.delete_object(Bucket=pdf_bucket, Key=pdf_key)
