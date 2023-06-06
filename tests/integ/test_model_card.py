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
from six.moves.urllib.parse import urlparse
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
)
from tests.integ import (
    MODEL_CARD_DEFAULT_TIMEOUT_MINUTES,
)
from tests.integ.timeout import timeout, timeout_and_delete_model_by_name


ROLE = "SageMakerRole"


@pytest.fixture(scope="module", name="binary_classifier")
def binary_classifier_fixture(
    sagemaker_session: Session,
    cpu_instance_type: str,
):
    """Manage the model required for the model card integration test.

    Args:
        sagemaker_session (Session): A SageMaker Session
                object, used for SageMaker interactions.
        cpu_instance_type (_type_): Instance type used for training model
            and deploy endpoint.
    """
    model_name = unique_name_from_base("integ-test-binary-classifier-endpoint")
    with timeout_and_delete_model_by_name(
        model_name=model_name,
        sagemaker_session=sagemaker_session,
        minutes=MODEL_CARD_DEFAULT_TIMEOUT_MINUTES,
    ):
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
