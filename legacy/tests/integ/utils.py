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
import logging
import shutil
from functools import wraps

from botocore.exceptions import ClientError

from tests.conftest import NO_P3_REGIONS, NO_M4_REGIONS
from sagemaker.exceptions import CapacityError
from sagemaker.session import Session

P2_INSTANCES = ["ml.p2.xlarge", "ml.p2.8xlarge", "ml.p2.16xlarge"]
P3_INSTANCES = ["ml.p3.2xlarge"]


def gpu_list(region):
    if region in NO_P3_REGIONS:
        return P2_INSTANCES
    else:
        return [*P2_INSTANCES, *P3_INSTANCES]


def cpu_list(region):
    if region in NO_M4_REGIONS:
        return ["ml.m5.xlarge"]
    else:
        return ["ml.m4.xlarge", "ml.m5.xlarge"]


def retry_with_instance_list(instance_list):
    """Decorator for running an Integ test with an instance_list and
    break on first success

    Args:
        instance_list (list): List of Compute instances for integ test.
    Usage:
        @retry_with_instance_list(instance_list=["ml.g3.2", "ml.g2"])
        def sample_function():
            print("xxxx....")
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not (instance_list and isinstance(instance_list, list)):
                error_string = f"Parameter instance_list = {instance_list} \
                is expected to be a non-empty list of instance types."
                raise Exception(error_string)
            for i_type in instance_list:
                logging.info(f"Using the instance type: {i_type} for {func.__name__}")
                try:
                    kwargs.update({"instance_type": i_type})
                    func(*args, **kwargs)
                except CapacityError as e:
                    if i_type != instance_list[-1]:
                        logging.warning(
                            "Failure using instance type: {}. {}".format(i_type, str(e))
                        )
                        continue
                    else:
                        raise
                break

        return wrapper

    return decorator


def create_repository(ecr_client, repository_name):
    """Creates an ECS Repository (ECR).

    When a new transform is being registered,
    we'll need a repository to push the image (and composed model images) to
    """
    try:
        response = ecr_client.create_repository(repositoryName=repository_name)
        return response["repository"]["repositoryUri"]
    except ClientError as e:
        # Handle when the repository already exists
        if "RepositoryAlreadyExistsException" == e.response.get("Error", {}).get("Code"):
            response = ecr_client.describe_repositories(repositoryNames=[repository_name])
            return response["repositories"][0]["repositoryUri"]
        else:
            raise


# takes functions attached to a predictor, model, etc where the
# resource names are available via class variables
# delete in the order of creation to avoid floating resources
def cleanup_model_resources(sagemaker_session: Session, model_name: str, endpoint_name: str):
    try:
        sagemaker_session.delete_model(model_name=model_name)
        sagemaker_session.delete_endpoint_config(endpoint_config_name=endpoint_name)
        sagemaker_session.delete_endpoint(endpoint_name=endpoint_name)
    except Exception:
        return


# takes a path which need to be deleted
def cleanup_dir(path: str):
    try:
        shutil.rmtree(path)
    except Exception:
        return
