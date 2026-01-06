from sagemaker.core.common_utils import (
    format_tags,
    resolve_value_from_config,
    update_list_of_dicts_with_values_from_config,
    _create_resource,
    can_model_package_source_uri_autopopulate,
)
from sagemaker.core.config import (
    MODEL_PACKAGE_VALIDATION_ROLE_PATH,
    VALIDATION_ROLE,
    VALIDATION_PROFILES,
    MODEL_PACKAGE_INFERENCE_SPECIFICATION_CONTAINERS_PATH,
    MODEL_PACKAGE_VALIDATION_PROFILES_PATH,
)
from botocore.exceptions import ClientError
import logging

logger = LOGGER = logging.getLogger("sagemaker")


def get_model_package_args(
    content_types=None,
    response_types=None,
    inference_instances=None,
    transform_instances=None,
    model_package_name=None,
    model_package_group_name=None,
    model_data=None,
    image_uri=None,
    model_metrics=None,
    metadata_properties=None,
    marketplace_cert=False,
    approval_status=None,
    description=None,
    tags=None,
    container_def_list=None,
    drift_check_baselines=None,
    customer_metadata_properties=None,
    validation_specification=None,
    domain=None,
    sample_payload_url=None,
    task=None,
    skip_model_validation=None,
    source_uri=None,
    model_card=None,
    model_life_cycle=None,
):
    if container_def_list is not None:
        containers = container_def_list
    else:
        container = {
            "Image": image_uri,
        }
        if model_data is not None:
            container["ModelDataUrl"] = model_data

        containers = [container]

    model_package_args = {
        "containers": containers,
        "inference_instances": inference_instances,
        "transform_instances": transform_instances,
        "marketplace_cert": marketplace_cert,
    }

    if content_types is not None:
        model_package_args["content_types"] = content_types
    if response_types is not None:
        model_package_args["response_types"] = response_types
    if model_package_name is not None:
        model_package_args["model_package_name"] = model_package_name
    if model_package_group_name is not None:
        model_package_args["model_package_group_name"] = model_package_group_name
    if model_metrics is not None:
        model_package_args["model_metrics"] = model_metrics._to_request_dict()
    if drift_check_baselines is not None:
        model_package_args["drift_check_baselines"] = drift_check_baselines._to_request_dict()
    if metadata_properties is not None:
        model_package_args["metadata_properties"] = metadata_properties._to_request_dict()
    if approval_status is not None:
        model_package_args["approval_status"] = approval_status
    if description is not None:
        model_package_args["description"] = description
    if tags is not None:
        model_package_args["tags"] = format_tags(tags)
    if customer_metadata_properties is not None:
        model_package_args["customer_metadata_properties"] = customer_metadata_properties
    if validation_specification is not None:
        model_package_args["validation_specification"] = validation_specification
    if domain is not None:
        model_package_args["domain"] = domain
    if sample_payload_url is not None:
        model_package_args["sample_payload_url"] = sample_payload_url
    if task is not None:
        model_package_args["task"] = task
    if skip_model_validation is not None:
        model_package_args["skip_model_validation"] = skip_model_validation
    if source_uri is not None:
        model_package_args["source_uri"] = source_uri
    if model_life_cycle is not None:
        model_package_args["model_life_cycle"] = model_life_cycle._to_request_dict()
    if model_card is not None:
        original_req = model_card._create_request_args()
        if original_req.get("ModelCardName") is not None:
            del original_req["ModelCardName"]
        if original_req.get("Content") is not None:
            original_req["ModelCardContent"] = original_req["Content"]
            del original_req["Content"]
        model_package_args["model_card"] = original_req
    return model_package_args


def get_create_model_package_request(
    model_package_name=None,
    model_package_group_name=None,
    containers=None,
    content_types=None,
    response_types=None,
    inference_instances=None,
    transform_instances=None,
    model_metrics=None,
    metadata_properties=None,
    marketplace_cert=False,
    approval_status="PendingManualApproval",
    description=None,
    tags=None,
    drift_check_baselines=None,
    customer_metadata_properties=None,
    validation_specification=None,
    domain=None,
    sample_payload_url=None,
    task=None,
    skip_model_validation="None",
    source_uri=None,
    model_card=None,
    model_life_cycle=None,
):
    if all([model_package_name, model_package_group_name]):
        raise ValueError(
            "model_package_name and model_package_group_name cannot be present at the " "same time."
        )
    if all([model_package_name, source_uri]):
        raise ValueError(
            "Un-versioned SageMaker Model Package currently cannot be " "created with source_uri."
        )
    if (containers is not None) and all(
        [
            model_package_name,
            any(
                [
                    (("ModelDataSource" in c) and (c["ModelDataSource"] is not None))
                    for c in containers
                ]
            ),
        ]
    ):
        raise ValueError(
            "Un-versioned SageMaker Model Package currently cannot be "
            "created with ModelDataSource."
        )
    request_dict = {}
    if model_package_name is not None:
        request_dict["ModelPackageName"] = model_package_name
    if model_package_group_name is not None:
        request_dict["ModelPackageGroupName"] = model_package_group_name
    if description is not None:
        request_dict["ModelPackageDescription"] = description
    if tags is not None:
        request_dict["Tags"] = format_tags(tags)
    if model_metrics:
        request_dict["ModelMetrics"] = model_metrics
    if drift_check_baselines:
        request_dict["DriftCheckBaselines"] = drift_check_baselines
    if metadata_properties:
        request_dict["MetadataProperties"] = metadata_properties
    if customer_metadata_properties is not None:
        request_dict["CustomerMetadataProperties"] = customer_metadata_properties
    if validation_specification:
        request_dict["ValidationSpecification"] = validation_specification
    if domain is not None:
        request_dict["Domain"] = domain
    if sample_payload_url is not None:
        request_dict["SamplePayloadUrl"] = sample_payload_url
    if task is not None:
        request_dict["Task"] = task
    if source_uri is not None:
        request_dict["SourceUri"] = source_uri
    if containers is not None:
        inference_specification = {
            "Containers": containers,
        }
        if content_types is not None:
            inference_specification.update(
                {
                    "SupportedContentTypes": content_types,
                }
            )
        if response_types is not None:
            inference_specification.update(
                {
                    "SupportedResponseMIMETypes": response_types,
                }
            )
        if model_package_group_name is not None:
            if inference_instances is not None:
                inference_specification.update(
                    {
                        "SupportedRealtimeInferenceInstanceTypes": inference_instances,
                    }
                )
            if transform_instances is not None:
                inference_specification.update(
                    {
                        "SupportedTransformInstanceTypes": transform_instances,
                    }
                )
        else:
            if not all([inference_instances, transform_instances]):
                raise ValueError(
                    "inference_instances and transform_instances "
                    "must be provided if model_package_group_name is not present."
                )
            inference_specification.update(
                {
                    "SupportedRealtimeInferenceInstanceTypes": inference_instances,
                    "SupportedTransformInstanceTypes": transform_instances,
                }
            )
        request_dict["InferenceSpecification"] = inference_specification
    request_dict["CertifyForMarketplace"] = marketplace_cert
    request_dict["ModelApprovalStatus"] = approval_status
    request_dict["SkipModelValidation"] = skip_model_validation
    if model_card is not None:
        request_dict["ModelCard"] = model_card
    if model_life_cycle is not None:
        request_dict["ModelLifeCycle"] = model_life_cycle
    return request_dict


def create_model_package_from_containers(
    sagemaker_session,
    containers=None,
    content_types=None,
    response_types=None,
    inference_instances=None,
    transform_instances=None,
    model_package_name=None,
    model_package_group_name=None,
    model_metrics=None,
    metadata_properties=None,
    marketplace_cert=False,
    approval_status="PendingManualApproval",
    description=None,
    drift_check_baselines=None,
    customer_metadata_properties=None,
    validation_specification=None,
    domain=None,
    sample_payload_url=None,
    task=None,
    skip_model_validation="None",
    source_uri=None,
    model_card=None,
    model_life_cycle=None,
):
    """Get request dictionary for CreateModelPackage API.

    Args:
        containers (list): A list of inference containers that can be used for inference
            specifications of Model Package (default: None).
        content_types (list): The supported MIME types for the input data (default: None).
        response_types (list): The supported MIME types for the output data (default: None).
        inference_instances (list): A list of the instance types that are used to
            generate inferences in real-time (default: None).
        transform_instances (list): A list of the instance types on which a transformation
            job can be run or on which an endpoint can be deployed (default: None).
        model_package_name (str): Model Package name, exclusive to `model_package_group_name`,
            using `model_package_name` makes the Model Package un-versioned (default: None).
        model_package_group_name (str): Model Package Group name, exclusive to
            `model_package_name`, using `model_package_group_name` makes the Model Package
            versioned (default: None).
        model_metrics (ModelMetrics): ModelMetrics object (default: None).
        metadata_properties (MetadataProperties): MetadataProperties object (default: None)
        marketplace_cert (bool): A boolean value indicating if the Model Package is certified
            for AWS Marketplace (default: False).
        approval_status (str): Model Approval Status, values can be "Approved", "Rejected",
            or "PendingManualApproval" (default: "PendingManualApproval").
        description (str): Model Package description (default: None).
        drift_check_baselines (DriftCheckBaselines): DriftCheckBaselines object (default: None).
        customer_metadata_properties (dict[str, str]): A dictionary of key-value paired
            metadata properties (default: None).
        domain (str): Domain values can be "COMPUTER_VISION", "NATURAL_LANGUAGE_PROCESSING",
            "MACHINE_LEARNING" (default: None).
        sample_payload_url (str): The S3 path where the sample payload is stored
            (default: None).
        task (str): Task values which are supported by Inference Recommender are "FILL_MASK",
            "IMAGE_CLASSIFICATION", "OBJECT_DETECTION", "TEXT_GENERATION", "IMAGE_SEGMENTATION",
            "CLASSIFICATION", "REGRESSION", "OTHER" (default: None).
        skip_model_validation (str): Indicates if you want to skip model validation.
            Values can be "All" or "None" (default: None).
        source_uri (str): The URI of the source for the model package (default: None).
        model_card (ModeCard or ModelPackageModelCard): document contains qualitative and
            quantitative information about a model (default: None).
        model_life_cycle (ModelLifeCycle): ModelLifeCycle object (default: None).
    """
    if containers:
        # Containers are provided. Now we can merge missing entries from config.
        # If Containers are not provided, it is safe to ignore. This is because,
        # if this object is provided to the API, then Image is required for Containers.
        # That is not supported by the config now. So if we merge values from config,
        # then API will throw an exception. In the future, when SageMaker Config starts
        # supporting other parameters we can add that.
        update_list_of_dicts_with_values_from_config(
            containers,
            MODEL_PACKAGE_INFERENCE_SPECIFICATION_CONTAINERS_PATH,
            required_key_paths=["Image"],
            sagemaker_session=sagemaker_session,
        )

    if validation_specification:
        # ValidationSpecification is provided. Now we can merge missing entries from config.
        # If ValidationSpecification is not provided, it is safe to ignore. This is because,
        # if this object is provided to the API, then both ValidationProfiles and ValidationRole
        # are required and for ValidationProfile, ProfileName is a required parameter. That is
        # not supported by the config now. So if we merge values from config, then API will
        # throw an exception. In the future, when SageMaker Config starts supporting other
        # parameters we can add that.
        validation_role = resolve_value_from_config(
            validation_specification.get(VALIDATION_ROLE, None),
            MODEL_PACKAGE_VALIDATION_ROLE_PATH,
            sagemaker_session=sagemaker_session,
        )
        validation_specification[VALIDATION_ROLE] = validation_role
        validation_profiles = validation_specification.get(VALIDATION_PROFILES, [])
        update_list_of_dicts_with_values_from_config(
            validation_profiles,
            MODEL_PACKAGE_VALIDATION_PROFILES_PATH,
            required_key_paths=["ProfileName", "TransformJobDefinition"],
            sagemaker_session=sagemaker_session,
        )
    model_pkg_request = get_create_model_package_request(
        model_package_name,
        model_package_group_name,
        containers,
        content_types,
        response_types,
        inference_instances,
        transform_instances,
        model_metrics,
        metadata_properties,
        marketplace_cert,
        approval_status,
        description,
        drift_check_baselines=drift_check_baselines,
        customer_metadata_properties=customer_metadata_properties,
        validation_specification=validation_specification,
        domain=domain,
        sample_payload_url=sample_payload_url,
        task=task,
        skip_model_validation=skip_model_validation,
        source_uri=source_uri,
        model_card=model_card,
        model_life_cycle=model_life_cycle,
    )

    def submit(request):
        if model_package_group_name is not None and not model_package_group_name.startswith("arn:"):
            is_model_package_group_present = False
            try:
                model_package_groups_response = sagemaker_session.search(
                    resource="ModelPackageGroup",
                    search_expression={
                        "Filters": [
                            {
                                "Name": "ModelPackageGroupName",
                                "Value": request["ModelPackageGroupName"],
                                "Operator": "Equals",
                            }
                        ],
                    },
                )
                if len(model_package_groups_response.get("Results")) > 0:
                    is_model_package_group_present = True
            except Exception:  # pylint: disable=W0703
                model_package_groups = []
                model_package_groups_response = (
                    sagemaker_session.sagemaker_client.list_model_package_groups(
                        NameContains=request["ModelPackageGroupName"],
                    )
                )
                model_package_groups = (
                    model_package_groups
                    + model_package_groups_response["ModelPackageGroupSummaryList"]
                )
                next_token = model_package_groups_response.get("NextToken")

                while next_token is not None and next_token != "":
                    model_package_groups_response = (
                        sagemaker_session.sagemaker_client.list_model_package_groups(
                            NameContains=request["ModelPackageGroupName"], NextToken=next_token
                        )
                    )
                    model_package_groups = (
                        model_package_groups
                        + model_package_groups_response["ModelPackageGroupSummaryList"]
                    )
                    next_token = model_package_groups_response.get("NextToken")

                filtered_model_package_group = list(
                    filter(
                        lambda mpg: mpg.get("ModelPackageGroupName")
                        == request["ModelPackageGroupName"],
                        model_package_groups,
                    )
                )
                is_model_package_group_present = len(filtered_model_package_group) > 0
            if not is_model_package_group_present:
                _create_resource(
                    lambda: sagemaker_session.sagemaker_client.create_model_package_group(
                        ModelPackageGroupName=request["ModelPackageGroupName"]
                    )
                )
        if "SourceUri" in request and request["SourceUri"] is not None:
            # Remove inference spec from request if the
            # given source uri can lead to auto-population of it
            if can_model_package_source_uri_autopopulate(request["SourceUri"]):
                if "InferenceSpecification" in request:
                    del request["InferenceSpecification"]
                return sagemaker_session.sagemaker_client.create_model_package(**request)
            # If source uri can't autopopulate,
            # first create model package with just the inference spec
            # and then update model package with the source uri.
            # Done this way because passing source uri and inference spec together
            # in create/update model package is not allowed in the base sdk.
            request_source_uri = request["SourceUri"]
            del request["SourceUri"]
            model_package = sagemaker_session.sagemaker_client.create_model_package(**request)
            update_source_uri_args = {
                "ModelPackageArn": model_package.get("ModelPackageArn"),
                "SourceUri": request_source_uri,
            }
            return sagemaker_session.sagemaker_client.update_model_package(**update_source_uri_args)
        return sagemaker_session.sagemaker_client.create_model_package(**request)

    return sagemaker_session._intercept_create_request(
        model_pkg_request, submit, create_model_package_from_containers.__name__
    )


def create_model_package_from_algorithm(self, name, description, algorithm_arn, model_data):
    """Create a SageMaker Model Package from the results of training with an Algorithm Package.

    Args:
        name (str): ModelPackage name
        description (str): Model Package description
        algorithm_arn (str): arn or name of the algorithm used for training.
        model_data (str or dict[str, Any]): s3 URI or a dictionary representing a
        ``ModelDataSource`` to the model artifacts produced by training
    """
    sourceAlgorithm = {"AlgorithmName": algorithm_arn}
    if isinstance(model_data, dict):
        sourceAlgorithm["ModelDataSource"] = model_data
    else:
        sourceAlgorithm["ModelDataUrl"] = model_data

    request = {
        "ModelPackageName": name,
        "ModelPackageDescription": description,
        "SourceAlgorithmSpecification": {"SourceAlgorithms": [sourceAlgorithm]},
    }
    try:
        logger.info("Creating model package with name: %s", name)
        self.sagemaker_client.create_model_package(**request)
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        message = e.response["Error"]["Message"]

        if error_code == "ValidationException" and "ModelPackage already exists" in message:
            logger.warning("Using already existing model package: %s", name)
        else:
            raise
