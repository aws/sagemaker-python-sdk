import json
from sagemaker.core.tools.method import Method
from sagemaker.core.tools.resources_codegen import ResourcesCodeGen
from sagemaker.core.tools.constants import SERVICE_JSON_FILE_PATH


class TestGenerateResource:
    @classmethod
    def setup_class(cls):
        # TODO: leverage pytest fixtures
        with open(SERVICE_JSON_FILE_PATH, "r") as file:
            service_json = json.load(file)

        # Initialize parameters here
        cls.resource_generator = ResourcesCodeGen(service_json)

    # create a unit test for generate_create_method()
    def test_generate_create_method(self):
        expected_output = '''
@classmethod
@Base.add_validate_call
def create(
    cls,
    compilation_job_name: str,
    role_arn: str,
    output_config: OutputConfig,
    stopping_condition: StoppingCondition,
    model_package_version_arn: Optional[str] = Unassigned(),
    input_config: Optional[InputConfig] = Unassigned(),
    vpc_config: Optional[NeoVpcConfig] = Unassigned(),
    tags: Optional[List[Tag]] = Unassigned(),
    session: Optional[Session] = None,
    region: Optional[str] = None,
) -> Optional["CompilationJob"]:
    """
    Create a CompilationJob resource
    
    Parameters:
        compilation_job_name: A name for the model compilation job. The name must be unique within the Amazon Web Services Region and within your Amazon Web Services account. 
        role_arn: The Amazon Resource Name (ARN) of an IAM role that enables Amazon SageMaker AI to perform tasks on your behalf.  During model compilation, Amazon SageMaker AI needs your permission to:   Read input data from an S3 bucket   Write model artifacts to an S3 bucket   Write logs to Amazon CloudWatch Logs   Publish metrics to Amazon CloudWatch   You grant permissions for all of these tasks to an IAM role. To pass this role to Amazon SageMaker AI, the caller of this API must have the iam:PassRole permission. For more information, see Amazon SageMaker AI Roles. 
        output_config: Provides information about the output location for the compiled model and the target device the model runs on.
        stopping_condition: Specifies a limit to how long a model compilation job can run. When the job reaches the time limit, Amazon SageMaker AI ends the compilation job. Use this API to cap model training costs.
        model_package_version_arn: The Amazon Resource Name (ARN) of a versioned model package. Provide either a ModelPackageVersionArn or an InputConfig object in the request syntax. The presence of both objects in the CreateCompilationJob request will return an exception.
        input_config: Provides information about the location of input model artifacts, the name and shape of the expected data inputs, and the framework in which the model was trained.
        vpc_config: A VpcConfig object that specifies the VPC that you want your compilation job to connect to. Control access to your models by configuring the VPC. For more information, see Protect Compilation Jobs by Using an Amazon Virtual Private Cloud.
        tags: An array of key-value pairs. You can use tags to categorize your Amazon Web Services resources in different ways, for example, by purpose, owner, or environment. For more information, see Tagging Amazon Web Services Resources.
        session: Boto3 session.
        region: Region name.
    
    Returns:
        The CompilationJob resource.
    
    Raises:
        botocore.exceptions.ClientError: This exception is raised for AWS service related errors. 
            The error message and error code can be parsed from the exception as follows:
            ```
            try:
                # AWS service call here
            except botocore.exceptions.ClientError as e:
                error_message = e.response['Error']['Message']
                error_code = e.response['Error']['Code']
            ```
        ResourceInUse: Resource being accessed is in use.
        ResourceLimitExceeded: You have exceeded an SageMaker resource limit. For example, you might have too many training jobs created.
        ConfigSchemaValidationError: Raised when a configuration file does not adhere to the schema
        LocalConfigNotFoundError: Raised when a configuration file is not found in local file system
        S3ConfigNotFoundError: Raised when a configuration file is not found in S3
    """

    logger.info("Creating compilation_job resource.")
    client =Base.get_sagemaker_client(session=session, region_name=region, service_name='sagemaker')

    operation_input_args = {
        'CompilationJobName': compilation_job_name,
        'RoleArn': role_arn,
        'ModelPackageVersionArn': model_package_version_arn,
        'InputConfig': input_config,
        'OutputConfig': output_config,
        'VpcConfig': vpc_config,
        'StoppingCondition': stopping_condition,
        'Tags': tags,
    }
    
    operation_input_args = Base.populate_chained_attributes(resource_name='CompilationJob', operation_input_args=operation_input_args)
        
    logger.debug(f"Input request: {operation_input_args}")
    # serialize the input request
    operation_input_args = serialize(operation_input_args)
    logger.debug(f"Serialized input request: {operation_input_args}")

    # create the resource
    response = client.create_compilation_job(**operation_input_args)
    logger.debug(f"Response: {response}")

    return cls.get(compilation_job_name=compilation_job_name, session=session, region=region)
'''
        assert (
            self.resource_generator.generate_create_method(
                "CompilationJob", needs_defaults_decorator=False
            )
            == expected_output
        )

    def test_generate_import_method(self):
        expected_output = '''
@classmethod
@Base.add_validate_call
def load(
    cls,
    hub_content_name: str,
    hub_content_type: str,
    document_schema_version: str,
    hub_name: str,
    hub_content_document: str,
    hub_content_version: Optional[str] = Unassigned(),
    hub_content_display_name: Optional[str] = Unassigned(),
    hub_content_description: Optional[str] = Unassigned(),
    hub_content_markdown: Optional[str] = Unassigned(),
    support_status: Optional[str] = Unassigned(),
    hub_content_search_keywords: Optional[List[str]] = Unassigned(),
    tags: Optional[List[Tag]] = Unassigned(),
    session: Optional[Session] = None,
    region: Optional[str] = None,
) -> Optional["HubContent"]:
    """
    Import a HubContent resource
    
    Parameters:
        hub_content_name: The name of the hub content to import.
        hub_content_type: The type of hub content to import.
        document_schema_version: The version of the hub content schema to import.
        hub_name: The name of the hub to import content into.
        hub_content_document: The hub content document that describes information about the hub content such as type, associated containers, scripts, and more.
        hub_content_version: The version of the hub content to import.
        hub_content_display_name: The display name of the hub content to import.
        hub_content_description: A description of the hub content to import.
        hub_content_markdown: A string that provides a description of the hub content. This string can include links, tables, and standard markdown formating.
        support_status: The status of the hub content resource.
        hub_content_search_keywords: The searchable keywords of the hub content.
        tags: Any tags associated with the hub content.
        session: Boto3 session.
        region: Region name.
    
    Returns:
        The HubContent resource.
    
    Raises:
        botocore.exceptions.ClientError: This exception is raised for AWS service related errors. 
            The error message and error code can be parsed from the exception as follows:
            ```
            try:
                # AWS service call here
            except botocore.exceptions.ClientError as e:
                error_message = e.response['Error']['Message']
                error_code = e.response['Error']['Code']
            ```
        ResourceInUse: Resource being accessed is in use.
        ResourceLimitExceeded: You have exceeded an SageMaker resource limit. For example, you might have too many training jobs created.
        ResourceNotFound: Resource being access is not found.
    """

    logger.info(f"Importing hub_content resource.")
    client = SageMakerClient(session=session, region_name=region, service_name='sagemaker').client

    operation_input_args = {
        'HubContentName': hub_content_name,
        'HubContentVersion': hub_content_version,
        'HubContentType': hub_content_type,
        'DocumentSchemaVersion': document_schema_version,
        'HubName': hub_name,
        'HubContentDisplayName': hub_content_display_name,
        'HubContentDescription': hub_content_description,
        'HubContentMarkdown': hub_content_markdown,
        'HubContentDocument': hub_content_document,
        'SupportStatus': support_status,
        'HubContentSearchKeywords': hub_content_search_keywords,
        'Tags': tags,
    }

    logger.debug(f"Input request: {operation_input_args}")
    # serialize the input request
    operation_input_args = serialize(operation_input_args)
    logger.debug(f"Serialized input request: {operation_input_args}")

    # import the resource
    response = client.import_hub_content(**operation_input_args)
    logger.debug(f"Response: {response}")

    return cls.get(hub_name=hub_name, hub_content_type=hub_content_type, hub_content_name=hub_content_name, session=session, region=region)
'''
        assert self.resource_generator.generate_import_method("HubContent") == expected_output

    def test_generate_update_method_with_decorator(self):
        expected_output = '''
@populate_inputs_decorator
@Base.add_validate_call
def update(
    self,
    retain_all_variant_properties: Optional[bool] = Unassigned(),
    exclude_retained_variant_properties: Optional[List[VariantProperty]] = Unassigned(),
    deployment_config: Optional[DeploymentConfig] = Unassigned(),
    retain_deployment_config: Optional[bool] = Unassigned(),
) -> Optional["Endpoint"]:
    """
    Update a Endpoint resource
    
    Parameters:
        retain_all_variant_properties: When updating endpoint resources, enables or disables the retention of variant properties, such as the instance count or the variant weight. To retain the variant properties of an endpoint when updating it, set RetainAllVariantProperties to true. To use the variant properties specified in a new EndpointConfig call when updating an endpoint, set RetainAllVariantProperties to false. The default is false.
        exclude_retained_variant_properties: When you are updating endpoint resources with RetainAllVariantProperties, whose value is set to true, ExcludeRetainedVariantProperties specifies the list of type VariantProperty to override with the values provided by EndpointConfig. If you don't specify a value for ExcludeRetainedVariantProperties, no variant properties are overridden. 
        deployment_config: The deployment configuration for an endpoint, which contains the desired deployment strategy and rollback configurations.
        retain_deployment_config: Specifies whether to reuse the last deployment configuration. The default value is false (the configuration is not reused).
    
    Returns:
        The Endpoint resource.
    
    Raises:
        botocore.exceptions.ClientError: This exception is raised for AWS service related errors. 
            The error message and error code can be parsed from the exception as follows:
            ```
            try:
                # AWS service call here
            except botocore.exceptions.ClientError as e:
                error_message = e.response['Error']['Message']
                error_code = e.response['Error']['Code']
            ```
        ResourceLimitExceeded: You have exceeded an SageMaker resource limit. For example, you might have too many training jobs created.
    """

    logger.info("Updating endpoint resource.")
    client = Base.get_sagemaker_client()

    operation_input_args = {
        'EndpointName': self.endpoint_name,
        'EndpointConfigName': self.endpoint_config_name,
        'RetainAllVariantProperties': retain_all_variant_properties,
        'ExcludeRetainedVariantProperties': exclude_retained_variant_properties,
        'DeploymentConfig': deployment_config,
        'RetainDeploymentConfig': retain_deployment_config,
    }
    logger.debug(f"Input request: {operation_input_args}")
    # serialize the input request
    operation_input_args = serialize(operation_input_args)
    logger.debug(f"Serialized input request: {operation_input_args}")

    # create the resource
    response = client.update_endpoint(**operation_input_args)
    logger.debug(f"Response: {response}")
    self.refresh()

    return self
'''
        class_attributes = self.resource_generator._get_class_attributes("Endpoint", ["get"])
        resource_attributes = list(class_attributes[0].keys())
        assert (
            self.resource_generator.generate_update_method(
                "Endpoint", resource_attributes=resource_attributes, needs_defaults_decorator=True
            )
            == expected_output
        )

    def test_generate_update_method(self):
        expected_output = '''
@Base.add_validate_call
def update(
    self,
    retain_all_variant_properties: Optional[bool] = Unassigned(),
    exclude_retained_variant_properties: Optional[List[VariantProperty]] = Unassigned(),
    deployment_config: Optional[DeploymentConfig] = Unassigned(),
    retain_deployment_config: Optional[bool] = Unassigned(),
) -> Optional["Endpoint"]:
    """
    Update a Endpoint resource
    
    Parameters:
        retain_all_variant_properties: When updating endpoint resources, enables or disables the retention of variant properties, such as the instance count or the variant weight. To retain the variant properties of an endpoint when updating it, set RetainAllVariantProperties to true. To use the variant properties specified in a new EndpointConfig call when updating an endpoint, set RetainAllVariantProperties to false. The default is false.
        exclude_retained_variant_properties: When you are updating endpoint resources with RetainAllVariantProperties, whose value is set to true, ExcludeRetainedVariantProperties specifies the list of type VariantProperty to override with the values provided by EndpointConfig. If you don't specify a value for ExcludeRetainedVariantProperties, no variant properties are overridden. 
        deployment_config: The deployment configuration for an endpoint, which contains the desired deployment strategy and rollback configurations.
        retain_deployment_config: Specifies whether to reuse the last deployment configuration. The default value is false (the configuration is not reused).
    
    Returns:
        The Endpoint resource.
    
    Raises:
        botocore.exceptions.ClientError: This exception is raised for AWS service related errors. 
            The error message and error code can be parsed from the exception as follows:
            ```
            try:
                # AWS service call here
            except botocore.exceptions.ClientError as e:
                error_message = e.response['Error']['Message']
                error_code = e.response['Error']['Code']
            ```
        ResourceLimitExceeded: You have exceeded an SageMaker resource limit. For example, you might have too many training jobs created.
    """

    logger.info("Updating endpoint resource.")
    client = Base.get_sagemaker_client()

    operation_input_args = {
        'EndpointName': self.endpoint_name,
        'EndpointConfigName': self.endpoint_config_name,
        'RetainAllVariantProperties': retain_all_variant_properties,
        'ExcludeRetainedVariantProperties': exclude_retained_variant_properties,
        'DeploymentConfig': deployment_config,
        'RetainDeploymentConfig': retain_deployment_config,
    }
    logger.debug(f"Input request: {operation_input_args}")
    # serialize the input request
    operation_input_args = serialize(operation_input_args)
    logger.debug(f"Serialized input request: {operation_input_args}")

    # create the resource
    response = client.update_endpoint(**operation_input_args)
    logger.debug(f"Response: {response}")
    self.refresh()

    return self
'''
        class_attributes = self.resource_generator._get_class_attributes("Endpoint", ["get"])
        resource_attributes = list(class_attributes[0].keys())
        assert (
            self.resource_generator.generate_update_method(
                "Endpoint", resource_attributes=resource_attributes, needs_defaults_decorator=False
            )
            == expected_output
        )

    def test_generate_get_method(self):
        expected_output = '''
@classmethod
@Base.add_validate_call
def get(
    cls,
    domain_id: str,
    app_type: str,
    app_name: str,
    user_profile_name: Optional[str] = Unassigned(),
    space_name: Optional[str] = Unassigned(),
    session: Optional[Session] = None,
    region: Optional[str] = None,
) -> Optional["App"]:
    """
    Get a App resource
    
    Parameters:
        domain_id: The domain ID.
        app_type: The type of app.
        app_name: The name of the app.
        user_profile_name: The user profile name. If this value is not set, then SpaceName must be set.
        space_name: The name of the space.
        session: Boto3 session.
        region: Region name.
    
    Returns:
        The App resource.
    
    Raises:
        botocore.exceptions.ClientError: This exception is raised for AWS service related errors. 
            The error message and error code can be parsed from the exception as follows:
            ```
            try:
                # AWS service call here
            except botocore.exceptions.ClientError as e:
                error_message = e.response['Error']['Message']
                error_code = e.response['Error']['Code']
            ```
        ResourceNotFound: Resource being access is not found.
    """

    operation_input_args = {
        'DomainId': domain_id,
        'UserProfileName': user_profile_name,
        'SpaceName': space_name,
        'AppType': app_type,
        'AppName': app_name,
    }
    # serialize the input request
    operation_input_args = serialize(operation_input_args)
    logger.debug(f"Serialized input request: {operation_input_args}")

    client = Base.get_sagemaker_client(session=session, region_name=region, service_name='sagemaker')
    response = client.describe_app(**operation_input_args)

    logger.debug(response)

    # deserialize the response
    transformed_response = transform(response, 'DescribeAppResponse')
    app = cls(**transformed_response)
    return app
'''
        assert self.resource_generator.generate_get_method("App") == expected_output

    def test_generate_refresh_method(self):
        expected_output = '''
@Base.add_validate_call
def refresh(
    self,
    
    ) -> Optional["App"]:
    """
    Refresh a App resource
    
    Returns:
        The App resource.
    
    Raises:
        botocore.exceptions.ClientError: This exception is raised for AWS service related errors. 
            The error message and error code can be parsed from the exception as follows:
            ```
            try:
                # AWS service call here
            except botocore.exceptions.ClientError as e:
                error_message = e.response['Error']['Message']
                error_code = e.response['Error']['Code']
            ```
        ResourceNotFound: Resource being access is not found.
    """

    operation_input_args = {
        'DomainId': self.domain_id,
        'UserProfileName': self.user_profile_name,
        'SpaceName': self.space_name,
        'AppType': self.app_type,
        'AppName': self.app_name,
    }
    # serialize the input request
    operation_input_args = serialize(operation_input_args)
    logger.debug(f"Serialized input request: {operation_input_args}")

    client = Base.get_sagemaker_client()
    response = client.describe_app(**operation_input_args)

    # deserialize response and update self
    transform(response, 'DescribeAppResponse', self)
    return self
'''
        assert (
            self.resource_generator.generate_refresh_method(
                "App",
                resource_attributes=[
                    "app_name",
                    "domain_id",
                    "user_profile_name",
                    "space_name",
                    "app_type",
                    "app_name",
                ],
            )
            == expected_output
        )

    def test_generate_delete_method(self):
        expected_output = '''
@Base.add_validate_call
def delete(
    self,

    ) -> None:
    """
    Delete a CompilationJob resource
    
    Raises:
        botocore.exceptions.ClientError: This exception is raised for AWS service related errors. 
            The error message and error code can be parsed from the exception as follows:
            ```
            try:
                # AWS service call here
            except botocore.exceptions.ClientError as e:
                error_message = e.response['Error']['Message']
                error_code = e.response['Error']['Code']
            ```
        ResourceNotFound: Resource being access is not found.
    """

    client = Base.get_sagemaker_client()

    operation_input_args = {
        'CompilationJobName': self.compilation_job_name,
    }
    # serialize the input request
    operation_input_args = serialize(operation_input_args)
    logger.debug(f"Serialized input request: {operation_input_args}")

    client.delete_compilation_job(**operation_input_args)
    
    logger.info(f"Deleting {self.__class__.__name__} - {self.get_name()}")
'''
        assert (
            self.resource_generator.generate_delete_method(
                "CompilationJob", resource_attributes=["compilation_job_name"]
            )
            == expected_output
        )

    # create a unit test for generate_stop_method
    def test_generate_stop_method(self):
        expected_output = '''
@Base.add_validate_call
def stop(self) -> None:
    """
    Stop a CompilationJob resource
    
    Raises:
        botocore.exceptions.ClientError: This exception is raised for AWS service related errors. 
            The error message and error code can be parsed from the exception as follows:
            ```
            try:
                # AWS service call here
            except botocore.exceptions.ClientError as e:
                error_message = e.response['Error']['Message']
                error_code = e.response['Error']['Code']
            ```
        ResourceNotFound: Resource being access is not found.
    """

    client = SageMakerClient().client

    operation_input_args = {
        'CompilationJobName': self.compilation_job_name,
    }
    # serialize the input request
    operation_input_args = serialize(operation_input_args)
    logger.debug(f"Serialized input request: {operation_input_args}")

    client.stop_compilation_job(**operation_input_args)

    logger.info(f"Stopping {self.__class__.__name__} - {self.get_name()}")
'''
        assert self.resource_generator.generate_stop_method("CompilationJob") == expected_output

    def test_generate_wait_method(self):
        expected_output = '''
@Base.add_validate_call
def wait(
    self,
    poll: int = 5,
    timeout: Optional[int] = None,
    
) -> None:
    """
    Wait for a CompilationJob resource.
    
    Parameters:
        poll: The number of seconds to wait between each poll.
        timeout: The maximum number of seconds to wait before timing out.
        
    Raises:
        TimeoutExceededError:  If the resource does not reach a terminal state before the timeout.
        FailedStatusError:   If the resource reaches a failed state.
        WaiterError: Raised when an error occurs while waiting.
    
    """
    terminal_states = ['COMPLETED', 'FAILED', 'STOPPED']
    start_time = time.time()

    progress = Progress(SpinnerColumn("bouncingBar"),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
    )
    progress.add_task("Waiting for CompilationJob...")
    status = Status("Current status:")
    

    with Live(
        Panel(
            Group(progress, status),
            title="Wait Log Panel",
            border_style=Style(color=Color.BLUE.value
            )
        ),
        transient=True
    ):
        while True:
            self.refresh()
            current_status = self.compilation_job_status
            status.update(f"Current status: [bold]{current_status}")
            
            if current_status in terminal_states:
                logger.info(f"Final Resource Status: [bold]{current_status}")
                
                if "failed" in current_status.lower():
                    raise FailedStatusError(resource_type="CompilationJob", status=current_status, reason=self.failure_reason)

                return

            if timeout is not None and time.time() - start_time >= timeout:
                raise TimeoutExceededError(resouce_type="CompilationJob", status=current_status)
            time.sleep(poll)
'''
        assert self.resource_generator.generate_wait_method("CompilationJob") == expected_output

    def test_generate_wait_method_with_logs(self):
        expected_output = '''
@Base.add_validate_call
def wait(
    self,
    poll: int = 5,
    timeout: Optional[int] = None,
    logs: Optional[bool] = False,
) -> None:
    """
    Wait for a TrainingJob resource.
    
    Parameters:
        poll: The number of seconds to wait between each poll.
        timeout: The maximum number of seconds to wait before timing out.
        logs: Whether to print logs while waiting.

    Raises:
        TimeoutExceededError:  If the resource does not reach a terminal state before the timeout.
        FailedStatusError:   If the resource reaches a failed state.
        WaiterError: Raised when an error occurs while waiting.
    
    """
    terminal_states = ['Completed', 'Failed', 'Stopped']
    start_time = time.time()

    progress = Progress(SpinnerColumn("bouncingBar"),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
    )
    progress.add_task("Waiting for TrainingJob...")
    status = Status("Current status:")
        
    instance_count = (
                    sum(instance_group.instance_count for instance_group in self.resource_config.instance_groups)
                    if self.resource_config.instance_groups and not isinstance(self.resource_config.instance_groups, Unassigned)
                    else self.resource_config.instance_count
                )
                
    if logs:
        multi_stream_logger = MultiLogStreamHandler(
            log_group_name=f"/aws/sagemaker/TrainingJobs",
            log_stream_name_prefix=self.get_name(),
            expected_stream_count=instance_count
        )


    with Live(
        Panel(
            Group(progress, status),
            title="Wait Log Panel",
            border_style=Style(color=Color.BLUE.value
            )
        ),
        transient=True
    ):
        while True:
            self.refresh()
            current_status = self.training_job_status
            status.update(f"Current status: [bold]{current_status}")
                        
            if logs and multi_stream_logger.ready():
                stream_log_events = multi_stream_logger.get_latest_log_events()
                for stream_id, event in stream_log_events:
                    logger.info(f"{stream_id}:\\n{event['message']}")

            if current_status in terminal_states:
                logger.info(f"Final Resource Status: [bold]{current_status}")
                
                if "failed" in current_status.lower():
                    raise FailedStatusError(resource_type="TrainingJob", status=current_status, reason=self.failure_reason)

                return

            if timeout is not None and time.time() - start_time >= timeout:
                raise TimeoutExceededError(resouce_type="TrainingJob", status=current_status)
            time.sleep(poll)
'''
        assert self.resource_generator.generate_wait_method("TrainingJob") == expected_output

    def test_generate_wait_for_status_method(self):
        expected_output = '''
@Base.add_validate_call
def wait_for_status(
    self,
    target_status: Literal['InService', 'Creating', 'Updating', 'Failed', 'Deleting'],
    poll: int = 5,
    timeout: Optional[int] = None
) -> None:
    """
    Wait for a InferenceComponent resource to reach certain status.
    
    Parameters:
        target_status: The status to wait for.
        poll: The number of seconds to wait between each poll.
        timeout: The maximum number of seconds to wait before timing out.
    
    Raises:
        TimeoutExceededError:  If the resource does not reach a terminal state before the timeout.
        FailedStatusError:   If the resource reaches a failed state.
        WaiterError: Raised when an error occurs while waiting.
    """
    start_time = time.time()

    progress = Progress(SpinnerColumn("bouncingBar"),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
    )
    progress.add_task(f"Waiting for InferenceComponent to reach [bold]{target_status} status...")
    status = Status("Current status:")

    with Live(
        Panel(
            Group(progress, status),
            title="Wait Log Panel",
            border_style=Style(color=Color.BLUE.value
            )
        ),
        transient=True
    ):
        while True:
            self.refresh()
            current_status = self.inference_component_status
            status.update(f"Current status: [bold]{current_status}")

            if target_status == current_status:
                logger.info(f"Final Resource Status: [bold]{current_status}")
                return
            
            if "failed" in current_status.lower():
                raise FailedStatusError(resource_type="InferenceComponent", status=current_status, reason=self.failure_reason)

            if timeout is not None and time.time() - start_time >= timeout:
                raise TimeoutExceededError(resouce_type="InferenceComponent", status=current_status)
            time.sleep(poll)
'''
        assert (
            self.resource_generator.generate_wait_for_status_method("InferenceComponent")
            == expected_output
        )

    def test_generate_wait_for_status_method_without_failed_state(self):
        expected_output = '''
@Base.add_validate_call
def wait_for_status(
    self,
    target_status: Literal['Creating', 'Created', 'Updating', 'Running', 'Starting', 'Stopping', 'Completed', 'Cancelled'],
    poll: int = 5,
    timeout: Optional[int] = None
) -> None:
    """
    Wait for a InferenceExperiment resource to reach certain status.
    
    Parameters:
        target_status: The status to wait for.
        poll: The number of seconds to wait between each poll.
        timeout: The maximum number of seconds to wait before timing out.
    
    Raises:
        TimeoutExceededError:  If the resource does not reach a terminal state before the timeout.
        FailedStatusError:   If the resource reaches a failed state.
        WaiterError: Raised when an error occurs while waiting.
    """
    start_time = time.time()

    progress = Progress(SpinnerColumn("bouncingBar"),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
    )
    progress.add_task(f"Waiting for InferenceExperiment to reach [bold]{target_status} status...")
    status = Status("Current status:")

    with Live(
        Panel(
            Group(progress, status),
            title="Wait Log Panel",
            border_style=Style(color=Color.BLUE.value
            )
        ),
        transient=True
    ):
        while True:
            self.refresh()
            current_status = self.status
            status.update(f"Current status: [bold]{current_status}")

            if target_status == current_status:
                logger.info(f"Final Resource Status: [bold]{current_status}")
                return

            if timeout is not None and time.time() - start_time >= timeout:
                raise TimeoutExceededError(resouce_type="InferenceExperiment", status=current_status)
            time.sleep(poll)
'''
        assert (
            self.resource_generator.generate_wait_for_status_method("InferenceExperiment")
            == expected_output
        )

    def test_generate_invoke_method(self):
        expected_output = '''

@Base.add_validate_call
def invoke(
    self,
    body: Any,
    content_type: Optional[str] = Unassigned(),
    accept: Optional[str] = Unassigned(),
    custom_attributes: Optional[str] = Unassigned(),
    target_model: Optional[str] = Unassigned(),
    target_variant: Optional[str] = Unassigned(),
    target_container_hostname: Optional[str] = Unassigned(),
    inference_id: Optional[str] = Unassigned(),
    enable_explanations: Optional[str] = Unassigned(),
    inference_component_name: Optional[str] = Unassigned(),
    session_id: Optional[str] = Unassigned(),
    session: Optional[Session] = None,
    region: Optional[str] = None,
) -> Optional[InvokeEndpointOutput]:
    """
    After you deploy a model into production using Amazon SageMaker hosting services, your client applications use this API to get inferences from the model hosted at the specified endpoint.
    
    Parameters:
        body: Provides input data, in the format specified in the ContentType request header. Amazon SageMaker passes all of the data in the body to the model.  For information about the format of the request body, see Common Data Formats-Inference.
        content_type: The MIME type of the input data in the request body.
        accept: The desired MIME type of the inference response from the model container.
        custom_attributes: Provides additional information about a request for an inference submitted to a model hosted at an Amazon SageMaker endpoint. The information is an opaque value that is forwarded verbatim. You could use this value, for example, to provide an ID that you can use to track a request or to provide other metadata that a service endpoint was programmed to process. The value must consist of no more than 1024 visible US-ASCII characters as specified in Section 3.3.6. Field Value Components of the Hypertext Transfer Protocol (HTTP/1.1).  The code in your model is responsible for setting or updating any custom attributes in the response. If your code does not set this value in the response, an empty value is returned. For example, if a custom attribute represents the trace ID, your model can prepend the custom attribute with Trace ID: in your post-processing function.  This feature is currently supported in the Amazon Web Services SDKs but not in the Amazon SageMaker Python SDK. 
        target_model: The model to request for inference when invoking a multi-model endpoint.
        target_variant: Specify the production variant to send the inference request to when invoking an endpoint that is running two or more variants. Note that this parameter overrides the default behavior for the endpoint, which is to distribute the invocation traffic based on the variant weights. For information about how to use variant targeting to perform a/b testing, see Test models in production 
        target_container_hostname: If the endpoint hosts multiple containers and is configured to use direct invocation, this parameter specifies the host name of the container to invoke.
        inference_id: If you provide a value, it is added to the captured data when you enable data capture on the endpoint. For information about data capture, see Capture Data.
        enable_explanations: An optional JMESPath expression used to override the EnableExplanations parameter of the ClarifyExplainerConfig API. See the EnableExplanations section in the developer guide for more information. 
        inference_component_name: If the endpoint hosts one or more inference components, this parameter specifies the name of inference component to invoke.
        session_id: Creates a stateful session or identifies an existing one. You can do one of the following:   Create a stateful session by specifying the value NEW_SESSION.   Send your request to an existing stateful session by specifying the ID of that session.   With a stateful session, you can send multiple requests to a stateful model. When you create a session with a stateful model, the model must create the session ID and set the expiration time. The model must also provide that information in the response to your request. You can get the ID and timestamp from the NewSessionId response parameter. For any subsequent request where you specify that session ID, SageMaker routes the request to the same instance that supports the session.
        session: Boto3 session.
        region: Region name.
    
    Returns:
        InvokeEndpointOutput
    
    Raises:
        botocore.exceptions.ClientError: This exception is raised for AWS service related errors. 
            The error message and error code can be parsed from the exception as follows:
            ```
            try:
                # AWS service call here
            except botocore.exceptions.ClientError as e:
                error_message = e.response['Error']['Message']
                error_code = e.response['Error']['Code']
            ```
        InternalDependencyException: Your request caused an exception with an internal dependency. Contact customer support.
        InternalFailure: An internal failure occurred. Try your request again. If the problem persists, contact Amazon Web Services customer support.
        ModelError: Model (owned by the customer in the container) returned 4xx or 5xx error code.
        ModelNotReadyException: Either a serverless endpoint variant's resources are still being provisioned, or a multi-model endpoint is still downloading or loading the target model. Wait and try your request again.
        ServiceUnavailable: The service is currently unavailable.
        ValidationError: There was an error validating your request.
    """


    use_serializer = False
    if ((self.serializer is not None and self.deserializer is None) or
    (self.serializer is None and self.deserializer is not None)):
        raise ValueError("Both serializer and deserializer must be provided together, or neither should be provided")
    if self.serializer is not None and self.deserializer is not None:
        use_serializer = True
    if use_serializer:
        body = self.serializer.serialize(body)
    operation_input_args = {
        'EndpointName': self.endpoint_name,
        'Body': body,
        'ContentType': content_type,
        'Accept': accept,
        'CustomAttributes': custom_attributes,
        'TargetModel': target_model,
        'TargetVariant': target_variant,
        'TargetContainerHostname': target_container_hostname,
        'InferenceId': inference_id,
        'EnableExplanations': enable_explanations,
        'InferenceComponentName': inference_component_name,
        'SessionId': session_id,
    }
    # serialize the input request
    operation_input_args = serialize(operation_input_args)
    logger.debug(f"Serialized input request: {operation_input_args}")

    client = Base.get_sagemaker_client(session=session, region_name=region, service_name='sagemaker-runtime')

    logger.debug(f"Calling invoke_endpoint API")
    response = client.invoke_endpoint(**operation_input_args)
    logger.debug(f"Response: {response}")

    transformed_response = transform(response, 'InvokeEndpointOutput')
    # Deserialize the body if a deserializer is provided
    if use_serializer:
        body_content = transformed_response["body"]
        deserialized_body = self.deserializer.deserialize(body_content, transformed_response["content_type"])
        transformed_response["body"] = deserialized_body
    return InvokeEndpointOutput(**transformed_response)
'''
        method = Method(
            **{
                "operation_name": "InvokeEndpoint",
                "resource_name": "Endpoint",
                "method_name": "invoke",
                "return_type": "InvokeEndpointOutput",
                "method_type": "object",
                "service_name": "sagemaker-runtime",
            }
        )
        method.get_docstring_title(self.resource_generator.operations["InvokeEndpoint"])
        assert self.resource_generator.generate_method(method, ["endpoint_name"]) == expected_output

    def test_generate_invoke_async_method(self):
        expected_output = '''

@Base.add_validate_call
def invoke_async(
    self,
    input_location: str,
    content_type: Optional[str] = Unassigned(),
    accept: Optional[str] = Unassigned(),
    custom_attributes: Optional[str] = Unassigned(),
    inference_id: Optional[str] = Unassigned(),
    request_ttl_seconds: Optional[int] = Unassigned(),
    invocation_timeout_seconds: Optional[int] = Unassigned(),
    session: Optional[Session] = None,
    region: Optional[str] = None,
) -> Optional[InvokeEndpointAsyncOutput]:
    """
    After you deploy a model into production using Amazon SageMaker hosting services, your client applications use this API to get inferences from the model hosted at the specified endpoint in an asynchronous manner.
    
    Parameters:
        input_location: The Amazon S3 URI where the inference request payload is stored.
        content_type: The MIME type of the input data in the request body.
        accept: The desired MIME type of the inference response from the model container.
        custom_attributes: Provides additional information about a request for an inference submitted to a model hosted at an Amazon SageMaker endpoint. The information is an opaque value that is forwarded verbatim. You could use this value, for example, to provide an ID that you can use to track a request or to provide other metadata that a service endpoint was programmed to process. The value must consist of no more than 1024 visible US-ASCII characters as specified in Section 3.3.6. Field Value Components of the Hypertext Transfer Protocol (HTTP/1.1).  The code in your model is responsible for setting or updating any custom attributes in the response. If your code does not set this value in the response, an empty value is returned. For example, if a custom attribute represents the trace ID, your model can prepend the custom attribute with Trace ID: in your post-processing function.  This feature is currently supported in the Amazon Web Services SDKs but not in the Amazon SageMaker Python SDK. 
        inference_id: The identifier for the inference request. Amazon SageMaker will generate an identifier for you if none is specified. 
        request_ttl_seconds: Maximum age in seconds a request can be in the queue before it is marked as expired. The default is 6 hours, or 21,600 seconds.
        invocation_timeout_seconds: Maximum amount of time in seconds a request can be processed before it is marked as expired. The default is 15 minutes, or 900 seconds.
        session: Boto3 session.
        region: Region name.
    
    Returns:
        InvokeEndpointAsyncOutput
    
    Raises:
        botocore.exceptions.ClientError: This exception is raised for AWS service related errors. 
            The error message and error code can be parsed from the exception as follows:
            ```
            try:
                # AWS service call here
            except botocore.exceptions.ClientError as e:
                error_message = e.response['Error']['Message']
                error_code = e.response['Error']['Code']
            ```
        InternalFailure: An internal failure occurred. Try your request again. If the problem persists, contact Amazon Web Services customer support.
        ServiceUnavailable: The service is currently unavailable.
        ValidationError: There was an error validating your request.
    """


    operation_input_args = {
        'EndpointName': self.endpoint_name,
        'ContentType': content_type,
        'Accept': accept,
        'CustomAttributes': custom_attributes,
        'InferenceId': inference_id,
        'InputLocation': input_location,
        'RequestTTLSeconds': request_ttl_seconds,
        'InvocationTimeoutSeconds': invocation_timeout_seconds,
    }
    # serialize the input request
    operation_input_args = serialize(operation_input_args)
    logger.debug(f"Serialized input request: {operation_input_args}")

    client = Base.get_sagemaker_client(session=session, region_name=region, service_name='sagemaker-runtime')

    logger.debug(f"Calling invoke_endpoint_async API")
    response = client.invoke_endpoint_async(**operation_input_args)
    logger.debug(f"Response: {response}")

    transformed_response = transform(response, 'InvokeEndpointAsyncOutput')
    return InvokeEndpointAsyncOutput(**transformed_response)
'''
        method = Method(
            **{
                "operation_name": "InvokeEndpointAsync",
                "resource_name": "Endpoint",
                "method_name": "invoke_async",
                "return_type": "InvokeEndpointAsyncOutput",
                "method_type": "object",
                "service_name": "sagemaker-runtime",
            }
        )
        method.get_docstring_title(self.resource_generator.operations["InvokeEndpointAsync"])
        assert self.resource_generator.generate_method(method, ["endpoint_name"]) == expected_output

    def test_generate_invoke_with_response_stream_method(self):
        expected_output = '''

@Base.add_validate_call
def invoke_with_response_stream(
    self,
    body: Any,
    content_type: Optional[str] = Unassigned(),
    accept: Optional[str] = Unassigned(),
    custom_attributes: Optional[str] = Unassigned(),
    target_variant: Optional[str] = Unassigned(),
    target_container_hostname: Optional[str] = Unassigned(),
    inference_id: Optional[str] = Unassigned(),
    inference_component_name: Optional[str] = Unassigned(),
    session_id: Optional[str] = Unassigned(),
    session: Optional[Session] = None,
    region: Optional[str] = None,
) -> Optional[InvokeEndpointWithResponseStreamOutput]:
    """
    Invokes a model at the specified endpoint to return the inference response as a stream.
    
    Parameters:
        body: Provides input data, in the format specified in the ContentType request header. Amazon SageMaker passes all of the data in the body to the model.  For information about the format of the request body, see Common Data Formats-Inference.
        content_type: The MIME type of the input data in the request body.
        accept: The desired MIME type of the inference response from the model container.
        custom_attributes: Provides additional information about a request for an inference submitted to a model hosted at an Amazon SageMaker endpoint. The information is an opaque value that is forwarded verbatim. You could use this value, for example, to provide an ID that you can use to track a request or to provide other metadata that a service endpoint was programmed to process. The value must consist of no more than 1024 visible US-ASCII characters as specified in Section 3.3.6. Field Value Components of the Hypertext Transfer Protocol (HTTP/1.1).  The code in your model is responsible for setting or updating any custom attributes in the response. If your code does not set this value in the response, an empty value is returned. For example, if a custom attribute represents the trace ID, your model can prepend the custom attribute with Trace ID: in your post-processing function.  This feature is currently supported in the Amazon Web Services SDKs but not in the Amazon SageMaker Python SDK. 
        target_variant: Specify the production variant to send the inference request to when invoking an endpoint that is running two or more variants. Note that this parameter overrides the default behavior for the endpoint, which is to distribute the invocation traffic based on the variant weights. For information about how to use variant targeting to perform a/b testing, see Test models in production 
        target_container_hostname: If the endpoint hosts multiple containers and is configured to use direct invocation, this parameter specifies the host name of the container to invoke.
        inference_id: An identifier that you assign to your request.
        inference_component_name: If the endpoint hosts one or more inference components, this parameter specifies the name of inference component to invoke for a streaming response.
        session_id: The ID of a stateful session to handle your request. You can't create a stateful session by using the InvokeEndpointWithResponseStream action. Instead, you can create one by using the  InvokeEndpoint  action. In your request, you specify NEW_SESSION for the SessionId request parameter. The response to that request provides the session ID for the NewSessionId response parameter.
        session: Boto3 session.
        region: Region name.
    
    Returns:
        InvokeEndpointWithResponseStreamOutput
    
    Raises:
        botocore.exceptions.ClientError: This exception is raised for AWS service related errors. 
            The error message and error code can be parsed from the exception as follows:
            ```
            try:
                # AWS service call here
            except botocore.exceptions.ClientError as e:
                error_message = e.response['Error']['Message']
                error_code = e.response['Error']['Code']
            ```
        InternalFailure: An internal failure occurred. Try your request again. If the problem persists, contact Amazon Web Services customer support.
        InternalStreamFailure: The stream processing failed because of an unknown error, exception or failure. Try your request again.
        ModelError: Model (owned by the customer in the container) returned 4xx or 5xx error code.
        ModelStreamError: An error occurred while streaming the response body. This error can have the following error codes:  ModelInvocationTimeExceeded  The model failed to finish sending the response within the timeout period allowed by Amazon SageMaker.  StreamBroken  The Transmission Control Protocol (TCP) connection between the client and the model was reset or closed.
        ServiceUnavailable: The service is currently unavailable.
        ValidationError: There was an error validating your request.
    """


    operation_input_args = {
        'EndpointName': self.endpoint_name,
        'Body': body,
        'ContentType': content_type,
        'Accept': accept,
        'CustomAttributes': custom_attributes,
        'TargetVariant': target_variant,
        'TargetContainerHostname': target_container_hostname,
        'InferenceId': inference_id,
        'InferenceComponentName': inference_component_name,
        'SessionId': session_id,
    }
    # serialize the input request
    operation_input_args = serialize(operation_input_args)
    logger.debug(f"Serialized input request: {operation_input_args}")

    client = Base.get_sagemaker_client(session=session, region_name=region, service_name='sagemaker-runtime')

    logger.debug(f"Calling invoke_endpoint_with_response_stream API")
    response = client.invoke_endpoint_with_response_stream(**operation_input_args)
    logger.debug(f"Response: {response}")

    transformed_response = transform(response, 'InvokeEndpointWithResponseStreamOutput')
    return InvokeEndpointWithResponseStreamOutput(**transformed_response)
'''
        method = Method(
            **{
                "operation_name": "InvokeEndpointWithResponseStream",
                "resource_name": "Endpoint",
                "method_name": "invoke_with_response_stream",
                "return_type": "InvokeEndpointWithResponseStreamOutput",
                "method_type": "object",
                "service_name": "sagemaker-runtime",
            }
        )
        method.get_docstring_title(
            self.resource_generator.operations["InvokeEndpointWithResponseStream"]
        )
        assert self.resource_generator.generate_method(method, ["endpoint_name"]) == expected_output

    def test_get_all_method(self):
        expected_output = '''
@classmethod
@Base.add_validate_call
def get_all(
    cls,
    sort_order: Optional[str] = Unassigned(),
    sort_by: Optional[str] = Unassigned(),
    domain_id_equals: Optional[str] = Unassigned(),
    user_profile_name_equals: Optional[str] = Unassigned(),
    space_name_equals: Optional[str] = Unassigned(),
    session: Optional[Session] = None,
    region: Optional[str] = None,
) -> ResourceIterator["App"]:
    """
    Get all App resources
    
    Parameters:
        next_token: If the previous response was truncated, you will receive this token. Use it in your next request to receive the next set of results.
        max_results: This parameter defines the maximum number of results that can be return in a single response. The MaxResults parameter is an upper bound, not a target. If there are more results available than the value specified, a NextToken is provided in the response. The NextToken indicates that the user should get the next set of results by providing this token as a part of a subsequent call. The default value for MaxResults is 10.
        sort_order: The sort order for the results. The default is Ascending.
        sort_by: The parameter by which to sort the results. The default is CreationTime.
        domain_id_equals: A parameter to search for the domain ID.
        user_profile_name_equals: A parameter to search by user profile name. If SpaceNameEquals is set, then this value cannot be set.
        space_name_equals: A parameter to search by space name. If UserProfileNameEquals is set, then this value cannot be set.
        session: Boto3 session.
        region: Region name.
    
    Returns:
        Iterator for listed App resources.
    
    Raises:
        botocore.exceptions.ClientError: This exception is raised for AWS service related errors. 
            The error message and error code can be parsed from the exception as follows:
            ```
            try:
                # AWS service call here
            except botocore.exceptions.ClientError as e:
                error_message = e.response['Error']['Message']
                error_code = e.response['Error']['Code']
            ```
    """

    client = Base.get_sagemaker_client(session=session, region_name=region, service_name="sagemaker")
        
    operation_input_args = {
        'SortOrder': sort_order,
        'SortBy': sort_by,
        'DomainIdEquals': domain_id_equals,
        'UserProfileNameEquals': user_profile_name_equals,
        'SpaceNameEquals': space_name_equals,
    }

    # serialize the input request
    operation_input_args = serialize(operation_input_args)
    logger.debug(f"Serialized input request: {operation_input_args}")
    
    return ResourceIterator(
        client=client,
        list_method='list_apps',
        summaries_key='Apps',
        summary_name='AppDetails',
        resource_cls=App,
        list_method_kwargs=operation_input_args
    )
'''
        assert self.resource_generator.generate_get_all_method("App") == expected_output

    def test_get_all_method_with_no_args(self):
        expected_output = '''
@classmethod
@Base.add_validate_call
def get_all(
    cls,
    session: Optional[Session] = None,
    region: Optional[str] = None,
) -> ResourceIterator["Domain"]:
    """
    Get all Domain resources.
    
    Parameters:
        session: Boto3 session.
        region: Region name.

    Returns:
        Iterator for listed Domain resources.

    """
    client = Base.get_sagemaker_client(session=session, region_name=region, service_name="sagemaker")

    return ResourceIterator(
        client=client,
        list_method='list_domains',
        summaries_key='Domains',
        summary_name='DomainDetails',
        resource_cls=Domain
    )
'''
        assert self.resource_generator.generate_get_all_method("Domain") == expected_output

    def test_get_all_method_with_custom_key_mapping(self):
        expected_output = '''
@classmethod
@Base.add_validate_call
def get_all(
    cls,
    endpoint_name: Optional[str] = Unassigned(),
    sort_by: Optional[str] = Unassigned(),
    sort_order: Optional[str] = Unassigned(),
    name_contains: Optional[str] = Unassigned(),
    creation_time_before: Optional[datetime.datetime] = Unassigned(),
    creation_time_after: Optional[datetime.datetime] = Unassigned(),
    session: Optional[Session] = None,
    region: Optional[str] = None,
) -> ResourceIterator["DataQualityJobDefinition"]:
    """
    Get all DataQualityJobDefinition resources
    
    Parameters:
        endpoint_name: A filter that lists the data quality job definitions associated with the specified endpoint.
        sort_by: The field to sort results by. The default is CreationTime.
        sort_order: Whether to sort the results in Ascending or Descending order. The default is Descending.
        next_token: If the result of the previous ListDataQualityJobDefinitions request was truncated, the response includes a NextToken. To retrieve the next set of transform jobs, use the token in the next request.&gt;
        max_results: The maximum number of data quality monitoring job definitions to return in the response.
        name_contains: A string in the data quality monitoring job definition name. This filter returns only data quality monitoring job definitions whose name contains the specified string.
        creation_time_before: A filter that returns only data quality monitoring job definitions created before the specified time.
        creation_time_after: A filter that returns only data quality monitoring job definitions created after the specified time.
        session: Boto3 session.
        region: Region name.
    
    Returns:
        Iterator for listed DataQualityJobDefinition resources.
    
    Raises:
        botocore.exceptions.ClientError: This exception is raised for AWS service related errors. 
            The error message and error code can be parsed from the exception as follows:
            ```
            try:
                # AWS service call here
            except botocore.exceptions.ClientError as e:
                error_message = e.response['Error']['Message']
                error_code = e.response['Error']['Code']
            ```
    """

    client = Base.get_sagemaker_client(session=session, region_name=region, service_name="sagemaker")
        
    operation_input_args = {
        'EndpointName': endpoint_name,
        'SortBy': sort_by,
        'SortOrder': sort_order,
        'NameContains': name_contains,
        'CreationTimeBefore': creation_time_before,
        'CreationTimeAfter': creation_time_after,
    }
    custom_key_mapping = {"monitoring_job_definition_name": "job_definition_name", "monitoring_job_definition_arn": "job_definition_arn"}
    # serialize the input request
    operation_input_args = serialize(operation_input_args)
    logger.debug(f"Serialized input request: {operation_input_args}")
    
    return ResourceIterator(
        client=client,
        list_method='list_data_quality_job_definitions',
        summaries_key='JobDefinitionSummaries',
        summary_name='MonitoringJobDefinitionSummary',
        resource_cls=DataQualityJobDefinition,
        custom_key_mapping=custom_key_mapping,
        list_method_kwargs=operation_input_args
    )
'''
        assert (
            self.resource_generator.generate_get_all_method("DataQualityJobDefinition")
            == expected_output
        )

    def test_get_node(self):
        expected_output = '''

@Base.add_validate_call
def get_node(
    self,
    node_id: str,
    session: Optional[Session] = None,
    region: Optional[str] = None,
) -> Optional[ClusterNodeDetails]:
    """
    Retrieves information of a node (also called a instance interchangeably) of a SageMaker HyperPod cluster.
    
    Parameters:
        node_id: The ID of the SageMaker HyperPod cluster node.
        session: Boto3 session.
        region: Region name.
    
    Returns:
        ClusterNodeDetails
    
    Raises:
        botocore.exceptions.ClientError: This exception is raised for AWS service related errors. 
            The error message and error code can be parsed from the exception as follows:
            ```
            try:
                # AWS service call here
            except botocore.exceptions.ClientError as e:
                error_message = e.response['Error']['Message']
                error_code = e.response['Error']['Code']
            ```
        ResourceNotFound: Resource being access is not found.
    """


    operation_input_args = {
        'ClusterName': self.cluster_name,
        'NodeId': node_id,
    }
    # serialize the input request
    operation_input_args = serialize(operation_input_args)
    logger.debug(f"Serialized input request: {operation_input_args}")

    client = Base.get_sagemaker_client(session=session, region_name=region, service_name='sagemaker')

    logger.debug(f"Calling describe_cluster_node API")
    response = client.describe_cluster_node(**operation_input_args)
    logger.debug(f"Response: {response}")

    transformed_response = transform(response, 'DescribeClusterNodeResponse')
    return ClusterNodeDetails(**transformed_response)
'''
        method = Method(
            **{
                "operation_name": "DescribeClusterNode",
                "resource_name": "Cluster",
                "method_name": "get_node",
                "return_type": "ClusterNodeDetails",
                "method_type": "object",
                "service_name": "sagemaker",
            }
        )
        method.get_docstring_title(self.resource_generator.operations["DescribeClusterNode"])
        assert self.resource_generator.generate_method(method, ["cluster_name"]) == expected_output

    def test_update_weights_and_capacities(self):
        expected_output = '''

@Base.add_validate_call
def update_weights_and_capacities(
    self,
    desired_weights_and_capacities: List[DesiredWeightAndCapacity],
    session: Optional[Session] = None,
    region: Optional[str] = None,
) -> None:
    """
    Updates variant weight of one or more variants associated with an existing endpoint, or capacity of one variant associated with an existing endpoint.
    
    Parameters:
        desired_weights_and_capacities: An object that provides new capacity and weight values for a variant.
        session: Boto3 session.
        region: Region name.
    
    Raises:
        botocore.exceptions.ClientError: This exception is raised for AWS service related errors. 
            The error message and error code can be parsed from the exception as follows:
            ```
            try:
                # AWS service call here
            except botocore.exceptions.ClientError as e:
                error_message = e.response['Error']['Message']
                error_code = e.response['Error']['Code']
            ```
        ResourceLimitExceeded: You have exceeded an SageMaker resource limit. For example, you might have too many training jobs created.
    """


    operation_input_args = {
        'EndpointName': self.endpoint_name,
        'DesiredWeightsAndCapacities': desired_weights_and_capacities,
    }
    # serialize the input request
    operation_input_args = serialize(operation_input_args)
    logger.debug(f"Serialized input request: {operation_input_args}")

    client = Base.get_sagemaker_client(session=session, region_name=region, service_name='sagemaker')

    logger.debug(f"Calling update_endpoint_weights_and_capacities API")
    response = client.update_endpoint_weights_and_capacities(**operation_input_args)
    logger.debug(f"Response: {response}")

'''
        method = Method(
            **{
                "operation_name": "UpdateEndpointWeightsAndCapacities",
                "resource_name": "Endpoint",
                "method_name": "update_weights_and_capacities",
                "return_type": "None",
                "method_type": "object",
                "service_name": "sagemaker",
            }
        )
        method.get_docstring_title(
            self.resource_generator.operations["UpdateEndpointWeightsAndCapacities"]
        )
        assert self.resource_generator.generate_method(method, ["endpoint_name"]) == expected_output

    def test_get_all_training_jobs(self):
        expected_output = '''

@Base.add_validate_call
def get_all_training_jobs(
    self,
    status_equals: Optional[str] = Unassigned(),
    sort_by: Optional[str] = Unassigned(),
    sort_order: Optional[str] = Unassigned(),    session: Optional[Session] = None,
    region: Optional[str] = None,
) -> ResourceIterator[HyperParameterTrainingJobSummary]:
    """
    Gets a list of TrainingJobSummary objects that describe the training jobs that a hyperparameter tuning job launched.
    
    Parameters:
        next_token: If the result of the previous ListTrainingJobsForHyperParameterTuningJob request was truncated, the response includes a NextToken. To retrieve the next set of training jobs, use the token in the next request.
        max_results: The maximum number of training jobs to return. The default value is 10.
        status_equals: A filter that returns only training jobs with the specified status.
        sort_by: The field to sort results by. The default is Name. If the value of this field is FinalObjectiveMetricValue, any training jobs that did not return an objective metric are not listed.
        sort_order: The sort order for results. The default is Ascending.
        session: Boto3 session.
        region: Region name.
    
    Returns:
        Iterator for listed HyperParameterTrainingJobSummary.
    
    Raises:
        botocore.exceptions.ClientError: This exception is raised for AWS service related errors. 
            The error message and error code can be parsed from the exception as follows:
            ```
            try:
                # AWS service call here
            except botocore.exceptions.ClientError as e:
                error_message = e.response['Error']['Message']
                error_code = e.response['Error']['Code']
            ```
        ResourceNotFound: Resource being access is not found.
    """


    operation_input_args = {
        'HyperParameterTuningJobName': self.hyper_parameter_tuning_job_name,
        'StatusEquals': status_equals,
        'SortBy': sort_by,
        'SortOrder': sort_order,
    }
    # serialize the input request
    operation_input_args = serialize(operation_input_args)
    logger.debug(f"Serialized input request: {operation_input_args}")

    client = Base.get_sagemaker_client(session=session, region_name=region, service_name='sagemaker')


    return ResourceIterator(
        client=client,
        list_method='list_training_jobs_for_hyper_parameter_tuning_job',
        summaries_key='TrainingJobSummaries',
        summary_name='HyperParameterTrainingJobSummary',
        resource_cls=HyperParameterTrainingJobSummary,
        list_method_kwargs=operation_input_args
    )
'''
        method = Method(
            **{
                "operation_name": "ListTrainingJobsForHyperParameterTuningJob",
                "resource_name": "HyperParameterTuningJob",
                "method_name": "get_all_training_jobs",
                "return_type": "HyperParameterTrainingJobSummary",
                "method_type": "object",
                "service_name": "sagemaker",
            }
        )
        method.get_docstring_title(
            self.resource_generator.operations["ListTrainingJobsForHyperParameterTuningJob"]
        )
        assert (
            self.resource_generator.generate_method(method, ["hyper_parameter_tuning_job_name"])
            == expected_output
        )

    def test_generate_presigned_domain_url(self):
        expected_output = '''class PresignedDomainUrl(Base):
    """
    Class representing resource PresignedDomainUrl
    
    Attributes:
        domain_id: The domain ID.
        user_profile_name: The name of the UserProfile to sign-in as.
        session_expiration_duration_in_seconds: The session expiration duration in seconds. This value defaults to 43200.
        expires_in_seconds: The number of seconds until the pre-signed URL expires. This value defaults to 300.
        space_name: The name of the space.
        landing_uri: The landing page that the user is directed to when accessing the presigned URL. Using this value, users can access Studio or Studio Classic, even if it is not the default experience for the domain. The supported values are:    studio::relative/path: Directs users to the relative path in Studio.    app:JupyterServer:relative/path: Directs users to the relative path in the Studio Classic application.    app:JupyterLab:relative/path: Directs users to the relative path in the JupyterLab application.    app:RStudioServerPro:relative/path: Directs users to the relative path in the RStudio application.    app:CodeEditor:relative/path: Directs users to the relative path in the Code Editor, based on Code-OSS, Visual Studio Code - Open Source application.    app:Canvas:relative/path: Directs users to the relative path in the Canvas application.  
        authorized_url: The presigned URL.
    
    """
    domain_id: str
    user_profile_name: Union[str, object]
    session_expiration_duration_in_seconds: Optional[int] = Unassigned()
    expires_in_seconds: Optional[int] = Unassigned()
    space_name: Optional[Union[str, object]] = Unassigned()
    landing_uri: Optional[str] = Unassigned()
    authorized_url: Optional[str] = Unassigned()
    
    def get_name(self) -> str:
        attributes = vars(self)
        resource_name = 'presigned_domain_url_name'
        resource_name_split = resource_name.split('_')
        attribute_name_candidates = []
        
        l = len(resource_name_split)
        for i in range(0, l):
            attribute_name_candidates.append("_".join(resource_name_split[i:l]))
        
        for attribute, value in attributes.items():
            if attribute == 'name' or attribute in attribute_name_candidates:
                return value
        logger.error("Name attribute not found for object presigned_domain_url")
        return None
    
    @classmethod
    @Base.add_validate_call
    def create(
        cls,
        domain_id: str,
        user_profile_name: Union[str, object],
        session_expiration_duration_in_seconds: Optional[int] = Unassigned(),
        expires_in_seconds: Optional[int] = Unassigned(),
        space_name: Optional[Union[str, object]] = Unassigned(),
        landing_uri: Optional[str] = Unassigned(),
        session: Optional[Session] = None,
        region: Optional[str] = None,
    ) -> Optional["PresignedDomainUrl"]:
        """
        Create a PresignedDomainUrl resource
        
        Parameters:
            domain_id: The domain ID.
            user_profile_name: The name of the UserProfile to sign-in as.
            session_expiration_duration_in_seconds: The session expiration duration in seconds. This value defaults to 43200.
            expires_in_seconds: The number of seconds until the pre-signed URL expires. This value defaults to 300.
            space_name: The name of the space.
            landing_uri: The landing page that the user is directed to when accessing the presigned URL. Using this value, users can access Studio or Studio Classic, even if it is not the default experience for the domain. The supported values are:    studio::relative/path: Directs users to the relative path in Studio.    app:JupyterServer:relative/path: Directs users to the relative path in the Studio Classic application.    app:JupyterLab:relative/path: Directs users to the relative path in the JupyterLab application.    app:RStudioServerPro:relative/path: Directs users to the relative path in the RStudio application.    app:CodeEditor:relative/path: Directs users to the relative path in the Code Editor, based on Code-OSS, Visual Studio Code - Open Source application.    app:Canvas:relative/path: Directs users to the relative path in the Canvas application.  
            session: Boto3 session.
            region: Region name.
        
        Returns:
            The PresignedDomainUrl resource.
        
        Raises:
            botocore.exceptions.ClientError: This exception is raised for AWS service related errors. 
                The error message and error code can be parsed from the exception as follows:
                ```
                try:
                    # AWS service call here
                except botocore.exceptions.ClientError as e:
                    error_message = e.response['Error']['Message']
                    error_code = e.response['Error']['Code']
                ```
            ResourceNotFound: Resource being access is not found.
            ConfigSchemaValidationError: Raised when a configuration file does not adhere to the schema
            LocalConfigNotFoundError: Raised when a configuration file is not found in local file system
            S3ConfigNotFoundError: Raised when a configuration file is not found in S3
        """
    
    
        operation_input_args = {
            'DomainId': domain_id,
            'UserProfileName': user_profile_name,
            'SessionExpirationDurationInSeconds': session_expiration_duration_in_seconds,
            'ExpiresInSeconds': expires_in_seconds,
            'SpaceName': space_name,
            'LandingUri': landing_uri,
        }
        # serialize the input request
        operation_input_args = serialize(operation_input_args)
        logger.debug(f"Serialized input request: {operation_input_args}")
    
        client = Base.get_sagemaker_client(session=session, region_name=region, service_name='sagemaker')
    
        logger.debug(f"Calling create_presigned_domain_url API")
        response = client.create_presigned_domain_url(**operation_input_args)
        logger.debug(f"Response: {response}")
    
        transformed_response = transform(response, 'CreatePresignedDomainUrlResponse')
        return cls(**operation_input_args, **transformed_response)
'''
        assert (
            self.resource_generator.generate_resource_class(
                "PresignedDomainUrl", ["create"], [], [], [], [], []
            )
            == expected_output
        )

    def test_generate_sagemaker_servicecatalog_portfolio(self):
        expected_output = '''class SagemakerServicecatalogPortfolio(Base):
    """
    Class representing resource SagemakerServicecatalogPortfolio
    
    """
    
    @staticmethod
    @Base.add_validate_call
    def disable(
        session: Optional[Session] = None,
        region: Optional[str] = None,
    ) -> None:
        """
        Disables using Service Catalog in SageMaker.
        
        Parameters:
            session: Boto3 session.
            region: Region name.
        
        Raises:
            botocore.exceptions.ClientError: This exception is raised for AWS service related errors. 
                The error message and error code can be parsed from the exception as follows:
                ```
                try:
                    # AWS service call here
                except botocore.exceptions.ClientError as e:
                    error_message = e.response['Error']['Message']
                    error_code = e.response['Error']['Code']
                ```
        """
    
    
    
        client = Base.get_sagemaker_client(session=session, region_name=region, service_name='sagemaker')
    
        logger.debug(f"Calling disable_sagemaker_servicecatalog_portfolio API")
        response = client.disable_sagemaker_servicecatalog_portfolio()
        logger.debug(f"Response: {response}")
    
    
    @staticmethod
    @Base.add_validate_call
    def enable(
        session: Optional[Session] = None,
        region: Optional[str] = None,
    ) -> None:
        """
        Enables using Service Catalog in SageMaker.
        
        Parameters:
            session: Boto3 session.
            region: Region name.
        
        Raises:
            botocore.exceptions.ClientError: This exception is raised for AWS service related errors. 
                The error message and error code can be parsed from the exception as follows:
                ```
                try:
                    # AWS service call here
                except botocore.exceptions.ClientError as e:
                    error_message = e.response['Error']['Message']
                    error_code = e.response['Error']['Code']
                ```
        """
    
    
    
        client = Base.get_sagemaker_client(session=session, region_name=region, service_name='sagemaker')
    
        logger.debug(f"Calling enable_sagemaker_servicecatalog_portfolio API")
        response = client.enable_sagemaker_servicecatalog_portfolio()
        logger.debug(f"Response: {response}")
    
    
    @staticmethod
    @Base.add_validate_call
    def get_status(
        session: Optional[Session] = None,
        region: Optional[str] = None,
    ) -> Optional[str]:
        """
        Gets the status of Service Catalog in SageMaker.
        
        Parameters:
            session: Boto3 session.
            region: Region name.
        
        Returns:
            str
        
        Raises:
            botocore.exceptions.ClientError: This exception is raised for AWS service related errors. 
                The error message and error code can be parsed from the exception as follows:
                ```
                try:
                    # AWS service call here
                except botocore.exceptions.ClientError as e:
                    error_message = e.response['Error']['Message']
                    error_code = e.response['Error']['Code']
                ```
        """
    
    
    
        client = Base.get_sagemaker_client(session=session, region_name=region, service_name='sagemaker')
    
        logger.debug(f"Calling get_sagemaker_servicecatalog_portfolio_status API")
        response = client.get_sagemaker_servicecatalog_portfolio_status()
        logger.debug(f"Response: {response}")
    
        return list(response.values())[0]
'''
        assert (
            self.resource_generator.generate_resource_class(
                "SagemakerServicecatalogPortfolio", [], [], [], [], [], []
            )
            == expected_output
        )

    def test_generate_wait_for_delete_method(self):
        expected_output = '''
@Base.add_validate_call
def wait_for_delete(
    self,
    poll: int = 5,
    timeout: Optional[int] = None,
) -> None:
    """
    Wait for a Domain resource to be deleted.
    
    Parameters:
        poll: The number of seconds to wait between each poll.
        timeout: The maximum number of seconds to wait before timing out.
    
    Raises:
        botocore.exceptions.ClientError: This exception is raised for AWS service related errors. 
            The error message and error code can be parsed from the exception as follows:
            ```
            try:
                # AWS service call here
            except botocore.exceptions.ClientError as e:
                error_message = e.response['Error']['Message']
                error_code = e.response['Error']['Code']
            ```
        TimeoutExceededError:  If the resource does not reach a terminal state before the timeout.
        DeleteFailedStatusError:   If the resource reaches a failed state.
        WaiterError: Raised when an error occurs while waiting.
    """
    start_time = time.time()

    progress = Progress(SpinnerColumn("bouncingBar"),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
    )
    progress.add_task("Waiting for Domain to be deleted...")
    status = Status("Current status:")

    with Live(Panel(Group(progress, status), title="Wait Log Panel", border_style=Style(color=Color.BLUE.value))):
        while True:
            try:
                self.refresh()
                current_status = self.status
                status.update(f"Current status: [bold]{current_status}")
                
                if "delete_failed" in current_status.lower() or "deletefailed" in current_status.lower():
                    raise DeleteFailedStatusError(resource_type="Domain", reason=self.failure_reason)



                if timeout is not None and time.time() - start_time >= timeout:
                    raise TimeoutExceededError(resouce_type="Domain", status=current_status)
            except botocore.exceptions.ClientError as e:
                error_code = e.response["Error"]["Code"]
                
                if "ResourceNotFound" in error_code or "ValidationException" in error_code:
                    logger.info("Resource was not found. It may have been deleted.")
                    return
                raise e
            time.sleep(poll)
'''
        assert self.resource_generator.generate_wait_for_delete_method("Domain") == expected_output

    def test_generate_wait_for_delete_method_without_failed_state(self):
        expected_output = '''
@Base.add_validate_call
def wait_for_delete(
    self,
    poll: int = 5,
    timeout: Optional[int] = None,
) -> None:
    """
    Wait for a Algorithm resource to be deleted.
    
    Parameters:
        poll: The number of seconds to wait between each poll.
        timeout: The maximum number of seconds to wait before timing out.
    
    Raises:
        botocore.exceptions.ClientError: This exception is raised for AWS service related errors. 
            The error message and error code can be parsed from the exception as follows:
            ```
            try:
                # AWS service call here
            except botocore.exceptions.ClientError as e:
                error_message = e.response['Error']['Message']
                error_code = e.response['Error']['Code']
            ```
        TimeoutExceededError:  If the resource does not reach a terminal state before the timeout.
        DeleteFailedStatusError:   If the resource reaches a failed state.
        WaiterError: Raised when an error occurs while waiting.
    """
    start_time = time.time()

    progress = Progress(SpinnerColumn("bouncingBar"),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
    )
    progress.add_task("Waiting for Algorithm to be deleted...")
    status = Status("Current status:")

    with Live(Panel(Group(progress, status), title="Wait Log Panel", border_style=Style(color=Color.BLUE.value))):
        while True:
            try:
                self.refresh()
                current_status = self.algorithm_status
                status.update(f"Current status: [bold]{current_status}")



                if timeout is not None and time.time() - start_time >= timeout:
                    raise TimeoutExceededError(resouce_type="Algorithm", status=current_status)
            except botocore.exceptions.ClientError as e:
                error_code = e.response["Error"]["Code"]
                
                if "ResourceNotFound" in error_code or "ValidationException" in error_code:
                    logger.info("Resource was not found. It may have been deleted.")
                    return
                raise e
            time.sleep(poll)
'''
        assert (
            self.resource_generator.generate_wait_for_delete_method("Algorithm") == expected_output
        )

    def test_generate_wait_for_delete_method_with_deleted_state(self):
        expected_output = '''
@Base.add_validate_call
def wait_for_delete(
    self,
    poll: int = 5,
    timeout: Optional[int] = None,
) -> None:
    """
    Wait for a App resource to be deleted.
    
    Parameters:
        poll: The number of seconds to wait between each poll.
        timeout: The maximum number of seconds to wait before timing out.
    
    Raises:
        botocore.exceptions.ClientError: This exception is raised for AWS service related errors. 
            The error message and error code can be parsed from the exception as follows:
            ```
            try:
                # AWS service call here
            except botocore.exceptions.ClientError as e:
                error_message = e.response['Error']['Message']
                error_code = e.response['Error']['Code']
            ```
        TimeoutExceededError:  If the resource does not reach a terminal state before the timeout.
        DeleteFailedStatusError:   If the resource reaches a failed state.
        WaiterError: Raised when an error occurs while waiting.
    """
    start_time = time.time()

    progress = Progress(SpinnerColumn("bouncingBar"),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
    )
    progress.add_task("Waiting for App to be deleted...")
    status = Status("Current status:")

    with Live(Panel(Group(progress, status), title="Wait Log Panel", border_style=Style(color=Color.BLUE.value))):
        while True:
            try:
                self.refresh()
                current_status = self.status
                status.update(f"Current status: [bold]{current_status}")

                
                if current_status.lower() == "deleted":
                    logger.info("Resource was deleted.")
                    return


                if timeout is not None and time.time() - start_time >= timeout:
                    raise TimeoutExceededError(resouce_type="App", status=current_status)
            except botocore.exceptions.ClientError as e:
                error_code = e.response["Error"]["Code"]
                
                if "ResourceNotFound" in error_code or "ValidationException" in error_code:
                    logger.info("Resource was not found. It may have been deleted.")
                    return
                raise e
            time.sleep(poll)
'''
        assert self.resource_generator.generate_wait_for_delete_method("App") == expected_output

    def test_generate_put_record(self):
        expected_output = '''

@Base.add_validate_call
def put_record(
    self,
    record: List[FeatureValue],
    target_stores: Optional[List[str]] = Unassigned(),
    ttl_duration: Optional[TtlDuration] = Unassigned(),
    session: Optional[Session] = None,
    region: Optional[str] = None,
) -> None:
    """
    The PutRecord API is used to ingest a list of Records into your feature group.
    
    Parameters:
        record: List of FeatureValues to be inserted. This will be a full over-write. If you only want to update few of the feature values, do the following:   Use GetRecord to retrieve the latest record.   Update the record returned from GetRecord.    Use PutRecord to update feature values.  
        target_stores: A list of stores to which you're adding the record. By default, Feature Store adds the record to all of the stores that you're using for the FeatureGroup.
        ttl_duration: Time to live duration, where the record is hard deleted after the expiration time is reached; ExpiresAt = EventTime + TtlDuration. For information on HardDelete, see the DeleteRecord API in the Amazon SageMaker API Reference guide.
        session: Boto3 session.
        region: Region name.
    
    Raises:
        botocore.exceptions.ClientError: This exception is raised for AWS service related errors. 
            The error message and error code can be parsed from the exception as follows:
            ```
            try:
                # AWS service call here
            except botocore.exceptions.ClientError as e:
                error_message = e.response['Error']['Message']
                error_code = e.response['Error']['Code']
            ```
        AccessForbidden: You do not have permission to perform an action.
        InternalFailure: An internal failure occurred. Try your request again. If the problem persists, contact Amazon Web Services customer support.
        ServiceUnavailable: The service is currently unavailable.
        ValidationError: There was an error validating your request.
    """


    operation_input_args = {
        'FeatureGroupName': self.feature_group_name,
        'Record': record,
        'TargetStores': target_stores,
        'TtlDuration': ttl_duration,
    }
    # serialize the input request
    operation_input_args = serialize(operation_input_args)
    logger.debug(f"Serialized input request: {operation_input_args}")

    client = Base.get_sagemaker_client(session=session, region_name=region, service_name='sagemaker-featurestore-runtime')

    logger.debug(f"Calling put_record API")
    response = client.put_record(**operation_input_args)
    logger.debug(f"Response: {response}")

'''

        method = Method(
            **{
                "operation_name": "PutRecord",
                "resource_name": "FeatureGroup",
                "method_name": "put_record",
                "return_type": "None",
                "method_type": "object",
                "service_name": "sagemaker-featurestore-runtime",
            }
        )
        method.get_docstring_title(self.resource_generator.operations["PutRecord"])
        assert (
            self.resource_generator.generate_method(method, ["feature_group_name"])
            == expected_output
        )
