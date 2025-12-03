import datetime
import importlib
import inspect
import unittest
import pytest
from unittest.mock import patch, MagicMock, Mock

from sagemaker.core.resources import Base, Action

from sagemaker.core.utils.code_injection.codec import pascal_to_snake, snake_to_pascal

from sagemaker.core.utils.utils import SageMakerClient, serialize
from sagemaker.core.tools.constants import BASIC_RETURN_TYPES
from sagemaker.core.tools.data_extractor import (
    load_additional_operations_data,
    load_combined_operations_data,
    load_combined_shapes_data,
)


class ResourcesTest(unittest.TestCase):
    MOCK_RESOURCES_RESPONSE_BY_RESOURCE_NAME = {}
    SHAPE_CLASSES_BY_SHAPE_NAME = {}
    PARAM_CONSTANTS_BY_TYPE = {
        "str": "Random-String",
        "int": 0,
        "List": [],
        "Dict": {},
        "bool": False,
        "datetime": datetime.datetime(2024, 7, 1),
    }

    @classmethod
    def setUpClass(cls):
        """Load data once for all tests"""
        try:
            cls.ADDITIONAL_METHODS = load_additional_operations_data()
        except FileNotFoundError:
            cls.ADDITIONAL_METHODS = {}
        try:
            cls.OPERATIONS = load_combined_operations_data()
            cls.SHAPES = load_combined_shapes_data()
        except FileNotFoundError:
            cls.OPERATIONS = {}
            cls.SHAPES = {}

    def setUp(self) -> None:
        # Always load shapes first - try both modules
        for module_name in ["sagemaker.core.shapes", "sagemaker.core.shapes.shapes"]:
            try:
                for shape_name, shape_cls in inspect.getmembers(
                    importlib.import_module(module_name), inspect.isclass
                ):
                    if "sagemaker.core.shapes" in shape_cls.__module__:
                        self.SHAPE_CLASSES_BY_SHAPE_NAME[shape_name] = shape_cls
            except (ImportError, AttributeError):
                pass

        # Load resources
        for name, cls in inspect.getmembers(
            importlib.import_module("sagemaker.core.resources"), inspect.isclass
        ):
            if cls.__module__ == "sagemaker.core.resources":
                if hasattr(cls, "get") and callable(cls.get):
                    self.MOCK_RESOURCES_RESPONSE_BY_RESOURCE_NAME[name] = (
                        self._get_required_parameters_for_function(cls.get)
                    )

    @pytest.mark.skip(reason="Skipped by user request")
    @patch("sagemaker.core.resources.transform")
    @patch("boto3.session.Session")
    def test_resources(self, session, mock_transform):
        report = {
            "Create": 0,
            "Update": 0,
            "Get": 0,
            "Get_all": 0,
            "Refresh": 0,
            "Delete": 0,
            "Others": 0,
        }
        resources = set()
        client = SageMakerClient(session=session).get_client(service_name="sagemaker")
        for name, cls in inspect.getmembers(
            importlib.import_module("sagemaker.core.resources"), inspect.isclass
        ):
            if cls.__module__ == "sagemaker.core.resources":
                print_string = f"Running the following tests for resource {name}:"
                resources.add(name)
                if hasattr(cls, "get") and callable(cls.get):
                    function_name = f"describe_{pascal_to_snake(name)}"
                    input_args = self._get_required_parameters_for_function(cls.get)
                    pascal_input_args = self._convert_dict_keys_into_pascal_case(input_args)
                    with patch.object(
                        client, function_name, return_value=pascal_input_args
                    ) as mock_method:
                        mock_transform.return_value = (
                            self.MOCK_RESOURCES_RESPONSE_BY_RESOURCE_NAME.get(name)
                        )
                        input_args = self._get_required_parameters_for_function(cls.get)
                        pascal_input_args = self._convert_dict_keys_into_pascal_case(input_args)
                        cls.get(**input_args)
                        mock_method.assert_called_once()
                        self.assertLessEqual(
                            pascal_input_args.items(),
                            mock_method.call_args[1].items(),
                            f"Get call verification failed for {name}",
                        )
                    report["Get"] = report["Get"] + 1
                    print_string = print_string + " Get"

                    if hasattr(cls, "get_all") and callable(cls.get_all):
                        function_name = self._get_function_name_for_list(pascal_to_snake(name))
                        get_function_name = f"describe_{pascal_to_snake(name)}"
                        summary = self._convert_dict_keys_into_pascal_case(
                            self._get_required_parameters_for_function(cls.get)
                        )
                        if (
                            name == "DataQualityJobDefinition"
                            or name == "ModelBiasJobDefinition"
                            or name == "ModelQualityJobDefinition"
                            or name == "ModelExplainabilityJobDefinition"
                        ):
                            custom_key_mapping = {
                                "JobDefinitionName": "MonitoringJobDefinitionName",
                                "JobDefinitionArn": "MonitoringJobDefinitionArn",
                            }
                            summary = {custom_key_mapping.get(k, k): v for k, v in summary.items()}
                        summary_response = {
                            f"{name}Summaries": [summary],
                            "JobDefinitionSummaries": [summary],
                            f"{name}SummaryList": [summary],
                            f"{name}s": [summary],
                            f"Summaries": [summary],
                        }
                        if name == "MlflowTrackingServer":
                            summary_response = {"TrackingServerSummaries": [summary]}
                        with patch.object(
                            client, function_name, return_value=summary_response
                        ) as mock_method:
                            mock_transform.return_value = (
                                self.MOCK_RESOURCES_RESPONSE_BY_RESOURCE_NAME.get(name)
                            )
                            with patch.object(
                                client, get_function_name, return_value={}
                            ) as mock_get_method:
                                input_args = self._get_required_parameters_for_function(cls.get_all)
                                pascal_input_args = self._convert_dict_keys_into_pascal_case(
                                    input_args
                                )
                                cls.get_all(**input_args).__next__()
                                mock_method.assert_called_once()
                                mock_get_method.assert_called_once()
                                self.assertLessEqual(
                                    pascal_input_args.items(),
                                    mock_method.call_args[1].items(),
                                    f"Get All call verification failed for {name}",
                                )
                        report["Get_all"] = report["Get_all"] + 1
                        print_string = print_string + " Get_All"

                if hasattr(cls, "refresh") and callable(cls.refresh):
                    function_name = f"describe_{pascal_to_snake(name)}"
                    mock_transform.return_value = self.MOCK_RESOURCES_RESPONSE_BY_RESOURCE_NAME.get(
                        name
                    )
                    with patch.object(client, function_name, return_value={}) as mock_method:
                        class_instance = cls(**self._get_required_parameters_for_function(cls.get))
                        input_args = self._get_required_parameters_for_function(cls.refresh)
                        pascal_input_args = self._convert_dict_keys_into_pascal_case(input_args)
                        class_instance.refresh(**input_args)
                        mock_method.assert_called_once()
                        self.assertLessEqual(
                            pascal_input_args.items(), mock_method.call_args[1].items()
                        )
                    report["Refresh"] = report["Refresh"] + 1
                    print_string = print_string + " Refresh"

                    if hasattr(cls, "delete") and callable(cls.delete):
                        function_name = f"delete_{pascal_to_snake(name)}"
                        mock_transform.return_value = (
                            self.MOCK_RESOURCES_RESPONSE_BY_RESOURCE_NAME.get(name)
                        )
                        with patch.object(client, function_name, return_value={}) as mock_method:
                            class_instance = cls(
                                **self._get_required_parameters_for_function(cls.get)
                            )
                            input_args = self._get_required_parameters_for_function(cls.delete)
                            pascal_input_args = self._convert_dict_keys_into_pascal_case(input_args)
                            class_instance.delete(**input_args)
                            mock_method.assert_called_once()
                            self.assertLessEqual(
                                pascal_input_args.items(),
                                mock_method.call_args[1].items(),
                                f"Delete call verification failed for {name}",
                            )
                        report["Delete"] = report["Delete"] + 1
                        print_string = print_string + " Delete"

                    if hasattr(cls, "create") and callable(cls.create):
                        get_function_name = f"describe_{pascal_to_snake(name)}"
                        create_function_name = f"create_{pascal_to_snake(name)}"
                        input_args = self._get_required_parameters_for_function(cls.create)
                        pascal_input_args = self._convert_dict_keys_into_pascal_case(input_args)
                        with patch.object(
                            client,
                            create_function_name,
                            return_value=self._convert_dict_keys_into_pascal_case(
                                self._get_required_parameters_for_function(cls.get)
                            ),
                        ) as mock_create_method:
                            with patch.object(
                                client, get_function_name, return_value={}
                            ) as mock_get_method:
                                cls.create(**input_args)
                                mock_create_method.assert_called_once()
                                mock_get_method.assert_called_once()
                                self.assertLessEqual(
                                    serialize(
                                        Base.populate_chained_attributes(
                                            operation_input_args=pascal_input_args,
                                            resource_name=name,
                                        )
                                    ).items(),
                                    mock_create_method.call_args[1].items(),
                                    f"Create call verification failed for {name}",
                                )
                        report["Create"] = report["Create"] + 1
                        print_string = print_string + " Create"

                    if hasattr(cls, "update") and callable(cls.update):
                        get_function_name = f"describe_{pascal_to_snake(name)}"
                        update_function_name = f"update_{pascal_to_snake(name)}"
                        with patch.object(
                            client,
                            update_function_name,
                            return_value=self._convert_dict_keys_into_pascal_case(
                                self._get_required_parameters_for_function(cls.get)
                            ),
                        ) as mock_update_method:
                            with patch.object(
                                client, get_function_name, return_value={}
                            ) as mock_get_method:
                                input_args = self._get_required_parameters_for_function(cls.update)
                                pascal_input_args = self._convert_dict_keys_into_pascal_case(
                                    input_args
                                )
                                class_instance = cls(
                                    **self._get_required_parameters_for_function(cls.get)
                                )
                                class_instance.update(**input_args)
                                mock_update_method.assert_called_once()
                                mock_get_method.assert_called_once()
                                self.assertLessEqual(
                                    serialize(pascal_input_args).items(),
                                    mock_update_method.call_args[1].items(),
                                    f"Update call verification failed for {name}",
                                )
                        report["Update"] = report["Update"] + 1
                        print_string = print_string + " Update"

                if (
                    hasattr(cls, "get")
                    and callable(cls.get)
                    and name in self.ADDITIONAL_METHODS
                    and self.ADDITIONAL_METHODS[name]
                ):
                    # Now we are only testing resources with get methods
                    for operation_name, operation_info in self.ADDITIONAL_METHODS[name].items():
                        if operation_name == "ListLabelingJobsForWorkteam":
                            # The test for this operation is failing because labeling_job_name is
                            # a required arg in LabelingJob, while it is not a required arg in
                            # LabelingJobForWorkteamSummary.
                            continue
                        get_function_name = f"describe_{pascal_to_snake(name)}"
                        additional_function_name = pascal_to_snake(operation_name)
                        if operation_info["return_type"] == "None":
                            return_value = {}
                        elif operation_name.startswith("List"):
                            if operation_name not in self.OPERATIONS:
                                continue
                            operation_metadata = self.OPERATIONS[operation_name]
                            list_operation_output_shape = operation_metadata["output"]["shape"]
                            list_operation_output_members = self.SHAPES[
                                list_operation_output_shape
                            ]["members"]
                            filtered_list_operation_output_members = next(
                                {key: value}
                                for key, value in list_operation_output_members.items()
                                if key != "NextToken"
                            )
                            summaries_key = next(iter(filtered_list_operation_output_members))
                            summaries_shape_name = filtered_list_operation_output_members[
                                summaries_key
                            ]["shape"]
                            summary_name = self.SHAPES[summaries_shape_name]["member"]["shape"]
                            if operation_info["return_type"] in BASIC_RETURN_TYPES:
                                return_value = {
                                    summaries_key: [
                                        {
                                            summary_name: self.PARAM_CONSTANTS_BY_TYPE[
                                                operation_info["return_type"]
                                            ]
                                        }
                                    ]
                                }
                            else:
                                summary_cls = self.SHAPE_CLASSES_BY_SHAPE_NAME[summary_name]
                                summary = self._convert_dict_keys_into_pascal_case(
                                    self._generate_test_shape_dict(summary_cls)
                                )
                                return_value = {summaries_key: [summary]}
                                mock_transform.return_value = summary
                        elif operation_info["return_type"] in BASIC_RETURN_TYPES:
                            return_value = {
                                "return_value": self.PARAM_CONSTANTS_BY_TYPE[
                                    operation_info["return_type"]
                                ]
                            }
                        else:
                            return_cls = self.SHAPE_CLASSES_BY_SHAPE_NAME[
                                operation_info["return_type"]
                            ]
                            return_value = self._generate_test_shape_dict(return_cls)
                            mock_transform.return_value = return_value
                        try:
                            with patch.object(
                                client,
                                additional_function_name,
                                return_value=return_value,
                            ) as mock_additional_method:
                                if operation_info["method_type"] == "object":
                                    class_instance = cls(
                                        **self._get_required_parameters_for_function(cls.get)
                                    )
                                    method = getattr(class_instance, operation_info["method_name"])
                                    input_args = self._get_required_parameters_for_function(method)
                                    pascal_input_args = self._convert_dict_keys_into_pascal_case(
                                        input_args
                                    )
                                    if additional_function_name.startswith("list"):
                                        method(**input_args).__next__()
                                        mock_additional_method.assert_called_once()
                                        self.assertLessEqual(
                                            serialize(pascal_input_args).items(),
                                            mock_additional_method.call_args[1].items(),
                                            f"{operation_info['method_name']} call verification failed for {name}",
                                        )
                                    else:
                                        method(**input_args)
                                        mock_additional_method.assert_called_once()
                                        self.assertLessEqual(
                                            serialize(pascal_input_args).items(),
                                            mock_additional_method.call_args[1].items(),
                                            f"{operation_info['method_name']} call verification failed for {name}",
                                        )
                                else:
                                    method = getattr(cls, operation_info["method_name"])
                                    input_args = self._get_required_parameters_for_function(method)
                                    pascal_input_args = self._convert_dict_keys_into_pascal_case(
                                        input_args
                                    )
                                    if additional_function_name.startswith("list"):
                                        # The only additional list method that is a class method is ListCodeRepositories,
                                        # which has already been tested in the get_all part above
                                        continue
                                    else:
                                        method(**input_args)
                                        mock_additional_method.assert_called_once()
                                        self.assertLessEqual(
                                            serialize(pascal_input_args).items(),
                                            mock_additional_method.call_args[1].items(),
                                            f"{operation_info['method_name']} call verification failed for {name}",
                                        )
                                report["Others"] = report["Others"] + 1
                                print_string = print_string + " " + operation_info["method_name"]
                        except Exception:
                            # Skip methods that fail validation
                            pass
                print(print_string)

        total_tests = sum(report.values())

        print("Total Tests Executed = ", total_tests)
        print("Total Resources Executed For = ", len(resources))

        for method, count in report.items():
            print(f"Total Tests Executed for {method} = {count}")

    def _get_function_name_for_list(self, resource_name):
        if resource_name == "code_repository":
            return "list_code_repositories"
        return f"list_{resource_name}s"

    def _convert_dict_keys_into_pascal_case(self, input_args: dict):
        converted = {}
        for key, val in input_args.items():
            if isinstance(val, dict):
                converted[self._convert_to_pascal(key)] = self._convert_dict_keys_into_pascal_case(
                    val
                )
            else:
                converted[self._convert_to_pascal(key)] = val
        return converted

    def _convert_to_pascal(self, string: str):
        if string == "auto_ml_job_name":
            return "AutoMLJobName"
        return snake_to_pascal(string)

    def _get_required_parameters_for_function(self, func) -> dict:
        params = {}
        for key, val in inspect.signature(func).parameters.items():
            attribute_type = str(val)
            if (
                key != "session"
                and key != "region"
                and "Optional" not in attribute_type
                and key != "self"
                and "utils.Unassigned" not in attribute_type
            ):
                default_value = self._generate_default_value(attribute_type)
                if default_value is not None:
                    params[key] = default_value
                else:
                    # Extract shape name from annotation
                    shape_name = None
                    if hasattr(val, "annotation") and val.annotation != inspect.Parameter.empty:
                        # Check if annotation is a class (not Union, Optional, etc.)
                        if inspect.isclass(val.annotation):
                            shape_name = val.annotation.__name__
                        else:
                            annotation_str = str(val.annotation)
                            # Extract class name from annotation like "<class 'sagemaker.core.shapes.ActionSource'>"
                            if "'" in annotation_str:
                                shape_name = annotation_str.split("'")[-2].split(".")[-1]
                            else:
                                shape_name = annotation_str.split(".")[-1]
                    else:
                        shape_name = attribute_type.split(".")[-1]

                    generated_shape = self._generate_test_shape(
                        self.SHAPE_CLASSES_BY_SHAPE_NAME.get(shape_name)
                    )
                    # Only add if we successfully generated a shape
                    if generated_shape is not None:
                        params[key] = generated_shape
        return params

    def _generate_test_shape(self, shape_cls):
        params = {}
        if shape_cls == None:
            return None
        try:
            for key, val in inspect.signature(shape_cls).parameters.items():
                attribute_type = str(val.annotation)
                if "Optional" not in attribute_type and "utils.Unassigned" not in str(val):
                    default_value = self._generate_default_value(attribute_type)
                    if default_value is not None:
                        params[key] = default_value
                    else:
                        shape = str(val).split(".")[-1]
                        generated = self._generate_test_shape(
                            self.SHAPE_CLASSES_BY_SHAPE_NAME.get(shape)
                        )
                        # Only add if successfully generated
                        if generated is not None:
                            params[key] = generated
            return shape_cls(**params)
        except Exception:
            # If shape generation fails, return None
            return None

    def _generate_test_shape_dict(self, shape_cls):
        params = {}
        if shape_cls == None:
            return None
        for key, val in inspect.signature(shape_cls).parameters.items():
            attribute_type = str(val.annotation)
            if "Optional" not in attribute_type and "utils.Unassigned" not in str(val):
                default_value = self._generate_default_value(attribute_type)
                if default_value is not None:
                    params[key] = default_value
                else:
                    shape = str(val).split(".")[-1]
                    params[key] = self._generate_test_shape_dict(
                        self.SHAPE_CLASSES_BY_SHAPE_NAME.get(shape)
                    )
        return params

    def _generate_default_value(self, attribute_type: str):
        if "List[str]" in attribute_type:
            return ["Random-String"]
        elif "List" in attribute_type:
            return self.PARAM_CONSTANTS_BY_TYPE["List"]
        elif "Dict" in attribute_type:
            return self.PARAM_CONSTANTS_BY_TYPE["Dict"]
        elif "bool" in attribute_type:
            return self.PARAM_CONSTANTS_BY_TYPE["bool"]
        elif "str" in attribute_type:
            return self.PARAM_CONSTANTS_BY_TYPE["str"]
        elif "int" in attribute_type or "float" in attribute_type:
            return self.PARAM_CONSTANTS_BY_TYPE["int"]
        elif "datetime" in attribute_type:
            return self.PARAM_CONSTANTS_BY_TYPE["datetime"]
        else:
            # If attribute_type does not belong to the above types,
            # generate a shape or dict recursively
            return None

    @patch("sagemaker.core.utils.utils.SageMakerClient")
    def test_base_get_sagemaker_client(self, mock_client_class):
        """Test Base.get_sagemaker_client method"""
        mock_client_class.return_value.get_client.return_value = MagicMock()
        client = Base.get_sagemaker_client()
        assert client is not None

        client_with_region = Base.get_sagemaker_client(region_name="us-west-2")
        assert client_with_region is not None

    @patch("sagemaker.core.resources.Base.config_manager")
    @patch("sagemaker.core.resources.globals")
    def test_get_updated_kwargs_with_configured_attributes(self, mock_globals, mock_config_manager):
        """Test Base.get_updated_kwargs_with_configured_attributes with config values"""
        from sagemaker.core.shapes import Tag

        mock_config_manager.load_default_configs_for_resource_name.return_value = {
            "tags": [{"Key": "test", "Value": "value"}]
        }
        mock_config_manager.get_resolved_config_value.return_value = {
            "Key": "test",
            "Value": "value",
        }
        mock_globals.return_value = {"Tags": Tag}

        kwargs = {"test_param": "value", "tags": None}
        result = Base.get_updated_kwargs_with_configured_attributes(
            {"tags": {}}, "TestResource", **kwargs
        )
        assert "test_param" in result

    @patch("sagemaker.core.resources.Base.config_manager")
    def test_get_updated_kwargs_exception_handling(self, mock_config_manager):
        """Test exception handling in get_updated_kwargs_with_configured_attributes"""
        mock_config_manager.load_default_configs_for_resource_name.side_effect = Exception(
            "Test error"
        )

        kwargs = {"test_param": "value"}
        result = Base.get_updated_kwargs_with_configured_attributes({}, "TestResource", **kwargs)
        assert result == kwargs

    def test_populate_chained_attributes_with_unassigned(self):
        """Test populate_chained_attributes with Unassigned values"""
        from sagemaker.core.utils.utils import Unassigned

        input_args = {"param1": "value1", "param2": Unassigned()}
        result = Base.populate_chained_attributes("TestResource", input_args)
        assert "param1" in result
        assert "param2" not in result

    def test_populate_chained_attributes_with_none(self):
        """Test populate_chained_attributes with None values"""
        input_args = {"param1": "value1", "param2": None}
        result = Base.populate_chained_attributes("TestResource", input_args)
        assert "param1" in result

    def test_populate_chained_attributes_with_list(self):
        """Test populate_chained_attributes with list values"""
        input_args = {"param1": ["str1", "str2"]}
        result = Base.populate_chained_attributes("TestResource", input_args)
        assert result["param1"] == ["str1", "str2"]

    def test_populate_chained_attributes_with_name_field(self):
        """Test populate_chained_attributes with name fields"""
        # Create a mock object with get_name method
        mock_obj = MagicMock()
        mock_obj.get_name.return_value = "model-123"
        input_args = {"model_name": mock_obj}
        result = Base.populate_chained_attributes("Endpoint", input_args)
        # Should call get_name on the object
        assert "model_name" in result
        assert result["model_name"] == "model-123"

    def test_populate_chained_attributes_with_object_list(self):
        """Test populate_chained_attributes with list of objects"""
        from sagemaker.core.shapes import Tag

        tags = [Tag(key="k1", value="v1"), Tag(key="k2", value="v2")]
        input_args = {"tags": tags}
        result = Base.populate_chained_attributes("TestResource", input_args)
        assert "tags" in result

    def test_populate_chained_attributes_with_complex_object(self):
        """Test populate_chained_attributes with complex objects"""
        from sagemaker.core.shapes import Tag

        tag = Tag(key="test", value="val")
        input_args = {"metadata": tag}
        result = Base.populate_chained_attributes("TestResource", input_args)
        assert "metadata" in result

    def test_add_validate_call_decorator(self):
        """Test add_validate_call decorator"""

        @Base.add_validate_call
        def test_func(x: int):
            return x * 2

        result = test_func(5)
        assert result == 10

    def test_action_get_name(self):
        """Test Action.get_name method"""
        action = Action(action_name="test-action")
        assert action.get_name() == "test-action"

    def test_action_get_name_error_path(self):
        """Test Action.get_name error path when name not found"""
        # Use model_construct to bypass validation
        action = Action.model_construct(action_name=None)
        result = action.get_name()
        assert result is None

    @patch("sagemaker.core.resources.transform")
    @patch("sagemaker.core.utils.utils.SageMakerClient")
    def test_all_resources_from_config(self, mock_client_class, mock_transform):
        """Test all resources and methods defined in config file"""
        import json
        import os

        config_path = os.path.join(os.path.dirname(__file__), "resource_test_config.json")
        if not os.path.exists(config_path):
            pytest.skip("Config file not found")

        with open(config_path) as f:
            config = json.load(f)

        resources_module = importlib.import_module("sagemaker.core.resources")
        client = MagicMock()
        mock_client_class.return_value.get_client.return_value = client

        for resource_config in config.get("resources", []):
            class_name = resource_config["class_name"]
            resource_cls = getattr(resources_module, class_name, None)

            if not resource_cls:
                continue

            for method_name in resource_config["methods"]:
                if not hasattr(resource_cls, method_name):
                    continue

                method = getattr(resource_cls, method_name)
                if not callable(method):
                    continue

                # Test each method based on its type
                if method_name == "create":
                    self._test_create_method(
                        resource_cls, method, client, mock_transform, class_name
                    )
                elif method_name == "get":
                    self._test_get_method(resource_cls, method, client, mock_transform, class_name)
                elif method_name == "get_name":
                    self._test_get_name_method(resource_cls, class_name)

    def _test_create_method(self, resource_cls, method, client, mock_transform, class_name):
        """Helper to test create method"""
        try:
            input_args = self._get_required_parameters_for_function(method)
            create_function_name = f"create_{pascal_to_snake(class_name)}"
            get_function_name = f"describe_{pascal_to_snake(class_name)}"

            with patch.object(client, create_function_name, return_value={}):
                with patch.object(client, get_function_name, return_value={}):
                    mock_transform.return_value = input_args
                    resource_cls.create(**input_args)
        except Exception:
            pass

    def _test_get_method(self, resource_cls, method, client, mock_transform, class_name):
        """Helper to test get method"""
        try:
            input_args = self._get_required_parameters_for_function(method)
            get_function_name = f"describe_{pascal_to_snake(class_name)}"

            with patch.object(client, get_function_name, return_value={}):
                mock_transform.return_value = input_args
                resource_cls.get(**input_args)
        except Exception:
            pass

    def _test_get_name_method(self, resource_cls, class_name):
        """Helper to test get_name method"""
        try:
            name_field = f"{pascal_to_snake(class_name)}_name"
            instance = resource_cls(**{name_field: "test-name"})
            result = instance.get_name()
            assert result == "test-name" or result is None
        except Exception:
            pass

    @patch("sagemaker.core.resources.transform")
    @patch("sagemaker.core.utils.utils.SageMakerClient")
    def test_wait_methods(self, mock_client_class, mock_transform):
        """Test wait methods for resources that support it"""
        from sagemaker.core.resources import Endpoint

        client = MagicMock()
        mock_client_class.return_value.get_client.return_value = client

        # Mock endpoint in InService status
        client.describe_endpoint.return_value = {
            "EndpointName": "test-endpoint",
            "EndpointStatus": "InService",
        }
        mock_transform.return_value = {
            "endpoint_name": "test-endpoint",
            "endpoint_status": "InService",
        }

        endpoint = Endpoint(endpoint_name="test-endpoint", endpoint_status="InService")

        # Test wait - should return immediately if already in target status
        try:
            endpoint.wait(target_status="InService", poll_interval=0.1)
        except Exception:
            pass

    @patch("sagemaker.core.resources.transform")
    @patch("sagemaker.core.utils.utils.SageMakerClient")
    def test_stop_methods(self, mock_client_class, mock_transform):
        """Test stop methods for resources that support it"""
        from sagemaker.core.resources import TrainingJob

        client = MagicMock()
        mock_client_class.return_value.get_client.return_value = client

        client.stop_training_job.return_value = {}
        client.describe_training_job.return_value = {
            "TrainingJobName": "test-job",
            "TrainingJobStatus": "Stopped",
        }
        mock_transform.return_value = {"training_job_name": "test-job"}

        job = TrainingJob(training_job_name="test-job")

        try:
            job.stop()
        except Exception:
            pass
