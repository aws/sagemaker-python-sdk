from sagemaker.core.config_schema import SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA


def test_config_schema():
    assert SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA

    assert SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA["required"] == ["SageMaker"]
    assert SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA["properties"]["SageMaker"]
    assert SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA["properties"]["SageMaker"]["required"] == [
        "PythonSDK"
    ]

    assert SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA["properties"]["SageMaker"]["properties"]["PythonSDK"]
    assert SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA["properties"]["SageMaker"]["properties"]["PythonSDK"][
        "required"
    ] == ["Resources"]

    assert SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA["properties"]["SageMaker"]["properties"]["PythonSDK"][
        "properties"
    ]["Resources"]
