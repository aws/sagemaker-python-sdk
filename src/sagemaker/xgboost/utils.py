from sagemaker.xgboost import defaults


def validate_py_version(py_version):
    """Placeholder docstring"""
    if py_version != "py3":
        raise ValueError("Unsupported Python version: {}.".format(py_version))


def validate_framework_version(framework_version):
    """Placeholder docstring"""

    xgboost_version = framework_version.split("-")[0]
    if xgboost_version in defaults.XGBOOST_UNSUPPORTED_VERSIONS:
        msg = defaults.XGBOOST_UNSUPPORTED_VERSIONS[xgboost_version]
        raise ValueError(msg)
