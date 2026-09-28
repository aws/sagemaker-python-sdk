"""Unit tests locking in that the SDK does not hijack stdout logging on import (#4387).

The library must not attach a stdout StreamHandler to the ``sagemaker.config`` logger
or disable its propagation, and it must install a NullHandler on the top-level
``sagemaker`` logger so records are safely discarded until the application configures
logging.
"""

import logging

from sagemaker.core.config.config_utils import get_sagemaker_config_logger


def test_config_logger_does_not_attach_stdout_handler():
    logger = get_sagemaker_config_logger()
    stream_handlers = [
        h
        for h in logger.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.NullHandler)
    ]
    assert stream_handlers == []


def test_config_logger_does_not_disable_propagation():
    logger = logging.getLogger("sagemaker.config")
    logger.propagate = True
    get_sagemaker_config_logger()
    # The library must let records propagate to the application's logging config.
    assert logger.propagate is True


def test_config_logger_defaults_to_info_level_when_unset():
    logger = logging.getLogger("sagemaker.config")
    logger.setLevel(logging.NOTSET)
    assert get_sagemaker_config_logger().level == logging.INFO


def test_root_sagemaker_logger_has_nullhandler_after_import():
    import sagemaker.core  # noqa: F401  # importing installs the NullHandler

    root = logging.getLogger("sagemaker")
    assert any(isinstance(h, logging.NullHandler) for h in root.handlers)
