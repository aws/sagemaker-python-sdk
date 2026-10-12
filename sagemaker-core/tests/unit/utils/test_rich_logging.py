"""Unit tests for opt-in rich console/traceback behavior in sagemaker.core.utils.utils.

These lock in that importing the SDK does not override sys.excepthook or restyle the
process-global rich console unless the user explicitly opts in via
SAGEMAKER_ENABLE_RICH_LOGGING (or a force=True call).
"""

import os
from unittest.mock import patch

import pytest

from sagemaker.core.utils import utils
from sagemaker.core.utils.utils import (
    RICH_LOGGING_OPT_IN_ENV_VAR,
    enable_textual_rich_console_and_traceback,
    is_rich_logging_enabled,
)


@pytest.fixture(autouse=True)
def _reset_latch():
    """Reset the one-shot 'already enabled' latch around every test."""
    saved = utils.textual_rich_console_and_traceback_enabled
    utils.textual_rich_console_and_traceback_enabled = False
    try:
        yield
    finally:
        utils.textual_rich_console_and_traceback_enabled = saved


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "Yes", "on", " on "])
def test_is_rich_logging_enabled_truthy(value):
    with patch.dict(os.environ, {RICH_LOGGING_OPT_IN_ENV_VAR: value}):
        assert is_rich_logging_enabled() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "nope"])
def test_is_rich_logging_enabled_falsey(value):
    with patch.dict(os.environ, {RICH_LOGGING_OPT_IN_ENV_VAR: value}):
        assert is_rich_logging_enabled() is False


def test_disabled_by_default_is_noop():
    # Env var absent -> importing/using the SDK must not touch the global console
    # or install rich tracebacks (no sys.excepthook override).
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop(RICH_LOGGING_OPT_IN_ENV_VAR, None)
        with (
            patch.object(utils, "reconfigure") as mock_reconfigure,
            patch.object(utils, "install") as mock_install,
        ):
            enable_textual_rich_console_and_traceback()
            mock_reconfigure.assert_not_called()
            mock_install.assert_not_called()
            assert utils.textual_rich_console_and_traceback_enabled is False


def test_enabled_when_opted_in_via_env():
    with patch.dict(os.environ, {RICH_LOGGING_OPT_IN_ENV_VAR: "true"}):
        with (
            patch.object(utils, "reconfigure") as mock_reconfigure,
            patch.object(utils, "install") as mock_install,
        ):
            enable_textual_rich_console_and_traceback()
            mock_reconfigure.assert_called_once()
            mock_install.assert_called_once()
            assert utils.textual_rich_console_and_traceback_enabled is True


def test_force_enables_regardless_of_env():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop(RICH_LOGGING_OPT_IN_ENV_VAR, None)
        with (
            patch.object(utils, "reconfigure") as mock_reconfigure,
            patch.object(utils, "install") as mock_install,
        ):
            enable_textual_rich_console_and_traceback(force=True)
            mock_reconfigure.assert_called_once()
            mock_install.assert_called_once()


def test_enable_is_idempotent_when_opted_in():
    with patch.dict(os.environ, {RICH_LOGGING_OPT_IN_ENV_VAR: "1"}):
        with (
            patch.object(utils, "reconfigure") as mock_reconfigure,
            patch.object(utils, "install") as mock_install,
        ):
            enable_textual_rich_console_and_traceback()
            enable_textual_rich_console_and_traceback()
            # The one-shot latch prevents re-installing on the second call.
            mock_reconfigure.assert_called_once()
            mock_install.assert_called_once()


def test_get_logger_does_not_call_basicconfig_when_opted_out():
    # Getting a module logger must not reconfigure the root logger by default.
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop(RICH_LOGGING_OPT_IN_ENV_VAR, None)
        with patch.object(utils.logging, "basicConfig") as mock_basic_config:
            returned = utils.get_textual_rich_logger("sagemaker.core.test.optout")
            mock_basic_config.assert_not_called()
            assert returned is utils.logging.getLogger("sagemaker.core.test.optout")


def test_get_logger_calls_basicconfig_when_opted_in():
    with patch.dict(os.environ, {RICH_LOGGING_OPT_IN_ENV_VAR: "1"}):
        with (
            patch.object(utils, "reconfigure"),
            patch.object(utils, "install"),
            patch.object(utils.logging, "basicConfig") as mock_basic_config,
        ):
            utils.get_textual_rich_logger("sagemaker.core.test.optin")
            mock_basic_config.assert_called_once()
