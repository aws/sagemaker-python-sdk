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
"""Tests for basic_script_driver module."""
from __future__ import absolute_import

import json
import pytest
from unittest.mock import patch, MagicMock

# Import the module under test
import sys
from pathlib import Path

# Add the container_drivers path to sys.path for imports
container_drivers_path = Path(__file__).parent.parent.parent.parent.parent / "src" / "sagemaker" / "train" / "container_drivers"
sys.path.insert(0, str(container_drivers_path))

from distributed_drivers.basic_script_driver import create_commands, main


class TestCreateCommands:
    """Test create_commands function."""

    @patch.dict("os.environ", {
        "SM_ENTRY_SCRIPT": "train.py",
        "SM_HPS": '{"learning_rate": "0.01", "batch_size": "32"}'
    })
    @patch("distributed_drivers.basic_script_driver.get_python_executable")
    @patch("distributed_drivers.basic_script_driver.hyperparameters_to_cli_args")
    def test_creates_python_command(self, mock_hp_to_args, mock_get_python):
        """Test creates command for Python script."""
        mock_get_python.return_value = "/usr/bin/python3"
        mock_hp_to_args.return_value = ["--learning_rate", "0.01", "--batch_size", "32"]

        commands = create_commands()

        assert commands[0] == "/usr/bin/python3"
        assert commands[1] == "train.py"
        assert "--learning_rate" in commands
        assert "0.01" in commands
        assert "--batch_size" in commands
        assert "32" in commands

    @patch.dict("os.environ", {
        "SM_ENTRY_SCRIPT": "train.sh",
        "SM_HPS": '{"epochs": "10"}'
    })
    @patch("distributed_drivers.basic_script_driver.get_python_executable")
    @patch("distributed_drivers.basic_script_driver.hyperparameters_to_cli_args")
    def test_creates_shell_command(self, mock_hp_to_args, mock_get_python):
        """Test creates command for shell script."""
        mock_hp_to_args.return_value = ["--epochs", "10"]

        commands = create_commands()

        assert commands[0] == "/bin/sh"
        assert commands[1] == "-c"
        assert "chmod +x train.sh" in commands[2]
        assert "./train.sh" in commands[2]
        assert "--epochs" in commands[2]
        assert "10" in commands[2]

    @patch.dict("os.environ", {
        "SM_ENTRY_SCRIPT": "train.py",
        "SM_HPS": '{}'
    })
    @patch("distributed_drivers.basic_script_driver.get_python_executable")
    @patch("distributed_drivers.basic_script_driver.hyperparameters_to_cli_args")
    def test_handles_empty_hyperparameters(self, mock_hp_to_args, mock_get_python):
        """Test handles empty hyperparameters."""
        mock_get_python.return_value = "/usr/bin/python3"
        mock_hp_to_args.return_value = []

        commands = create_commands()

        assert commands == ["/usr/bin/python3", "train.py"]

    @patch.dict("os.environ", {
        "SM_ENTRY_SCRIPT": "train.txt",
        "SM_HPS": '{}'
    })
    @patch("distributed_drivers.basic_script_driver.get_python_executable")
    @patch("distributed_drivers.basic_script_driver.hyperparameters_to_cli_args")
    def test_raises_error_for_unsupported_script_type(self, mock_hp_to_args, mock_get_python):
        """Test raises error for unsupported script type."""
        mock_get_python.return_value = "/usr/bin/python3"
        mock_hp_to_args.return_value = []

        with pytest.raises(ValueError, match="Unsupported entry script type"):
            create_commands()

    @patch.dict("os.environ", {
        "SM_ENTRY_SCRIPT": "train.sh",
        "SM_HPS": '{"arg_with_space": "value with spaces", "special": "value\'with\'quotes"}'
    })
    @patch("distributed_drivers.basic_script_driver.get_python_executable")
    @patch("distributed_drivers.basic_script_driver.hyperparameters_to_cli_args")
    def test_properly_quotes_shell_arguments(self, mock_hp_to_args, mock_get_python):
        """Test properly quotes shell arguments with special characters."""
        mock_hp_to_args.return_value = ["--arg_with_space", "value with spaces", "--special", "value'with'quotes"]

        commands = create_commands()

        # Check that arguments are properly quoted in the shell command
        assert commands[0] == "/bin/sh"
        assert commands[1] == "-c"
        # The command should contain quoted arguments
        assert "train.sh" in commands[2]


class TestMain:
    """Test main function."""

    @patch("distributed_drivers.basic_script_driver.write_failure_file")
    @patch("distributed_drivers.basic_script_driver.execute_commands")
    @patch("distributed_drivers.basic_script_driver.create_commands")
    def test_main_success(self, mock_create_commands, mock_execute, mock_write_failure):
        """Test main function with successful execution."""
        mock_create_commands.return_value = ["/usr/bin/python3", "train.py"]
        mock_execute.return_value = (0, None)

        # Should not raise any exception
        main()

        mock_create_commands.assert_called_once()
        mock_execute.assert_called_once_with(["/usr/bin/python3", "train.py"])
        mock_write_failure.assert_not_called()

    @patch("distributed_drivers.basic_script_driver.write_failure_file")
    @patch("distributed_drivers.basic_script_driver.execute_commands")
    @patch("distributed_drivers.basic_script_driver.create_commands")
    def test_main_failure(self, mock_create_commands, mock_execute, mock_write_failure):
        """Test main function with failed execution."""
        mock_create_commands.return_value = ["/usr/bin/python3", "train.py"]
        mock_execute.return_value = (1, "Error traceback")

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
        mock_write_failure.assert_called_once_with("Error traceback")

    @patch("distributed_drivers.basic_script_driver.write_failure_file")
    @patch("distributed_drivers.basic_script_driver.execute_commands")
    @patch("distributed_drivers.basic_script_driver.create_commands")
    def test_main_logs_command(self, mock_create_commands, mock_execute, mock_write_failure):
        """Test main function logs the command being executed."""
        mock_create_commands.return_value = ["/usr/bin/python3", "train.py", "--arg", "value"]
        mock_execute.return_value = (0, None)

        with patch("distributed_drivers.basic_script_driver.logger") as mock_logger:
            main()

            # Verify logger was called with the command
            mock_logger.info.assert_called()
            call_args = mock_logger.info.call_args[0][0]
            assert "Executing command:" in call_args
            assert "/usr/bin/python3" in call_args
            assert "train.py" in call_args
