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
"""This module is the entry point for the Basic Script Driver."""
from __future__ import absolute_import

import sys
import shlex

from typing import List

from utils import (
    logger,
    get_python_executable,
    read_source_code_json,
    read_hyperparameters_json,
    execute_commands,
    write_failure_file,
    hyperparameters_to_cli_args,
)


def create_commands() -> List[str]:
    """Create the commands to execute."""
    source_code = read_source_code_json()
    hyperparameters = read_hyperparameters_json()
    python_executable = get_python_executable()

    entry_script = source_code["entry_script"]
    args = hyperparameters_to_cli_args(hyperparameters)
    if entry_script.endswith(".py"):
        commands = [python_executable, entry_script]
        commands += args
    elif entry_script.endswith(".sh"):
        args_str = " ".join(shlex.quote(arg) for arg in args)
        commands = [
            "/bin/sh",
            "-c",
            f"chmod +x {entry_script} && ./{entry_script} {args_str}",
        ]
    else:
        raise ValueError(
            f"Unsupported entry script type: {entry_script}. Only .py and .sh are supported."
        )
    return commands


def main():
    """Main function for the Basic Script Driver.

    This function is the entry point for the Basic Script Driver.

    Execution Lifecycle:
    1. Read the source code and hyperparameters JSON files.
    2. Set hyperparameters as command line arguments.
    3. Create the commands to execute.
    4. Execute the commands.
    """

    cmd = create_commands()

    logger.info(f"Executing command: {' '.join(cmd)}")
    exit_code, traceback = execute_commands(cmd)
    if exit_code != 0:
        write_failure_file(traceback)
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
