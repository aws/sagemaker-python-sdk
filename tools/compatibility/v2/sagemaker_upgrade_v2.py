# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""A tool to upgrade SageMaker Python SDK code to be compatible with v2."""
from __future__ import absolute_import

import argparse

import files


def _parse_and_validate_args():
    """Parses CLI arguments"""
    parser = argparse.ArgumentParser(
        description="A tool to convert files to be compatible with v2 of the SageMaker Python SDK."
        "\nSimple usage: sagemaker_upgrade_v2.py --in-file foo.py --out-file bar.py"
    )
    parser.add_argument(
        "--in-file", help="If converting a single file, the name of the file to convert"
    )
    parser.add_argument(
        "--out-file",
        help="If converting a single file, the output file destination. If needed, "
        "directories in the output file path are created. If the output file already exists, "
        "it is overwritten.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_and_validate_args()

    files.PyFileUpdater(input_path=args.in_file, output_path=args.out_file).update()
