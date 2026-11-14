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
"""BYOC fixture: customer-supplied transformation code for the DT integ suite.

Uploaded to S3 by the ``s3_fixtures`` fixture and passed as
``TransformationCodeS3Uri`` to the BYOC operator. It is not imported by the
test process — the service downloads it and executes it inside a
RestrictedPython sandbox, so it must stay within that sandbox's contract:

* Define ``transform_dataset(input_path, output_path) -> dict``. The sandbox
  looks that name up by string and raises KeyError if it is absent.
* Return a stats dict; the service logs it and treats ``None`` as ``{}``.
* Only the stdlib allowlist is importable (``json`` is on it). Writes are
  permitted under ``/tmp`` only, which is where the service stages both
  paths.
* No leading-underscore names: RestrictedPython rejects them, and the data
  plane's source sanitizer would silently rename them.

Behavior: keep only rows whose ``completion`` is exactly ``"positive"``.
"""

import json


def transform_dataset(input_path, output_path):
    """Filter the JSONL at input_path down to positive-sentiment rows."""
    total = 0
    kept = 0
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            stripped = line.strip()
            if not stripped:
                continue
            total = total + 1
            row = json.loads(stripped)
            if row.get("completion") == "positive":
                fout.write(json.dumps(row) + "\n")
                kept = kept + 1
    return {"total_records": total, "transformed_records": kept, "errors": 0}
