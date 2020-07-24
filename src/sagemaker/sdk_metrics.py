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

from datetime import datetime


class SDKMetrics(object):
    def __init__(self, module, function=None):
        self.module = module
        self.function = function
        self.timestamp = datetime.now()

    def get_metrics_string(self):
        metrics = "MODULE/{}, DURATION/{}".format(
            self.module,
            (datetime.now() - self.timestamp).microseconds
        )

        if self.function:
            metrics = "FUNCTION/{} {}".format(self.function, metrics)

        return metrics
