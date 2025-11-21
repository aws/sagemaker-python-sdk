#!/usr/bin/env bash
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

set -euo pipefail

echo =================== $1 execution time ===================

start_time=$2
end_time=`date +%s`
total_time=$(expr $end_time - $start_time + 1)
hours=$((total_time/60/60%24))
minutes=$((total_time/60%60))
secs=$((total_time%60))

(( $hours > 0 )) && printf '%d hours ' $hours
(( $minutes > 0 )) && printf '%d minutes ' $minutes
(( $hours > 0 || $minutes > 0 )) && printf 'and '
printf '%d seconds\n\n' $secs
