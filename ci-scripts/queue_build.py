# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from __future__ import absolute_import

import os
import time
import boto3

account = boto3.client("sts").get_caller_identity()["Account"]
bucket_name = "sagemaker-us-west-2-%s" % account


def queue_build():
    build_id = os.environ.get("CODEBUILD_BUILD_ID", "CODEBUILD-BUILD-ID")
    source_version = os.environ.get("CODEBUILD_SOURCE_VERSION", "CODEBUILD-SOURCE-VERSION").replace(
        "/", "-"
    )
    ticket_number = int(1000 * time.time())
    filename = "%s_%s_%s" % (ticket_number, build_id, source_version)

    print("Created queue ticket %s" % ticket_number)

    _write_ticket(filename)
    files = _list_tickets()
    _cleanup_tickets_older_than_8_hours(files)
    _wait_for_other_builds(files, ticket_number)


def _build_info_from_file(file):
    filename = file.key.split("/")[1]
    ticket_number, build_id, source_version = filename.split("_")
    return int(ticket_number), build_id, source_version


def _wait_for_other_builds(files, ticket_number):
    newfiles = list(filter(lambda file: not _file_older_than(file), files))
    sorted_files = list(sorted(newfiles, key=lambda y: y.key))

    print("build queue status:")
    print()

    for order, file in enumerate(sorted_files):
        file_ticket_number, build_id, source_version = _build_info_from_file(file)
        print(
            "%s -> %s %s, ticket number: %s" % (order, build_id, source_version, file_ticket_number)
        )

    for file in sorted_files:
        file_ticket_number, build_id, source_version = _build_info_from_file(file)

        if file_ticket_number == ticket_number:

            break
        else:
            while True:
                client = boto3.client("codebuild")
                response = client.batch_get_builds(ids=[build_id])
                build_status = response["builds"][0]["buildStatus"]

                if build_status == "IN_PROGRESS":
                    print(
                        "waiting on build %s %s %s" % (build_id, source_version, file_ticket_number)
                    )
                    time.sleep(30)
                else:
                    print("build %s finished, deleting lock" % build_id)
                    file.delete()
                    break


def _cleanup_tickets_older_than_8_hours(files):
    oldfiles = list(filter(_file_older_than, files))
    for file in oldfiles:
        print("object %s older than 8 hours. Deleting" % file.key)
        file.delete()
    return files


def _list_tickets():
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)
    objects = [file for file in bucket.objects.filter(Prefix="ci-lock/")]
    files = list(filter(lambda x: x != "ci-lock/", objects))
    return files


def _file_older_than(file):
    timelimit = 1000 * 60 * 60 * 8

    file_ticket_number, build_id, source_version = _build_info_from_file(file)

    return int(time.time()) - file_ticket_number > timelimit


def _write_ticket(ticket_number):

    if not os.path.exists("ci-lock"):
        os.mkdir("ci-lock")

    filename = "ci-lock/" + ticket_number
    with open(filename, "w") as file:
        file.write(ticket_number)
    boto3.Session().resource("s3").Object(bucket_name, filename).upload_file(filename)


if __name__ == "__main__":
    queue_build()
