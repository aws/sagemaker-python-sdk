import botocore

from boto3.session import Session
import botocore.client
from botocore.config import Config
from typing import Generator, Tuple, List
from sagemaker.core.utils.utils import SingletonMeta


class CloudWatchLogsClient(metaclass=SingletonMeta):
    """
    A singleton class for creating a CloudWatchLogs client.
    """

    client: botocore.client = None

    def __init__(self):
        if not self.client:
            session = Session()
            self.client = session.client(
                "logs",
                session.region_name,
                config=Config(retries={"max_attempts": 10, "mode": "standard"}),
            )


class LogStreamHandler:
    log_group_name: str = None
    log_stream_name: str = None
    stream_id: int = None
    next_token: str = None
    cw_client = None

    def __init__(self, log_group_name: str, log_stream_name: str, stream_id: int):
        self.log_group_name = log_group_name
        self.log_stream_name = log_stream_name
        self.cw_client = CloudWatchLogsClient().client
        self.stream_id = stream_id

    def get_latest_log_events(self) -> Generator[Tuple[str, dict], None, None]:
        """
        This method gets all the latest log events for this stream that exist at this moment in time.

        cw_client.get_log_events() always returns a nextForwardToken even if the current batch of events is empty.
        You can keep calling cw_client.get_log_events() with the same token until a new batch of log events exist.

        API Reference: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/logs/client/get_log_events.html

        Returns:
            Generator[tuple[str, dict], None, None]: Generator that yields a tuple that consists for two values
                str: stream_name,
                dict: event dict in format
                    {
                        "ingestionTime": number,
                        "message": "string",
                        "timestamp": number
                    }
        """
        while True:
            if not self.next_token:
                token_args = {}
            else:
                token_args = {"nextToken": self.next_token}

            response = self.cw_client.get_log_events(
                logGroupName=self.log_group_name,
                logStreamName=self.log_stream_name,
                startFromHead=True,
                **token_args,
            )

            self.next_token = response["nextForwardToken"]
            if not response["events"]:
                break

            for event in response["events"]:
                yield self.log_stream_name, event


class MultiLogStreamHandler:
    log_group_name: str = None
    log_stream_name_prefix: str = None
    expected_stream_count: int = None
    streams: List[LogStreamHandler] = []
    cw_client = None

    def __init__(
        self, log_group_name: str, log_stream_name_prefix: str, expected_stream_count: int
    ):
        self.log_group_name = log_group_name
        self.log_stream_name_prefix = log_stream_name_prefix
        self.expected_stream_count = expected_stream_count
        self.cw_client = CloudWatchLogsClient().client

    def get_latest_log_events(self) -> Generator[Tuple[str, dict], None, None]:
        """
        This method gets all the latest log events from each stream that exist at this moment.

        Returns:
             Generator[tuple[str, dict], None, None]: Generator that yields a tuple that consists for two values
                str: stream_name,
                dict: event dict in format -
                    {
                        "ingestionTime": number,
                        "message": "string",
                        "timestamp": number
                    }
        """
        if not self.ready():
            return []

        for stream in self.streams:
            yield from stream.get_latest_log_events()

    def ready(self) -> bool:
        """
        Checks whether or not MultiLogStreamHandler is ready to serve new log events at this moment.

        If self.streams is already set, return True.
        Otherwise, check if the current number of log streams in the log group match the exptected stream count.

        Returns:
            bool: Whether or not MultiLogStreamHandler is ready to serve new log events.
        """

        if len(self.streams) >= self.expected_stream_count:
            return True

        try:
            response = self.cw_client.describe_log_streams(
                logGroupName=self.log_group_name,
                logStreamNamePrefix=self.log_stream_name_prefix + "/",
                orderBy="LogStreamName",
            )
            stream_names = [stream["logStreamName"] for stream in response["logStreams"]]

            next_token = response.get("nextToken")
            while next_token:
                response = self.cw_client.describe_log_streams(
                    logGroupName=self.log_group_name,
                    logStreamNamePrefix=self.log_stream_name_prefix + "/",
                    orderBy="LogStreamName",
                    nextToken=next_token,
                )
                stream_names.extend([stream["logStreamName"] for stream in response["logStreams"]])
                next_token = response.get("nextToken", None)

            if len(stream_names) >= self.expected_stream_count:
                self.streams = [
                    LogStreamHandler(self.log_group_name, log_stream_name, index)
                    for index, log_stream_name in enumerate(stream_names)
                ]

                return True
            else:
                # Log streams are created whenever a container starts writing to stdout/err,
                # so if the stream count is less than the expected number, return False
                return False

        except botocore.exceptions.ClientError as e:
            # On the very first training job run on an account, there's no log group until
            # the container starts logging, so ignore any errors thrown about that
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                return False
            else:
                raise
