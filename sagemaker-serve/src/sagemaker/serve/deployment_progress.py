"""Rich-based live progress display for SageMaker deployment operations."""

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.status import Status
from rich.console import Group
from rich.live import Live
from rich.style import Style


class EndpointDeploymentProgress:
    """Rich console progress interface matching ModelTrainer design"""

    def __init__(self, endpoint_name: str):
        self.endpoint_name = endpoint_name
        self.console = Console()
        self.current_status = "Creating"
        self.live = None

        # Create progress bar with timer (like ModelTrainer)
        self.progress = Progress(
            SpinnerColumn("bouncingBar"),
            TextColumn("{task.description}"),
            TimeElapsedColumn(),
        )
        self.progress.add_task("Waiting for Endpoint...")

        # Create status display
        self.status = Status("Current status: Creating")

    def __enter__(self):
        """Start the live progress display and return self."""
        panel = Panel(
            Group(self.progress, self.status),
            title="Wait Log Panel",
            border_style=Style(color="blue"),
        )
        # Use the same console with frequent refresh for animations and timer
        self.live = Live(panel, console=self.console, refresh_per_second=4)
        self.live.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the live progress display on context exit."""
        if self.live:
            self.live.stop()

    def log(self, message: str):
        """Log a message above the progress bar"""
        self.console.print(message)

    def update_status(self, status: str):
        """Update the deployment status"""
        self.current_status = status
        if self.status:
            self.status.update(f"Current status: [bold]{status}")


def _deploy_done_with_progress(sagemaker_client, endpoint_name, progress_tracker=None):
    """Enhanced deployment checker with rich progress support"""
    in_progress_statuses = ["Creating", "Updating"]

    desc = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
    status = desc["EndpointStatus"]

    if progress_tracker:
        progress_tracker.update_status(status)
    else:
        # Fallback to original dots
        print("-" if status in in_progress_statuses else "!", end="", flush=True)

    return None if status in in_progress_statuses else desc


def _live_logging_deploy_done_with_progress(
    sagemaker_client,
    endpoint_name,
    paginator,
    paginator_config,
    poll,
    progress_tracker=None,
    not_found_budget=None,
):
    """Live logging deployment checker that routes logs to Rich progress tracker.

    Returns the ``DescribeEndpoint`` response once the endpoint is no longer
    ``Creating``/``Updating``, and ``None`` while it still is. The endpoint log
    group only exists once a container has started, so an endpoint that fails
    before any instance is provisioned (for example on InsufficientInstanceCapacity)
    never gets one: a missing log group must not keep a finished deployment waiting.
    ``not_found_budget`` optionally bounds how many consecutive polls a missing
    endpoint is tolerated.
    """
    import time
    from botocore.exceptions import ClientError

    in_progress_statuses = ["Creating", "Updating"]
    try:
        desc = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_status = desc["EndpointStatus"]
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            if not_found_budget is not None:
                not_found_budget.miss(e)
            return None
        raise e
    if not_found_budget is not None:
        not_found_budget.reset()

    # Update status and check if we should stop
    stop = endpoint_status not in in_progress_statuses
    if stop:
        if endpoint_status == "InService" and progress_tracker:
            progress_tracker.log(f"Created endpoint with name {endpoint_name}")
        elif endpoint_status != "InService":
            time.sleep(poll)

    try:
        # Fetch and route CloudWatch logs to progress tracker
        pages = paginator.paginate(
            logGroupName=f"/aws/sagemaker/Endpoints/{endpoint_name}",
            logStreamNamePrefix="AllTraffic/",
            PaginationConfig=paginator_config,
        )

        for page in pages:
            if "nextToken" in page:
                paginator_config["StartingToken"] = page["nextToken"]
                for event in page["events"]:
                    if progress_tracker:
                        progress_tracker.log(event["message"])
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise e

    # Update progress tracker status
    if progress_tracker:
        progress_tracker.update_status(endpoint_status)

    # Return desc if we should stop polling
    return desc if stop else None
