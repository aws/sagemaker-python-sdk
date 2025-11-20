import re
import logging
from typing import List, Iterator, Optional
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.resources import HubContent

logger = logging.getLogger(__name__)


class _Filter:
    """
    A filter that evaluates logical expressions against a list of keyword strings.

    Supports logical operators (AND, OR, NOT), parentheses for grouping, and wildcard patterns
    (e.g., `text-*`, `*ai`, `@task:foo`).

    Example:
        filt = _Filter("(@framework:huggingface OR text-*) AND NOT deprecated")
        filt.match(["@framework:huggingface", "text-generation"])  # Returns True
    """

    def __init__(self, expression: str) -> None:
        """
        Initialize the filter with a string expression.

        Args:
            expression (str): A logical expression to evaluate against keywords.
                Supports AND, OR, NOT, parentheses, and wildcard patterns (*).
        """
        self.expression: str = expression

    def match(self, keywords: List[str]) -> bool:
        """
        Evaluate the filter expression against a list of keywords.

        Args:
            keywords (List[str]): A list of keyword strings to test.

        Returns:
            bool: True if the expression evaluates to True for the given keywords, else False.
        """
        expr: str = self._convert_expression(self.expression)
        try:
            return eval(expr, {"__builtins__": {}}, {"keywords": keywords, "any": any})
        except Exception:
            return False

    def _convert_expression(self, expr: str) -> str:
        """
        Convert the logical filter expression into a Python-evaluable string.

        Args:
            expr (str): The raw expression to convert.

        Returns:
            str: A Python expression string using 'any' and logical operators.
        """
        tokens: List[str] = re.findall(
            r"\bAND\b|\bOR\b|\bNOT\b|[^\s()]+|\(|\)", expr, flags=re.IGNORECASE
        )

        def wildcard_condition(pattern: str) -> str:
            pattern = pattern.strip('"').strip("'")
            stripped = pattern.strip("*")

            if pattern.startswith("*") and pattern.endswith("*"):
                return f"{repr(stripped)} in k"
            elif pattern.startswith("*"):
                return f"k.endswith({repr(stripped)})"
            elif pattern.endswith("*"):
                return f"k.startswith({repr(stripped)})"
            else:
                return f"k == {repr(pattern)}"

        def convert_token(token: str) -> str:
            upper = token.upper()
            if upper == "AND":
                return "and"
            elif upper == "OR":
                return "or"
            elif upper == "NOT":
                return "not"
            elif token in ("(", ")"):
                return token
            else:
                return f"any({wildcard_condition(token)} for k in keywords)"

        converted_tokens = [convert_token(tok) for tok in tokens]
        return " ".join(converted_tokens)


def _list_all_hub_models(hub_name: str, sm_client: Session) -> Iterator[HubContent]:
    """
    Retrieve all model entries from the specified hub and yield them one by one.

    This function paginates through the SageMaker Hub API to retrieve all published models of type "Model"
    and yields them as `HubContent` objects.

    Args:
        hub_name (str): The name of the hub to query.
        sm_client (Session): The SageMaker session.

    Yields:
        HubContent: A `HubContent` object representing a single model entry from the hub.
    """
    next_token = None

    while True:
        # Prepare the request parameters
        params = {"HubName": hub_name, "HubContentType": "Model", "MaxResults": 100}

        # Add NextToken if it exists
        if next_token:
            params["NextToken"] = next_token

        # Make the API call
        response = sm_client.list_hub_contents(**params)

        # Yield each content summary
        for content in response["HubContentSummaries"]:
            yield HubContent(
                hub_name=hub_name,
                hub_content_arn=content["HubContentArn"],
                hub_content_type="Model",
                hub_content_name=content["HubContentName"],
                hub_content_version=content["HubContentVersion"],
                hub_content_description=content.get("HubContentDescription", ""),
                hub_content_search_keywords=content.get("HubContentSearchKeywords", []),
            )

        # Check if there are more results
        next_token = response.get("NextToken", None)
        if not next_token or len(response["HubContentSummaries"]) == 0:
            break  # Exit the loop if there are no more pages


def search_public_hub_models(
    query: str,
    hub_name: Optional[str] = "SageMakerPublicHub",
    sagemaker_session: Optional[Session] = None,
) -> List[HubContent]:
    """
    Search and filter models from hub using a keyword expression.

    Args:
        query (str): A logical expression used to filter models by keywords.
            Example: "@task:text-generation AND NOT @framework:legacy"
        hub_name (Optional[str]): The name of the hub to query. Defaults to "SageMakerPublicHub".
        sagemaker_session (Optional[Session]): An optional SageMaker `Session` object. If not provided,
            a default session will be created and a warning will be logged.

    Returns:
        List[HubContent]: A list of filtered `HubContent` model objects that match the query.
    """
    if sagemaker_session is None:
        sagemaker_session = Session()
        logger.warning("SageMaker session not provided. Using default Session.")
    sm_client = sagemaker_session.sagemaker_client

    models = _list_all_hub_models(hub_name, sm_client)
    filt = _Filter(query)
    results: List[HubContent] = []

    for model in models:
        keywords = model.hub_content_search_keywords
        normalized_keywords = [kw.replace(" ", "-") for kw in keywords]

        if filt.match(normalized_keywords):
            results.append(model)

    return results
