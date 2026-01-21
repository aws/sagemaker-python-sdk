import re
import logging
from typing import List, Iterator, Optional
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.resources import HubContent

logger = logging.getLogger(__name__)


class _ExpressionNode:
    """Base class for expression AST nodes."""

    def evaluate(self, keywords: List[str]) -> bool:
        """Evaluate this node against the given keywords."""
        raise NotImplementedError


class _AndNode(_ExpressionNode):
    """AND logical operator node."""

    def __init__(self, left: _ExpressionNode, right: _ExpressionNode):
        self.left = left
        self.right = right

    def evaluate(self, keywords: List[str]) -> bool:
        return self.left.evaluate(keywords) and self.right.evaluate(keywords)


class _OrNode(_ExpressionNode):
    """OR logical operator node."""

    def __init__(self, left: _ExpressionNode, right: _ExpressionNode):
        self.left = left
        self.right = right

    def evaluate(self, keywords: List[str]) -> bool:
        return self.left.evaluate(keywords) or self.right.evaluate(keywords)


class _NotNode(_ExpressionNode):
    """NOT logical operator node."""

    def __init__(self, operand: _ExpressionNode):
        self.operand = operand

    def evaluate(self, keywords: List[str]) -> bool:
        return not self.operand.evaluate(keywords)


class _PatternNode(_ExpressionNode):
    """Pattern matching node for keywords with wildcard support."""

    def __init__(self, pattern: str):
        self.pattern = pattern.strip('"').strip("'")

    def evaluate(self, keywords: List[str]) -> bool:
        """Check if any keyword matches this pattern."""
        for keyword in keywords:
            if self._matches_pattern(keyword, self.pattern):
                return True
        return False

    def _matches_pattern(self, keyword: str, pattern: str) -> bool:
        """Check if a keyword matches a pattern with wildcard support."""
        if pattern.startswith("*") and pattern.endswith("*"):
            # Contains pattern: *text*
            stripped = pattern.strip("*")
            return stripped in keyword
        elif pattern.startswith("*"):
            # Ends with pattern: *text
            stripped = pattern[1:]
            return keyword.endswith(stripped)
        elif pattern.endswith("*"):
            # Starts with pattern: text*
            stripped = pattern[:-1]
            return keyword.startswith(stripped)
        else:
            # Exact match
            return keyword == pattern


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
        self._ast: Optional[_ExpressionNode] = None

    def match(self, keywords: List[str]) -> bool:
        """
        Evaluate the filter expression against a list of keywords.

        Args:
            keywords (List[str]): A list of keyword strings to test.

        Returns:
            bool: True if the expression evaluates to True for the given keywords, else False.
        """
        try:
            if self._ast is None:
                self._ast = self._parse_expression(self.expression)
            return self._ast.evaluate(keywords)
        except Exception:
            return False

    def _parse_expression(self, expr: str) -> _ExpressionNode:
        """
        Parse the logical filter expression into an AST.

        Args:
            expr (str): The raw expression to parse.

        Returns:
            _ExpressionNode: Root node of the parsed expression AST.
        """
        tokens = self._tokenize(expr)
        result, _ = self._parse_or_expression(tokens, 0)
        return result

    def _tokenize(self, expr: str) -> List[str]:
        """Tokenize the expression into logical operators, keywords, and parentheses."""
        return re.findall(r"\bAND\b|\bOR\b|\bNOT\b|[^\s()]+|\(|\)", expr, flags=re.IGNORECASE)

    def _parse_or_expression(self, tokens: List[str], pos: int) -> tuple[_ExpressionNode, int]:
        """Parse OR expression (lowest precedence)."""
        left, pos = self._parse_and_expression(tokens, pos)

        while pos < len(tokens) and tokens[pos].upper() == "OR":
            pos += 1  # Skip OR token
            right, pos = self._parse_and_expression(tokens, pos)
            left = _OrNode(left, right)

        return left, pos

    def _parse_and_expression(self, tokens: List[str], pos: int) -> tuple[_ExpressionNode, int]:
        """Parse AND expression (medium precedence)."""
        left, pos = self._parse_not_expression(tokens, pos)

        while pos < len(tokens) and tokens[pos].upper() == "AND":
            pos += 1  # Skip AND token
            right, pos = self._parse_not_expression(tokens, pos)
            left = _AndNode(left, right)

        return left, pos

    def _parse_not_expression(self, tokens: List[str], pos: int) -> tuple[_ExpressionNode, int]:
        """Parse NOT expression (highest precedence)."""
        if pos < len(tokens) and tokens[pos].upper() == "NOT":
            pos += 1  # Skip NOT token
            operand, pos = self._parse_primary_expression(tokens, pos)
            return _NotNode(operand), pos
        else:
            return self._parse_primary_expression(tokens, pos)

    def _parse_primary_expression(self, tokens: List[str], pos: int) -> tuple[_ExpressionNode, int]:
        """Parse primary expression (parentheses or pattern)."""
        if pos >= len(tokens):
            raise ValueError("Unexpected end of expression")

        token = tokens[pos]

        if token == "(":
            pos += 1  # Skip opening parenthesis
            expr, pos = self._parse_or_expression(tokens, pos)
            if pos >= len(tokens) or tokens[pos] != ")":
                raise ValueError("Missing closing parenthesis")
            pos += 1  # Skip closing parenthesis
            return expr, pos
        elif token == ")":
            raise ValueError("Unexpected closing parenthesis")
        else:
            # Pattern token
            return _PatternNode(token), pos + 1

    def _convert_expression(self, expr: str) -> str:
        """
        Legacy method for backward compatibility.
        This method is no longer used but kept to avoid breaking changes.
        """
        # This method is deprecated and should not be used
        # It's kept only for backward compatibility
        return expr


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
