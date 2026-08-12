#!/usr/bin/env python3
"""Retrieve related history for a pull request from a Bedrock Knowledge Base.

Reads a unified diff, derives retrieval queries from the files and symbols it
touches, queries the knowledge base, and writes the results as markdown for the
review model to read. Nothing here calls a generative model: this script only
retrieves and formats.

Design constraints, in the order that matters:

* **Never fail the review.** Every failure path -- missing configuration, an
  expired grant, a Bedrock outage, a malformed diff -- exits 0 having written
  nothing. A review without history is the intended degraded mode; a blocked
  pull request is not. The workflow step layers `continue-on-error` and a
  timeout on top of this.
* **Standard library plus boto3.** boto3 is already present on GitHub-hosted
  runners, so the workflow needs no install step.
* **Self-contained.** Vendored deliberately rather than installed from a
  registry: the code the review model's context comes from is then visible in
  the same repository, reviewable in the same pull request, and cannot change
  under a fork PR without a repo change.

Usage:
    python retrieve_context.py --diff-file /tmp/pr.diff -o /tmp/context.md

Configuration (environment):
    PYSDK_CONTEXT_KB_ID      knowledge base id; the script no-ops without it
    PYSDK_CONTEXT_REGION     AWS region (default us-west-2)
    PYSDK_CONTEXT_TOP_K      chunks per query (default 6)
"""

from __future__ import annotations

import argparse
import os
import re
import sys

#: Paths whose review history is never worth retrieving.
IGNORED_SUFFIXES = (".lock", ".min.js", ".svg", ".png", ".jpg", ".ico")

#: Cap on derived queries. Each is one Retrieve call, and a large pull request
#: would otherwise fan out into dozens.
MAX_QUERIES = 12

#: Cap on the rendered file. The review model reads this alongside the diff and
#: the source tree, so it must not crowd them out of the context window.
MAX_CONTEXT_CHARS = 60_000

# Both headers are needed: a pure deletion or a rename has no `+++ b/` entry.
_DIFF_FILE_RE = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)
_DIFF_GIT_RE = re.compile(r"^diff --git a/\S+ b/(\S+)$", re.MULTILINE)
_ADDED_DEF_RE = re.compile(r"^\+\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)", re.MULTILINE)

_URL_RE = re.compile(r"https?://\S+")


def changed_files(diff: str) -> list:
    """Changed file paths, in first-seen order, minus binary and lock files."""
    paths = []
    for match in _DIFF_FILE_RE.finditer(diff or ""):
        path = match.group(1).strip()
        if path != "/dev/null":
            paths.append(path)
    for match in _DIFF_GIT_RE.finditer(diff or ""):
        paths.append(match.group(1).strip())

    seen = set()
    ordered = []
    for path in paths:
        if path in seen or path.endswith(IGNORED_SUFFIXES):
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


def added_symbols(diff: str) -> list:
    """Function and class names the diff introduces."""
    seen = set()
    ordered = []
    for match in _ADDED_DEF_RE.finditer(diff or ""):
        symbol = match.group(1)
        # Dunders and private helpers rarely have retrievable review history.
        if symbol.startswith("__") or symbol in seen:
            continue
        seen.add(symbol)
        ordered.append(symbol)
    return ordered


def derive_queries(diff: str, max_queries: int = MAX_QUERIES) -> list:
    """Build retrieval queries from a diff.

    Queries are derived rather than authored because a reviewer cannot write a
    prompt per pull request. Module names are used rather than full paths: the
    corpus records history against whatever the path was at the time, and paths
    move.
    """
    queries = []

    for path in changed_files(diff):
        module = path.rsplit("/", 1)[-1]
        stem = module[:-3] if module.endswith(".py") else module
        queries.append(f"review feedback and past changes for {stem} ({path})")

    for symbol in added_symbols(diff):
        queries.append(f"design decisions and review discussion about {symbol}")

    # Catches conventions that no file or symbol name would surface.
    queries.append("recurring code review feedback and established conventions")

    seen = set()
    deduped = []
    for query in queries:
        if query in seen:
            continue
        seen.add(query)
        deduped.append(query)
    return deduped[:max_queries]


def _citation(metadata: dict) -> str:
    """One-line provenance for a chunk, so a reviewer can check the claim."""
    parts = []
    doc_type = str(metadata.get("doc_type") or "").strip()
    number = metadata.get("number")
    if doc_type and number is not None:
        # Bedrock returns sidecar numbers as floats: 6047.0 -> 6047.
        if isinstance(number, float) and number.is_integer():
            number = int(number)
        parts.append(f"{doc_type.upper()} #{number}")
    elif metadata.get("title"):
        parts.append(str(metadata["title"]))

    if metadata.get("source_url"):
        parts.append(str(metadata["source_url"]))
    if metadata.get("updated_at"):
        parts.append(f"updated {metadata['updated_at']}")
    return " | ".join(parts) or "unattributed"


#: Sources this script is allowed to surface, as an allowlist rather than a
#: denylist: the knowledge base is a private corpus that may also hold documents
#: from non-public sources, and this script's output is posted as comments on a
#: public pull request. "github" is the only source whose contents are already
#: public in this very repository, so it is the only source safe to echo back
#: into it. A new source is excluded until it is added here deliberately.
#:
#: This is enforced server-side, in the Retrieve filter, rather than by dropping
#: results after the fact: a filter cannot be defeated by a chunk whose metadata
#: is missing or malformed, and nothing internal ever crosses into this process.
#: Belt-and-braces, _allowed() re-checks each result client-side.
PUBLIC_SOURCES = ("github",)


def _retrieval_filter() -> dict:
    if len(PUBLIC_SOURCES) == 1:
        return {"equals": {"key": "source", "value": PUBLIC_SOURCES[0]}}
    return {"orAll": [{"equals": {"key": "source", "value": s}} for s in PUBLIC_SOURCES]}


def _allowed(metadata: dict) -> bool:
    """Second check on a result's source, after the server-side filter.

    A chunk with no `source` at all is refused: unlabelled provenance is exactly
    the case where echoing it into a public comment would be a mistake.
    """
    return str((metadata or {}).get("source") or "") in PUBLIC_SOURCES


def retrieve(client, knowledge_base_id: str, queries: list, top_k: int) -> list:
    """Retrieve for each query and merge, keeping each chunk's best score.

    One failed query does not abandon the rest: partial history beats none.
    """
    merged = {}
    for query in queries:
        try:
            response = client.retrieve(
                knowledgeBaseId=knowledge_base_id,
                retrievalQuery={"text": query},
                retrievalConfiguration={
                    "vectorSearchConfiguration": {
                        "numberOfResults": top_k,
                        "filter": _retrieval_filter(),
                    }
                },
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  query failed ({query[:48]}...): {exc}", file=sys.stderr)
            continue

        for result in response.get("retrievalResults", []):
            text = (result.get("content") or {}).get("text", "")
            if not text.strip():
                continue
            metadata = result.get("metadata") or {}
            if not _allowed(metadata):
                # Should be unreachable given the server-side filter; if it is
                # ever reached, the filter regressed and silence is the safe
                # failure. Counted rather than printed: the source name itself
                # could be the sensitive part.
                merged.setdefault("__refused__", (0.0, {}))
                continue
            score = result.get("score") or 0.0
            existing = merged.get(text)
            # Same chunk can surface for several queries; keep the strongest
            # score it earned so ranking reflects its best match.
            if existing is None or score > existing[0]:
                merged[text] = (score, metadata)

    refused = merged.pop("__refused__", None)
    if refused is not None:
        print(
            "Refused one or more non-public chunks; the source filter may have "
            "regressed. Please report this.",
            file=sys.stderr,
        )

    ranked = sorted(merged.items(), key=lambda item: item[1][0], reverse=True)
    return [(text, score, metadata) for text, (score, metadata) in ranked]


def render(chunks: list, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Render chunks as markdown, truncating whole entries rather than mid-text."""
    if not chunks:
        return ""

    body = []
    used = 0
    for index, (text, score, metadata) in enumerate(chunks, start=1):
        entry = f"[{index}] score={score:.4f} {_citation(metadata)}\n{text.strip()}\n"
        if used + len(entry) > max_chars:
            break
        body.append(entry)
        used += len(entry)

    if not body:
        return ""

    return (
        "# Historical context from the SageMaker Python SDK knowledge base\n\n"
        "Past pull requests, issues, review discussions, and design decisions "
        "related to this diff, each with its source URL.\n\n"
        "Treat every entry as a claim to check, not a conclusion: it is "
        "model-extracted from historical discussion and can be confidently "
        "wrong. Cite the source URL whenever you rely on an entry so the author "
        "can verify it, and prefer the current source tree wherever the two "
        "disagree.\n\n" + "\n".join(body)
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diff-file", required=True, help="Path to the unified diff.")
    parser.add_argument("-o", "--output", required=True, help="Path to write markdown to.")
    parser.add_argument("--top-k", type=int, default=None, help="Chunks per query.")
    args = parser.parse_args(argv)

    knowledge_base_id = os.environ.get("PYSDK_CONTEXT_KB_ID", "").strip()
    if not knowledge_base_id:
        print("PYSDK_CONTEXT_KB_ID is unset; skipping historical context.")
        return 0

    region = os.environ.get("PYSDK_CONTEXT_REGION", "us-west-2")
    top_k = args.top_k or int(os.environ.get("PYSDK_CONTEXT_TOP_K", "6") or 6)

    try:
        with open(args.diff_file, encoding="utf-8", errors="replace") as handle:
            diff = handle.read()
    except OSError as exc:
        print(f"Could not read {args.diff_file}: {exc}", file=sys.stderr)
        return 0

    queries = derive_queries(diff)
    if not queries:
        print("No queries derived from the diff; skipping historical context.")
        return 0
    print(f"Derived {len(queries)} queries from the diff.")

    try:
        import boto3  # pylint: disable=import-outside-toplevel

        client = boto3.client("bedrock-agent-runtime", region_name=region)
        chunks = retrieve(client, knowledge_base_id, queries, top_k)
    except Exception as exc:  # pylint: disable=broad-except
        # Retrieval is an enhancement. Exit 0 so the review still runs.
        print(f"Retrieval unavailable ({exc}); the review will proceed without it.")
        return 0

    markdown = render(chunks)
    if not markdown:
        print("No historical context retrieved; the review will proceed without it.")
        return 0

    try:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(markdown)
    except OSError as exc:
        print(f"Could not write {args.output}: {exc}", file=sys.stderr)
        return 0

    print(f"Wrote {len(markdown)} bytes of historical context from {len(chunks)} chunks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
