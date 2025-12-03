"""
Utility functions for displaying evaluation results.
Supports both Benchmark and LLM As Judge evaluation types.
"""

import logging
import json
import boto3
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


# ============================================================================
# Common Utilities
# ============================================================================

def _extract_training_job_name_from_steps(pipeline_execution, step_name_pattern: str = 'Evaluate') -> Optional[str]:
    """Extract training job name from pipeline step metadata
    
    For LLMAJ evaluations, prioritizes EvaluateCustomModelMetrics over EvaluateBaseModelMetrics.
    For other evaluations, searches for steps matching the pattern.
    
    Args:
        pipeline_execution: EvaluationPipelineExecution instance
        step_name_pattern: Pattern to match in step name (default: 'Evaluate')
    
    Returns:
        Training job name if found, None otherwise
    """
    if not pipeline_execution._pipeline_execution:
        return None
    
    try:
        # Collect all matching steps
        matching_steps = []
        steps_iterator = pipeline_execution._pipeline_execution.get_all_steps()
        
        for step in steps_iterator:
            step_name = getattr(step, 'step_name', '')
            if step_name_pattern in step_name:
                metadata = getattr(step, 'metadata', None)
                if metadata and hasattr(metadata, 'training_job'):
                    training_job = metadata.training_job
                    if hasattr(training_job, 'arn'):
                        job_name = training_job.arn.split('/')[-1]
                        matching_steps.append((step_name, job_name))
        
        # Priority order: EvaluateCustomModelMetrics > EvaluateBaseModelMetrics > any other match
        for step_name, job_name in matching_steps:
            if 'EvaluateCustomModelMetrics' in step_name:
                logger.info(f"Extracted training job name: {job_name} from step: {step_name} (priority: Custom)")
                return job_name
        
        for step_name, job_name in matching_steps:
            if 'EvaluateBaseModelMetrics' in step_name:
                logger.info(f"Extracted training job name: {job_name} from step: {step_name} (priority: Base)")
                return job_name
        
        # Return first match if no priority steps found
        if matching_steps:
            step_name, job_name = matching_steps[0]
            logger.info(f"Extracted training job name: {job_name} from step: {step_name}")
            return job_name
            
    except Exception as e:
        logger.warning(f"Failed to extract training job name: {e}")
    
    return None


# ============================================================================
# Benchmark Evaluation Results Display
# ============================================================================

def _extract_metrics_from_results(results_dict: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract metrics from results dictionary.
    
    Tries to get metrics from results["all"] first (standard case for benchmarks like MMLU).
    Falls back to finding metrics in nested keys like "custom|gen_qa_gen_qa|0" (gen_qa case).
    
    Args:
        results_dict: Full results dictionary from results_*.json
        
    Returns:
        Dict of metric names to values
    """
    results = results_dict.get('results', {})
    
    # Try standard "all" key first (used by standard benchmarks like MMLU, BBH, etc.)
    if 'all' in results and results['all']:
        logger.info("Using metrics from 'all' key (standard benchmark format)")
        return results['all']
    
    # Fallback: Look for task-specific keys (gen_qa and custom_scorer case)
    # Pattern: "custom|task_name_strategy|0" or similar
    for key, value in results.items():
        if isinstance(value, dict) and value:  # Non-empty dict
            logger.info(f"Using metrics from key: '{key}' (gen_qa or custom_scorer format)")
            return value
    
    # No metrics found
    logger.warning("No metrics found in results dictionary")
    return {}


def _show_benchmark_results(pipeline_execution):
    """
    Display benchmark evaluation results by downloading from S3 and showing with Rich tables.
    
    This simplified implementation:
    1. Extracts training job names from pipeline step metadata
    2. Downloads results JSON directly from S3
    3. Parses and extracts only results["all"] metrics
    4. Displays with Rich library (works in both notebook and terminal)
    """
    # Validate we have required information
    if not pipeline_execution.s3_output_path:
        raise ValueError(
            "[PySDK Error] Cannot download results: s3_output_path is not set. "
            "Ensure the evaluation job was started with an s3_output_path."
        )
    
    # Parse S3 location
    s3_path = pipeline_execution.s3_output_path[5:]  # Remove 's3://'
    bucket_name = s3_path.split('/')[0]
    s3_prefix = '/'.join(s3_path.split('/')[1:]).rstrip('/')
    
    logger.info(f"S3 bucket: {bucket_name}, prefix: {s3_prefix}")
    
    # Get S3 client
    s3_client = boto3.client('s3')
    
    # Extract training job names using common utility
    custom_job_name = _extract_training_job_name_from_steps(pipeline_execution, 'EvaluateCustomModel')
    base_job_name = _extract_training_job_name_from_steps(pipeline_execution, 'EvaluateBaseModel')
    
    if not custom_job_name and not base_job_name:
        raise ValueError(
            "[PySDK Error] Could not extract training job name from pipeline steps. "
            "Unable to locate evaluation results."
        )
    
    # Helper to download and parse results JSON
    def download_results_json(training_job_name: str) -> Dict[str, Any]:
        """Download and parse results JSON from S3 by searching recursively"""
        # Search recursively under output/output/ for results_*.json
        search_prefix = f"{s3_prefix}/{training_job_name}/output/output/"
        
        logger.info(f"Searching for results_*.json in s3://{bucket_name}/{search_prefix}")
        
        # List all files recursively (list_objects_v2 returns all keys with this prefix)
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=search_prefix)
        
        if 'Contents' not in response:
            raise FileNotFoundError(
                f"[PySDK Error] No files found at s3://{bucket_name}/{search_prefix}. "
                f"Evaluation results may not have been generated yet."
            )
        
        # Find results_*.json file
        results_file_key = None
        for obj in response['Contents']:
            key = obj['Key']
            # Check if filename matches results_*.json pattern
            filename = key.split('/')[-1]
            if filename.startswith('results_') and filename.endswith('.json'):
                results_file_key = key
                logger.info(f"Found results file: {key}")
                break
        
        if not results_file_key:
            raise FileNotFoundError(
                f"[PySDK Error] No results_*.json file found in s3://{bucket_name}/{search_prefix}. "
                f"Evaluation results may not have been generated yet."
            )
        
        # Download and parse
        obj = s3_client.get_object(Bucket=bucket_name, Key=results_file_key)
        content = obj['Body'].read().decode('utf-8')
        return json.loads(content)
    
    # Download results
    custom_results_full = download_results_json(custom_job_name) if custom_job_name else None
    base_results_full = download_results_json(base_job_name) if base_job_name else None
    
    # Extract metrics (handles both standard benchmark "all" key and gen_qa nested keys)
    custom_metrics = _extract_metrics_from_results(custom_results_full) if custom_results_full else None
    base_metrics = _extract_metrics_from_results(base_results_full) if base_results_full else None
    
    # Construct full S3 paths for display
    s3_paths = {
        'custom': f"s3://{bucket_name}/{s3_prefix}/{custom_job_name}/output/output/None/eval_results/",
        'base': f"s3://{bucket_name}/{s3_prefix}/{base_job_name}/output/output/None/eval_results/" if base_job_name else None
    }
    
    # Display with Rich
    _display_metrics_tables(custom_metrics, base_metrics, s3_paths)


def _display_metrics_tables(custom_metrics: Dict[str, float], 
                            base_metrics: Optional[Dict[str, float]], 
                            s3_paths: Dict[str, Optional[str]]):
    """Display metrics in Rich tables with detailed S3 paths"""
    
    # Detect Jupyter
    is_jupyter = False
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is not None and 'IPKernelApp' in ipython.config:
            is_jupyter = True
    except:
        pass
    
    # Display with Rich
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.box import ROUNDED
    
    console = Console(force_jupyter=is_jupyter) if is_jupyter else Console()
    
    if custom_metrics:
        # Create custom model table
        custom_table = Table(
            show_header=True, 
            header_style="bold green",
            title="[bold green]Custom Model Results[/bold green]",
            box=ROUNDED
        )
        custom_table.add_column("Metric", style="cyan", width=30)
        custom_table.add_column("Value", style="white", justify="right", width=15)
        
        for metric_name, value in sorted(custom_metrics.items()):
            # Format value as percentage if it's between 0 and 1
            if 0 <= value <= 1 and 'stderr' not in metric_name.lower():
                formatted_value = f"{value * 100:.2f}%"
            else:
                formatted_value = f"{value:.4f}"
            custom_table.add_row(metric_name, formatted_value)
        
        console.print(custom_table)
    
    # Create base model table if available
    if base_metrics:
        console.print("")  # Empty line
        base_table = Table(
            show_header=True,
            header_style="bold yellow",
            title="[bold yellow]Base Model Results[/bold yellow]",
            box=ROUNDED
        )
        base_table.add_column("Metric", style="cyan", width=30)
        base_table.add_column("Value", style="white", justify="right", width=15)
        
        for metric_name, value in sorted(base_metrics.items()):
            if 0 <= value <= 1 and 'stderr' not in metric_name.lower():
                formatted_value = f"{value * 100:.2f}%"
            else:
                formatted_value = f"{value:.4f}"
            base_table.add_row(metric_name, formatted_value)
        
        console.print(base_table)
    
    # Add S3 location info
    console.print("")
    message = Text()
    message.append("\n📦 ", style="bold blue")
    message.append("Full evaluation artifacts available at:\n\n", style="bold")
    message.append("Custom Model:\n", style="bold green")
    message.append(f"  {s3_paths['custom']}\n", style="cyan")
    
    if s3_paths.get('base'):
        message.append("\nBase Model:\n", style="bold yellow")
        message.append(f"  {s3_paths['base']}", style="cyan")
    
    console.print(Panel(
        message,
        title="[bold blue]Result Artifacts Location[/bold blue]",
        border_style="blue",
        padding=(1, 2)
    ))


# ============================================================================
# LLM As Judge Evaluation Results Display
# ============================================================================

def _parse_prompt(prompt_str: str) -> str:
    """Parse prompt from format: "[{'role': 'user', 'content': '...'}]" """
    try:
        parsed = json.loads(prompt_str.replace("'", '"'))
        if isinstance(parsed, list) and len(parsed) > 0 and 'content' in parsed[0]:
            return parsed[0]['content']
        return prompt_str
    except Exception:
        return prompt_str


def _parse_response(response_str: str) -> str:
    """Parse response from format: "['response text']" """
    try:
        parsed = json.loads(response_str.replace("'", '"'))
        if isinstance(parsed, list) and len(parsed) > 0:
            return parsed[0]
        return response_str
    except Exception:
        return response_str


def _format_score(score: float) -> str:
    """Format score as percentage: 0.8333 -> '83.3%' """
    return f"{score * 100:.1f}%"


def _truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to max_length with ellipsis if needed"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def _download_llmaj_results_from_s3(pipeline_execution) -> List[Dict[str, Any]]:
    """Download LLM As Judge evaluation results JSONL from S3
    
    Returns:
        List of evaluation result dictionaries (one per JSONL line)
    """
    import os
    
    if not pipeline_execution.s3_output_path:
        raise ValueError(
            "[PySDK Error] Cannot download results: s3_output_path is not set. "
            "Ensure the evaluation job was started with an s3_output_path."
        )
    
    # Parse S3 location
    s3_path = pipeline_execution.s3_output_path[5:]  # Remove 's3://'
    bucket_name = s3_path.split('/')[0]
    s3_prefix = '/'.join(s3_path.split('/')[1:]).rstrip('/')
    
    logger.info(f"S3 bucket: {bucket_name}, prefix: {s3_prefix}")
    
    # Get S3 client (DO NOT use SageMaker endpoint for S3)
    s3_client = boto3.client('s3')
    
    # Extract training job name using common utility
    training_job_name = _extract_training_job_name_from_steps(pipeline_execution, 'Evaluate')
    
    if not training_job_name:
        raise ValueError(
            "[PySDK Error] Could not extract training job name from pipeline steps. "
            "Unable to locate evaluation results."
        )
    
    # Find the JSONL file in S3
    # For LLM As Judge, structure is:
    # s3://bucket/prefix/{training_job}/output/output/{job_name}/eval_results/bedrock_llm_judge_results.json
    # s3://bucket/prefix/{job_name}/{bedrock_job_id}/models/.../output.jsonl
    
    import re
    
    # Search for bedrock summary JSON to extract job name
    summary_prefix = f"{s3_prefix}/{training_job_name}/output/output/"
    logger.info(f"Searching for bedrock summary in s3://{bucket_name}/{summary_prefix}")
    
    summary_response = s3_client.list_objects_v2(
        Bucket=bucket_name,
        Prefix=summary_prefix,
        MaxKeys=1000
    )
    
    bedrock_job_name = None
    if 'Contents' in summary_response:
        for obj in summary_response['Contents']:
            if 'bedrock_llm_judge_results.json' in obj['Key']:
                # Extract job name: .../output/output/{job_name}/eval_results/...
                match = re.search(r'/output/output/([^/]+)/', obj['Key'])
                if match:
                    bedrock_job_name = match.group(1)
                    logger.info(f"Found bedrock job name: {bedrock_job_name}")
                    break
    
    # Search for JSONL file
    if bedrock_job_name:
        search_prefix = f"{s3_prefix}/{bedrock_job_name}/"
    else:
        logger.warning("Could not find bedrock job name, searching broadly")
        search_prefix = s3_prefix
    
    logger.info(f"Searching for JSONL in s3://{bucket_name}/{search_prefix}")
    
    response = s3_client.list_objects_v2(
        Bucket=bucket_name,
        Prefix=search_prefix,
        MaxKeys=1000
    )
    
    if 'Contents' not in response:
        raise FileNotFoundError(
            f"[PySDK Error] No results found at s3://{bucket_name}/{search_prefix}. "
            f"Evaluation results may not have been generated yet."
        )
    
    # Find _output.jsonl file
    jsonl_key = None
    for obj in response['Contents']:
        if obj['Key'].endswith('_output.jsonl'):
            jsonl_key = obj['Key']
            logger.info(f"Found JSONL: {jsonl_key}")
            break
    
    if not jsonl_key:
        raise FileNotFoundError(
            f"[PySDK Error] No _output.jsonl file found in s3://{bucket_name}/{search_prefix}. "
            f"Evaluation results may not have been generated yet."
        )
    
    logger.info(f"Found results file: {jsonl_key}")
    
    # Download and parse JSONL
    obj = s3_client.get_object(Bucket=bucket_name, Key=jsonl_key)
    content = obj['Body'].read().decode('utf-8')
    
    # Parse JSONL (one JSON per line)
    results = []
    for line in content.strip().split('\n'):
        if line.strip():
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSONL line: {e}")
                continue
    
    logger.info(f"Loaded {len(results)} evaluation results")
    return results


def _display_single_llmaj_evaluation(
    result: Dict[str, Any],
    index: int,
    total: int,
    console,
    show_explanations: bool = False
):
    """Display a single LLM As Judge evaluation result
    
    Args:
        result: Single evaluation result dict from JSONL
        index: Current evaluation index (0-based)
        total: Total number of evaluations
        console: Rich Console instance
        show_explanations: Whether to show judge explanations (default: False for brevity)
    """
    from rich.table import Table
    from rich.text import Text
    from rich.box import SIMPLE
    
    # Extract data
    input_record = result.get('inputRecord', {})
    scores = result.get('automatedEvaluationResult', {}).get('scores', [])
    
    prompt = _parse_prompt(input_record.get('prompt', 'N/A'))
    model_responses = result.get('modelResponses', [])
    response = _parse_response(model_responses[0]['response']) if model_responses else 'N/A'
    
    # Create header
    header = Text()
    header.append(f"\n═══ Evaluation {index + 1} of {total} ═══\n", style="bold cyan")
    console.print(header)
    
    # Prompt and Response
    console.print(f"[bold]Prompt:[/bold] {_truncate_text(prompt, 200)}")
    console.print(f"[bold]Model Response:[/bold] {_truncate_text(response, 200)}")
    console.print()
    
    # Scores table
    scores_table = Table(
        show_header=True,
        header_style="bold magenta",
        box=SIMPLE,
        padding=(0, 1)
    )
    scores_table.add_column("Metric", style="cyan", width=30)
    scores_table.add_column("Score", style="green", justify="right", width=10)
    
    if show_explanations:
        scores_table.add_column("Explanation", style="white", width=50)
    
    for score in scores:
        metric_name = score.get('metricName', 'Unknown')
        score_value = score.get('result', 0.0)
        
        row_data = [metric_name, _format_score(score_value)]
        
        if show_explanations:
            evaluator_details = score.get('evaluatorDetails', [])
            explanation = evaluator_details[0].get('explanation', 'N/A') if evaluator_details else 'N/A'
            row_data.append(_truncate_text(explanation, 80))
        
        scores_table.add_row(*row_data)
    
    console.print(scores_table)
    console.print()


def _show_llmaj_results(
    pipeline_execution,
    limit: int = 5,
    offset: int = 0,
    show_explanations: bool = False
):
    """Display LLM As Judge evaluation results with pagination
    
    Args:
        pipeline_execution: EvaluationPipelineExecution instance
        limit: Number of evaluations to display (default: 5, None for all)
        offset: Starting index for pagination (default: 0)
        show_explanations: Whether to show judge explanations (default: False)
    """
    # Detect Jupyter
    is_jupyter = False
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is not None and 'IPKernelApp' in ipython.config:
            is_jupyter = True
    except:
        pass
    
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    
    console = Console(force_jupyter=is_jupyter) if is_jupyter else Console()
    
    # Show S3 location first
    if pipeline_execution.s3_output_path:
        # Parse S3 to construct detailed path
        s3_path = pipeline_execution.s3_output_path[5:]
        bucket_name = s3_path.split('/')[0]
        s3_prefix = '/'.join(s3_path.split('/')[1:]).rstrip('/')
        
        # Get job name using common utility
        job_name = _extract_training_job_name_from_steps(pipeline_execution, 'Evaluate') or "unknown"
        
        s3_full_path = f"s3://{bucket_name}/{s3_prefix}/{job_name}/"
        
        message = Text()
        message.append("\n📦 ", style="bold blue")
        message.append("Full evaluation artifacts available at:\n", style="bold")
        message.append(f"  {s3_full_path}\n", style="cyan")
        
        console.print(Panel(
            message,
            title="[bold blue]Result Artifacts Location[/bold blue]",
            border_style="blue",
            padding=(1, 2)
        ))
        console.print()
    
    # Download results
    results = _download_llmaj_results_from_s3(pipeline_execution)
    total = len(results)
    
    # Apply pagination
    if limit is None:
        limit = total
    
    start_idx = offset
    end_idx = min(offset + limit, total)
    
    if start_idx >= total:
        console.print(f"[yellow]Offset {offset} is beyond total {total} evaluations[/yellow]")
        return
    
    # Display evaluations
    for i in range(start_idx, end_idx):
        _display_single_llmaj_evaluation(
            results[i],
            i,
            total,
            console,
            show_explanations=show_explanations
        )
    
    # Show pagination info
    console.print("═" * 70)
    console.print(f"[bold cyan]Showing evaluations {start_idx + 1}-{end_idx} of {total}[/bold cyan]\n")
    
    if end_idx < total:
        console.print("[dim]To see more:[/dim]")
        console.print(f"  [cyan]job.show_results(limit={limit}, offset={end_idx})[/cyan]  # Next {limit}")
        if limit != total:
            console.print(f"  [cyan]job.show_results(limit=None)[/cyan]  # Show all {total}")
    
    console.print("═" * 70)
