"""Holds mixin logic to support deployment of Model ID"""

from __future__ import absolute_import

import logging
from time import perf_counter
import collections
from multiprocessing.pool import ThreadPool
from math import ceil
import pandas as pd
from numpy import percentile, std
from sagemaker.serve.model_server.djl_serving.utils import _tokens_from_chars, _tokens_from_words
from sagemaker.base_predictor import PredictorBase

WARMUP = 2
INVOCATIONS = 10
CONCURRENCY = 10
MARGIN = 10

logger = logging.getLogger(__name__)


def _pretty_print_results(results: dict):
    """Placeholder docstring"""
    avg_latencies = []
    tensor_parallel_degrees = []
    dtypes = []
    p90s = []
    avg_tokens_per_seconds = []
    throughput_per_seconds = []
    standard_deviations = []
    ordered = collections.OrderedDict(sorted(results.items()))

    for key, value in ordered.items():
        avg_latencies.append(key)
        tensor_parallel_degrees.append(value[0]["TENSOR_PARALLEL_DEGREE"])
        dtypes.append(value[0]["OPTION_DTYPE"])
        p90s.append(value[1])
        avg_tokens_per_seconds.append(value[2])
        throughput_per_seconds.append(value[3])
        standard_deviations.append(value[4])

    df = pd.DataFrame(
        {
            "AverageLatency (Serial)": avg_latencies,
            "P90_Latency (Serial)": p90s,
            "AverageTokensPerSecond (Serial)": avg_tokens_per_seconds,
            "ThroughputPerSecond (Concurrent)": throughput_per_seconds,
            "StandardDeviationResponse (Concurrent)": standard_deviations,
            "TensorParallelDegree": tensor_parallel_degrees,
            "DType": dtypes,
        }
    )
    logger.info(
        "\n================================================================== Benchmark "
        "Results ==================================================================\n%s"
        "\n============================================================================"
        "===========================================================================\n",
        df.to_string(),
    )


def _pretty_print_results_tgi(results: dict):
    """Placeholder docstring"""
    avg_latencies = []
    num_shard = []
    dtypes = []
    p90s = []
    avg_tokens_per_seconds = []
    throughput_per_seconds = []
    standard_deviations = []
    ordered = collections.OrderedDict(sorted(results.items()))

    for key, value in ordered.items():
        avg_latencies.append(key)
        num_shard.append(value[0]["NUM_SHARD"])
        dtypes.append(value[0]["DTYPE"])
        p90s.append(value[1])
        avg_tokens_per_seconds.append(value[2])
        throughput_per_seconds.append(value[3])
        standard_deviations.append(value[4])

    df = pd.DataFrame(
        {
            "AverageLatency (Serial)": avg_latencies,
            "P90_Latency (Serial)": p90s,
            "AverageTokensPerSecond (Serial)": avg_tokens_per_seconds,
            "ThroughputPerSecond (Concurrent)": throughput_per_seconds,
            "StandardDeviationResponse (Concurrent)": standard_deviations,
            "NumShard": num_shard,
            "DType": dtypes,
        }
    )
    logger.info(
        "\n================================================================== Benchmark "
        "Results ==================================================================\n%s"
        "\n============================================================================"
        "===========================================================================\n",
        df.to_string(),
    )


def _pretty_print_results_jumpstart(results: dict, model_env_vars=None):
    """Pretty prints benchmark results"""
    if model_env_vars is None:
        model_env_vars = []

    __env_var_data = {}
    for model_env_var in model_env_vars:
        __env_var_data[model_env_var] = []

    avg_latencies = []
    p90s = []
    avg_tokens_per_seconds = []
    throughput_per_seconds = []
    standard_deviations = []
    ordered = collections.OrderedDict(sorted(results.items()))

    for key, value in ordered.items():
        avg_latencies.append(key)
        p90s.append(value[1])
        avg_tokens_per_seconds.append(value[2])
        throughput_per_seconds.append(value[3])
        standard_deviations.append(value[4])

        for model_env_var in __env_var_data:
            __env_var_data[model_env_var].append(value[0][model_env_var])

    df = pd.DataFrame(
        {
            "AverageLatency (Serial)": avg_latencies,
            "P90_Latency (Serial)": p90s,
            "AverageTokensPerSecond (Serial)": avg_tokens_per_seconds,
            "ThroughputPerSecond (Concurrent)": throughput_per_seconds,
            "StandardDeviationResponse (Concurrent)": standard_deviations,
            **__env_var_data,
        }
    )

    logger.info(
        "\n================================================================== Benchmark "
        "Results ==================================================================\n%s"
        "\n============================================================================"
        "===========================================================================\n",
        df.to_string(),
    )


def _tokens_per_second(generated_text: str, max_token_length: int, latency: float) -> int:
    """Placeholder docstring"""
    est_tokens = (_tokens_from_chars(generated_text) + _tokens_from_words(generated_text)) / 2
    return min(est_tokens, max_token_length) / latency


def _timed_invoke(predict: callable, sample_input: object) -> tuple:
    """Placeholder docstring"""
    start_timer = perf_counter()
    response = predict(sample_input)
    stop_timer = perf_counter()

    elapsed_time = stop_timer - start_timer

    if isinstance(response, list):
        generated_text = response[0]["generated_text"]
    else:
        generated_text = response["generated_text"]

    tokens_per_second = _tokens_per_second(
        generated_text, sample_input["parameters"]["max_new_tokens"], elapsed_time
    )

    return (elapsed_time, tokens_per_second)


def _serial_benchmark(predictor: PredictorBase, sample_input: object) -> tuple:
    """Placeholder docstring"""
    latencies = []
    tokens_per_seconds = []

    for _ in range(WARMUP):
        predictor.predict(sample_input)

    logger.info("")
    logger.info("=============== Running Serial Benchmark.... ================")
    for itr in range(INVOCATIONS):
        elapsed_time, tokens_per_second = _timed_invoke(predictor.predict, sample_input)

        logger.info(
            "Invocation: %s => Latency: %s, Tokens/s: %s",
            itr,
            elapsed_time,
            tokens_per_second,
        )

        tokens_per_seconds.append(tokens_per_second)
        latencies.append(elapsed_time)
    logger.info("================ Completed Serial Benchmark =================\n")

    avg_latency = sum(latencies) / len(latencies)
    avg_tokens_per_second = sum(tokens_per_seconds) / len(tokens_per_seconds)

    return (
        avg_latency,
        percentile(latencies, 90),
        avg_tokens_per_second,
    )


def _pretty_print_ordered_histogram(data: list, label: str):
    """Placeholder docstring"""
    i = 0
    data.sort()
    for d in data:
        line = "x" * ceil(d)
        logger.info("%s: %s", label + " " + str(i), line)
        i += 1


def _concurrent_benchmark(predictor: PredictorBase, sample_input: object) -> tuple:
    """Placeholder docstring"""
    logger.info("============= Running Concurrent Benchmark.... ==============")
    concurrent_users = [(predictor.predict, sample_input)] * CONCURRENCY
    latencies = []
    with ThreadPool(CONCURRENCY) as pool:
        t = 0
        for latency, _ in pool.starmap(_timed_invoke, concurrent_users):
            latencies.append(latency)
            logger.info("User: %s => latency: %s seconds", t, latency)
            t += 1

    throughput_per_second = CONCURRENCY / sum(latencies)
    standard_deviation = std(latencies)

    logger.info("")
    logger.info("Model Latencies for Queued Requests:")
    _pretty_print_ordered_histogram(latencies, "queued request")
    logger.info("")

    logger.info(
        "Concurrent Benchmark Metrics => throughput/s: %s, standard deviation: %s",
        throughput_per_second,
        standard_deviation,
    )
    logger.info("============== Completed Concurrent Benchmark ===============\n")

    return (throughput_per_second, standard_deviation)


def _within_margins(margin: int, threshold: int, value1: float, value2: float) -> bool:
    """Placeholder docstring"""
    diff = abs(value1 - value2)
    return 0 < diff * margin <= threshold


def _more_performant(best_tuned_configuration: list, tuned_configuration: list) -> bool:
    """Placeholder docstring"""
    best_avg_latency = best_tuned_configuration[0]
    tuned_avg_latency = tuned_configuration[0]
    best_standard_deviation = best_tuned_configuration[6]
    tuned_standard_deviation = tuned_configuration[6]

    if _within_margins(MARGIN, 5, tuned_avg_latency, best_avg_latency):
        if tuned_standard_deviation <= best_standard_deviation:
            return True
        return False
    return tuned_avg_latency <= best_avg_latency


def _sharded_supported(model_id: str, config_dict: dict) -> bool:
    """Check if sharded is supported for this ``Model``"""
    model_type = config_dict.get("model_type", None)

    if model_type is None:
        return False

    if model_id.startswith("facebook/galactica"):
        return True

    if model_type in ["bloom", "mpt", "ssm", "gpt_neox", "phi", "phi-msft", "opt", "t5"]:
        return True

    if model_type in ["RefinedWeb", "RefinedWebModel", "falcon"] and not config_dict.get(
        "alibi", False
    ):
        return True

    return False
