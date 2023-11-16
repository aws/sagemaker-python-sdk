"""DJL ModelBuilder Utils"""
from __future__ import absolute_import
from urllib.error import HTTPError
import math
import logging
from sagemaker.serve.utils.types import _DjlEngine
from sagemaker.djl_inference import defaults
from sagemaker.djl_inference.model import _get_model_config_properties_from_hf
from sagemaker.serve.utils.local_hardware import _get_available_gpus
from sagemaker.serve.builder.schema_builder import SchemaBuilder

logger = logging.getLogger(__name__)

ATTENTION_HEAD_NAME_VARIENTS = ["n_head", "n_heads", "num_head", "num_heads", "num_attention_heads"]
CHARS_PER_TOKEN = 4
TOKENS_PER_WORD = 0.75


def _auto_detect_engine(model_id: str, hf_hub_token: str) -> tuple:
    """Placeholder docstring"""
    try:
        hf_model_config = _get_model_config_properties_from_hf(model_id, hf_hub_token)
        model_type = hf_model_config.get("model_type")

        if len(model_type) < 1:
            logger.warning(
                "Unable to detect the model architecture from provided model_id %s.\
            Defaulting to HuggingFaceAccelerate."
                % model_id
            )
            engine = _DjlEngine.HUGGINGFACE_ACCELERATE
        elif model_type in defaults.DEEPSPEED_RECOMMENDED_ARCHITECTURES:
            logger.info("Model architecture %s is recommended to be run on DeepSpeed." % model_type)
            engine = _DjlEngine.DEEPSPEED
        elif model_type in defaults.FASTER_TRANSFORMER_RECOMMENDED_ARCHITECTURES:
            logger.info(
                "Model architecture %s is recommended to be run on FasterTransformer." % model_type
            )
            engine = _DjlEngine.FASTER_TRANSFORMER
        else:
            logger.info(
                "Model architecture %s does not have a recommended engine. Defaulting to HuggingFaceAccelerate."
                % model_type
            )
            engine = _DjlEngine.HUGGINGFACE_ACCELERATE
    except HTTPError as e:
        raise ValueError(
            "The provided HuggingFace Model ID could not be accessed from HuggingFace Hub. %s",
            str(e),
        )
    except ValueError as e:
        raise e
    except Exception as e:
        logger.warning(
            "Unable to detect the model's architecture: %s. Defaulting to HuggingFaceAccelerate."
            % str(e)
        )
        engine = _DjlEngine.HUGGINGFACE_ACCELERATE

    return (engine, hf_model_config)


def _get_default_tensor_parallel_degree(hf_model_config: dict, gpu_count: int = None) -> int:
    """Placeholder docstring"""
    available_gpus = _get_available_gpus()
    if not available_gpus and not gpu_count:
        return None

    attention_heads = None
    for variant in ATTENTION_HEAD_NAME_VARIENTS:
        attention_heads = hf_model_config.get(variant)
        if attention_heads:
            break

    if not attention_heads:
        return 1

    tot_gpus = len(available_gpus) if available_gpus else gpu_count
    for i in (n + 1 for n in reversed(range(tot_gpus))):
        if attention_heads % i == 0:
            logger.info(
                "Max GPU parallelism of %s is allowed. Total attention heads %s", i, attention_heads
            )
            return i

    return 1


def _get_default_data_type() -> tuple:
    """Placeholder docstring"""
    return "fp16"


def _get_default_batch_size() -> int:
    """Placeholder docstring"""
    return 1


def _set_tokens_to_tokens_threshold(tokens: int) -> int:
    """Placeholder docstring"""
    if tokens <= 128:
        return 128
    if tokens <= 256:
        return 256
    elif tokens <= 512:
        return 512
    elif tokens <= 1024:
        return 1024
    elif tokens <= 2048:
        return 2048
    return 4096


def _tokens_from_chars(text: str) -> int:
    """Placeholder docstring"""
    return len(text) / CHARS_PER_TOKEN


def _tokens_from_words(text: str) -> int:
    """Placeholder docstring"""
    return math.ceil(len(text.split(" ")) * TOKENS_PER_WORD)


def _get_default_max_tokens(sample_input, sample_output) -> tuple:
    """Placeholder docstring"""
    inputs = sample_input.get("inputs")
    generated_text = sample_output[0].get("generated_text")

    input_tokens_from_chars = _tokens_from_chars(inputs)
    input_tokens_from_words = _tokens_from_words(inputs)
    max_input_tokens = max(input_tokens_from_chars, input_tokens_from_words)

    output_tokens_from_chars = _tokens_from_chars(generated_text)
    output_tokens_from_words = _tokens_from_words(generated_text)
    max_output_tokens = max(output_tokens_from_chars, output_tokens_from_words)

    max_total_tokens = _set_tokens_to_tokens_threshold(max_input_tokens + max_output_tokens)

    max_new_tokens = sample_input.get("parameters").get("max_new_tokens")
    if not max_new_tokens:
        max_new_tokens = _set_tokens_to_tokens_threshold(max_output_tokens)

    return (max_total_tokens, max_new_tokens)


def _set_serve_properties(hf_model_config: dict, schema_builder: SchemaBuilder) -> tuple:
    """Placeholder docstring"""
    default_tensor_parallel_degree = _get_default_tensor_parallel_degree(hf_model_config)
    default_data_type = _get_default_data_type()
    default_batch_size = _get_default_batch_size()
    default_max_tokens, default_max_new_tokens = _get_default_max_tokens(
        schema_builder.sample_input, schema_builder.sample_output
    )

    return (
        default_tensor_parallel_degree,
        default_data_type,
        default_batch_size,
        default_max_tokens,
        default_max_new_tokens,
    )


def _get_admissible_tensor_parallel_degrees(hf_model_config: dict) -> int:
    """Placeholder docstring"""
    available_gpus = _get_available_gpus()

    attention_heads = None
    for variant in ATTENTION_HEAD_NAME_VARIENTS:
        attention_heads = hf_model_config.get(variant)
        if attention_heads:
            break

    if not attention_heads:
        return 1

    admissible_parallel_degrees = []
    for i in (n + 1 for n in reversed(range(len(available_gpus)))):
        if attention_heads % i == 0:
            admissible_parallel_degrees.append(i)

    logger.info("Model can be sharded across %s GPUs", admissible_parallel_degrees)

    return admissible_parallel_degrees


def _get_admissible_dtypes():
    """Placeholder docstring"""
    return ["bf16"]
