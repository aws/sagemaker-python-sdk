# Standard Library
import json
import os
import re
from pydoc import locate

# Third Party
import yaml

# NOTE: this file is duplicated across smp and SageMaker Python SDK.
# any changes to this file requires an update of the corresponding file
# in SageMaker Python SDK upon release.

try:
    from smdistributed.modelparallel.backend.logger import get_logger

    logger = get_logger()
    info_func = logger.info
    warn_func = logger.warning
    py_sdk = False
except ImportError:
    # the case where this file runs on Python SDK for config validation
    info_func = print
    warn_func = print
    py_sdk = True


def int_or_float_if_possible(s):
    try:
        float_s = float(s)
        int_s = int(float_s)
        return int_s if int_s == float_s else float_s
    except ValueError:
        return s


class ConfigParam:
    def __init__(self, name, input_value, cfg_dict, existing_params, provided=False):
        self.name = name
        self.cfg_dict = cfg_dict
        self.existing_params = existing_params
        self.provided = provided
        self._value = self._set_value(input_value)

    def get_value(self):
        return self._value

    def _get_default(self):
        default = self._handle_dependencies(self.cfg_dict["default"])
        if isinstance(default, (float, int)):
            # should explicitly enforce upper and lower bounds for dynamic formulas in case
            # the default evaluates out of bounds
            if "upper_bound" in self.cfg_dict:
                default = min(default, self._handle_dependencies(self.cfg_dict["upper_bound"]))
            if "lower_bound" in self.cfg_dict:
                default = max(default, self._handle_dependencies(self.cfg_dict["lower_bound"]))

        return default

    def _set_value(self, input_value):
        if not self.provided:
            if "default" not in self.cfg_dict:
                raise ValueError(f"Config parameter {self.name} is required.")

            return self._get_default()

        if "type" in self.cfg_dict:
            expected_type = locate(self.cfg_dict["type"])
            if expected_type != type(input_value):
                raise TypeError(
                    f"Config parameter {self.name} needs to be of type {expected_type.__name__}. Found: {type(input_value)}"
                )

        if "options" in self.cfg_dict:
            options = self._handle_dependencies(self.cfg_dict["options"])
            if input_value not in options:
                raise ValueError(
                    f"Config parameter {self.name} must be one of {self.cfg_dict['options']}. Found: {input_value}."
                )

        if "lower_bound" in self.cfg_dict:
            lower_bound = self._handle_dependencies(self.cfg_dict["lower_bound"])
            if input_value < lower_bound:
                raise ValueError(
                    f"Config parameter {self.name} ({input_value}) cannot be less than {self.cfg_dict['lower_bound']} ({lower_bound})."
                )

        if "upper_bound" in self.cfg_dict:
            upper_bound = self._handle_dependencies(self.cfg_dict["upper_bound"])
            if input_value > upper_bound:
                raise ValueError(
                    f"Config parameter {self.name} ({input_value}) cannot be larger than {self.cfg_dict['upper_bound']} ({upper_bound})."
                )

        if "requires" in self.cfg_dict:
            default = self._get_default()
            for k, v in self.cfg_dict["requires"].items():
                if self.existing_params[k].get_value() != v and input_value != default:
                    raise ValueError(
                        f"Setting config parameter {self.name} to non-default value {input_value} requires {k} to be set to {v}. Found: {self.existing_params[k].get_value()}"
                    )

        if "requires_not" in self.cfg_dict:
            default = self._get_default()
            for k, v in self.cfg_dict["requires_not"].items():
                if self.existing_params[k].get_value() == v and input_value != default:
                    raise ValueError(
                        f"Setting config parameter {self.name} to non-default value {input_value} requires {k} to not be {v}."
                    )

        return input_value

    def _maybe_convert(self, value):
        if value[0] == "(" and value[-1] == ")" and value[1:-1] in self.existing_params:
            value = self.existing_params[value[1:-1]].get_value()

        return int_or_float_if_possible(value)

    def _handle_dependencies(self, value):
        if isinstance(value, str):
            tokens = re.split("\\+|\\-|\\*|\\/", value)
            ops = [c for c in value if c in ["+", "-", "*", "/"]]

            assert len(tokens) == len(ops) + 1, f"Malformed formula: {value}"

            # if there are operations, all terms must be convertible to float or int
            # if not, cur_value can be a string
            tokens = [self._maybe_convert(token.strip()) for token in tokens]
            cur_value = tokens[0]
            for op, val in zip(ops, tokens[1:]):
                if op == "+":
                    cur_value += val
                elif op == "-":
                    cur_value -= val
                elif op == "*":
                    cur_value *= val
                elif op == "/":
                    cur_value /= val
            return cur_value
        else:
            return value


class DependencyIterator:
    def __init__(self, config):
        self.config = config
        self.seen = set()

    def __iter__(self):
        return self

    def __next__(self):
        for k in self.config:
            if k not in self.seen:
                if "dependencies" not in self.config[k]:
                    self.seen.add(k)
                    return k

                if all([(d in self.seen) for d in self.config[k]["dependencies"]]):
                    self.seen.add(k)
                    return k

        raise StopIteration


class ModelParallelConfig:
    """Structure that holds the user-defined parameters for SMP."""

    def __init__(self, config):
        if not py_sdk:
            SM_CONFIG = json.loads(os.environ.get("SM_HP_MP_PARAMETERS", default="{}"))
            for each_sm_config in SM_CONFIG:
                config[each_sm_config] = SM_CONFIG[each_sm_config]

        # load the schema from yaml file
        dir_path = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(dir_path, "config.yaml")
        with open(config_path, "r") as f:
            schema = yaml.safe_load(f)

        # handle aliases - note below logic assumes at most one alias per config parameter
        aliases = {v["alias"]: k for k, v in schema.items() if "alias" in v}
        for alias, orig in aliases.items():
            if alias in config:
                if orig in config and config[alias] != config[orig]:
                    raise ValueError(
                        f"Conflicting values {config[orig]} and {config[alias]} are provided for config parameter {orig} and its alias {alias}."
                    )
                config[orig] = config[alias]
                del config[alias]

        # make sure there are no invalid config parameters
        for k in config:
            if k not in schema:
                raise ValueError(f"Unrecognized config parameter {k}.")

        # parse and validate the inputs
        params = {}
        for k in DependencyIterator(schema):
            provided = k in config
            input_value = config[k] if provided else None
            params[k] = ConfigParam(k, input_value, schema[k], params, provided)

        # set the attributes
        for k in params:
            setattr(self, k, params[k].get_value())

        self._input_config = config
        self._config_dict = params

        # enforce additional constraints - need to be careful here to make sure these do not conflict with the
        # existing constraints
        if self.active_microbatches != self.microbatches and self.pipeline != "interleaved":
            # PT limitation right now
            self.pipeline = "interleaved"
            info_func(
                "Simple pipeline is only supported when 'active_microbatches' is equal to 'microbatches'. Using interleaved pipeline instead."
            )

        if self.pipeline_parallel_degree > 1 and self.checkpoint_attentions:
            warn_func(
                f"Cannot checkpoint attentions when pipeline-parallel degree is more than 1, disabling attention checkpointing."
            )
            self.checkpoint_attentions = False

    def display_config(self):
        assert hasattr(self, "_config_dict")

        if not py_sdk:
            info_func("Configuration parameters:")
            for k, v in self._config_dict.items():
                if "internal" not in v.cfg_dict or not v.cfg_dict["internal"]:
                    info_func(f"  {k}: {v.get_value()}")
