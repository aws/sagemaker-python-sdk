# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""This module stores notebook utils related to SageMaker JumpStart."""
from __future__ import absolute_import
import copy

from concurrent.futures import ThreadPoolExecutor, as_completed

from functools import cmp_to_key
import json
from typing import Any, Generator, List, Optional, Tuple, Union, Set, Dict
from packaging.version import Version
from sagemaker.jumpstart import accessors
from sagemaker.jumpstart.constants import (
    DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    PROPRIETARY_MODEL_SPEC_PREFIX,
)
from sagemaker.jumpstart.enums import JumpStartScriptScope, JumpStartModelType
from sagemaker.jumpstart.filters import (
    SPECIAL_SUPPORTED_FILTER_KEYS,
    ProprietaryModelFilterIdentifiers,
    BooleanValues,
    Identity,
    SpecialSupportedFilterKeys,
)
from sagemaker.jumpstart.filters import Constant, ModelFilter, Operator, evaluate_filter_expression
from sagemaker.jumpstart.types import JumpStartModelHeader, JumpStartModelSpecs
from sagemaker.jumpstart.utils import (
    get_jumpstart_content_bucket,
    get_region_fallback,
    get_sagemaker_version,
    verify_model_region_and_return_specs,
    validate_model_id_and_get_type,
)
from sagemaker.session import Session

MAX_SEARCH_WORKERS = int(100 * 1e6 / 25 * 1e3)  # max 100MB total memory, 25kB per thread)


def _compare_model_version_tuples(  # pylint: disable=too-many-return-statements
    model_version_1: Optional[Tuple[str, str]] = None,
    model_version_2: Optional[Tuple[str, str]] = None,
) -> int:
    """Performs comparison of sdk specs paths, in order to sort them.

    Args:
        model_version_1 (Tuple[str, str]): The first model ID and version tuple to compare.
        model_version_2 (Tuple[str, str]): The second model ID and version tuple to compare.
    """
    if model_version_1 is None or model_version_2 is None:
        if model_version_2 is not None:
            return -1
        if model_version_1 is not None:
            return 1
        return 0

    model_id_1, version_1 = model_version_1

    model_id_2, version_2 = model_version_2

    if model_id_1 < model_id_2:
        return -1

    if model_id_2 < model_id_1:
        return 1

    if Version(version_1) < Version(version_2):
        return 1

    if Version(version_2) < Version(version_1):
        return -1

    return 0


def _model_filter_in_operator_generator(filter_operator: Operator) -> Generator:
    """Generator for model filters in an operator."""
    for operator in filter_operator:
        if isinstance(operator.unresolved_value, ModelFilter):
            yield operator


def _put_resolved_booleans_into_filter(
    filter_operator: Operator, model_filters_to_resolved_values: Dict[ModelFilter, BooleanValues]
) -> None:
    """Iterate over the operators in the filter, assign resolved value if found in second arg.

    If not found, assigns ``UNKNOWN``.
    """
    for operator in _model_filter_in_operator_generator(filter_operator):
        model_filter = operator.unresolved_value
        operator.resolved_value = model_filters_to_resolved_values.get(
            model_filter, BooleanValues.UNKNOWN
        )


def _populate_model_filters_to_resolved_values(
    manifest_specs_cached_values: Dict[str, Any],
    model_filters_to_resolved_values: Dict[ModelFilter, BooleanValues],
    model_filters: Operator,
) -> None:
    """Iterate over the model filters, if the filter key has a cached value, evaluate the filter.

    The resolved filter values are placed in ``model_filters_to_resolved_values``.
    """
    for model_filter in model_filters:
        if model_filter.key in manifest_specs_cached_values:
            cached_model_value = manifest_specs_cached_values[model_filter.key]
            evaluated_expression: BooleanValues = evaluate_filter_expression(
                model_filter, cached_model_value
            )
            model_filters_to_resolved_values[model_filter] = evaluated_expression


def extract_framework_task_model(model_id: str) -> Tuple[str, str, str]:
    """Parse the model ID, return a tuple framework, task, rest-of-id.

    Args:
        model_id (str): The model ID for which to extract the framework/task/model.
    """
    _id_parts = model_id.split("-")

    if len(_id_parts) < 3:
        return "", "", ""

    framework = _id_parts[0]
    task = _id_parts[1]
    name = "-".join(_id_parts[2:])

    return framework, task, name


def extract_model_type_filter_representation(spec_key: str) -> str:
    """Parses model spec key, determine if the model is proprietary or open weight.

    Args:
        spek_key (str): The model spec key for which to extract the model type.
    """
    model_spec_prefix = spec_key.split("/")[0]

    if model_spec_prefix == PROPRIETARY_MODEL_SPEC_PREFIX:
        return JumpStartModelType.PROPRIETARY.value

    return JumpStartModelType.OPEN_WEIGHTS.value


def list_jumpstart_tasks(  # pylint: disable=redefined-builtin
    filter: Union[Operator, str] = Constant(BooleanValues.TRUE),
    region: Optional[str] = None,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
) -> List[str]:
    """List tasks for JumpStart, and optionally apply filters to result.

    Args:
        filter (Union[Operator, str]): Optional. The filter to apply to list tasks. This can be
            either an ``Operator`` type filter (e.g. ``And("task == ic", "framework == pytorch")``),
            or simply a string filter which will get serialized into an Identity filter.
            (e.g. ``"task == ic"``). If this argument is not supplied, all tasks will be listed.
            (Default: Constant(BooleanValues.TRUE)).
        region (str): Optional. The AWS region from which to retrieve JumpStart metadata regarding
            models. (Default: None).
        sagemaker_session (sagemaker.session.Session): Optional. The SageMaker Session to
            use to perform the model search. (Default: DEFAULT_JUMPSTART_SAGEMAKER_SESSION).
    """

    region = region or get_region_fallback(
        sagemaker_session=sagemaker_session,
    )
    tasks: Set[str] = set()
    for model_id, _ in _generate_jumpstart_model_versions(
        filter=filter,
        region=region,
        sagemaker_session=sagemaker_session,
        model_type=JumpStartModelType.OPEN_WEIGHTS,
    ):
        _, task, _ = extract_framework_task_model(model_id)
        tasks.add(task)
    return sorted(list(tasks))


def list_jumpstart_frameworks(  # pylint: disable=redefined-builtin
    filter: Union[Operator, str] = Constant(BooleanValues.TRUE),
    region: Optional[str] = None,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
) -> List[str]:
    """List frameworks for JumpStart, and optionally apply filters to result.

    Args:
        filter (Union[Operator, str]): Optional. The filter to apply to list frameworks. This can be
            either an ``Operator`` type filter (e.g. ``And("task == ic", "framework == pytorch")``),
            or simply a string filter which will get serialized into an Identity filter.
            (eg. ``"task == ic"``). If this argument is not supplied, all frameworks will be listed.
            (Default: Constant(BooleanValues.TRUE)).
        region (str): Optional. The AWS region from which to retrieve JumpStart metadata regarding
            models. (Default: None).
        sagemaker_session (sagemaker.session.Session): Optional. The SageMaker Session
            to use to perform the model search. (Default: DEFAULT_JUMPSTART_SAGEMAKER_SESSION).
    """

    region = region or get_region_fallback(
        sagemaker_session=sagemaker_session,
    )
    frameworks: Set[str] = set()
    for model_id, _ in _generate_jumpstart_model_versions(
        filter=filter,
        region=region,
        sagemaker_session=sagemaker_session,
        model_type=JumpStartModelType.OPEN_WEIGHTS,
    ):
        framework, _, _ = extract_framework_task_model(model_id)
        frameworks.add(framework)
    return sorted(list(frameworks))


def list_jumpstart_scripts(  # pylint: disable=redefined-builtin
    filter: Union[Operator, str] = Constant(BooleanValues.TRUE),
    region: Optional[str] = None,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
) -> List[str]:
    """List scripts for JumpStart, and optionally apply filters to result.

    Args:
        filter (Union[Operator, str]): Optional. The filter to apply to list scripts. This can be
            either an ``Operator`` type filter (e.g. ``And("task == ic", "framework == pytorch")``),
            or simply a string filter which will get serialized into an Identity filter.
            (e.g. ``"task == ic"``). If this argument is not supplied, all scripts will be listed.
            (Default: Constant(BooleanValues.TRUE)).
        region (str): Optional. The AWS region from which to retrieve JumpStart metadata regarding
            models. (Default: None).
        sagemaker_session (sagemaker.session.Session): Optional. The SageMaker Session to
            use to perform the model search. (Default: DEFAULT_JUMPSTART_SAGEMAKER_SESSION).
    """
    region = region or get_region_fallback(
        sagemaker_session=sagemaker_session,
    )
    if (isinstance(filter, Constant) and filter.resolved_value == BooleanValues.TRUE) or (
        isinstance(filter, str) and filter.lower() == BooleanValues.TRUE.lower()
    ):
        return sorted([e.value for e in JumpStartScriptScope])

    scripts: Set[str] = set()
    for model_id, version in _generate_jumpstart_model_versions(
        filter=filter,
        region=region,
        sagemaker_session=sagemaker_session,
        model_type=JumpStartModelType.OPEN_WEIGHTS,
    ):
        scripts.add(JumpStartScriptScope.INFERENCE)
        model_specs = verify_model_region_and_return_specs(
            region=region,
            model_id=model_id,
            version=version,
            sagemaker_session=sagemaker_session,
            scope=JumpStartScriptScope.INFERENCE,
        )
        if model_specs.training_supported:
            scripts.add(JumpStartScriptScope.TRAINING)

        if scripts == {e.value for e in JumpStartScriptScope}:
            break
    return sorted(list(scripts))


def _is_valid_version(version: str) -> bool:
    """Checks if the version is convertable to Version class."""
    try:
        Version(version)
        return True
    except Exception:  # pylint: disable=broad-except
        return False


def list_jumpstart_models(  # pylint: disable=redefined-builtin
    filter: Union[Operator, str] = Constant(BooleanValues.TRUE),
    region: Optional[str] = None,
    list_incomplete_models: bool = False,
    list_old_models: bool = False,
    list_versions: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
) -> List[Union[Tuple[str], Tuple[str, str]]]:
    """List models for JumpStart, and optionally apply filters to result.

    Args:
        filter (Union[Operator, str]): Optional. The filter to apply to list models. This can be
            either an ``Operator`` type filter (e.g. ``And("task == ic", "framework == pytorch")``),
            or simply a string filter which will get serialized into an Identity filter.
            (e.g. ``"task == ic"``). If this argument is not supplied, all models will be listed.
            (Default: Constant(BooleanValues.TRUE)).
        region (str): Optional. The AWS region from which to retrieve JumpStart metadata regarding
            models. (Default: None).
        list_incomplete_models (bool): Optional. If a model does not contain metadata fields
            requested by the filter, and the filter cannot be resolved to a include/not include,
            whether the model should be included. By default, these models are omitted from results.
            (Default: False).
        list_old_models (bool): Optional. If there are older versions of a model, whether the older
            versions should be included in the returned result. (Default: False).
        list_versions (bool): Optional. True if versions for models should be returned in addition
            to the id of the model. (Default: False).
        sagemaker_session (sagemaker.session.Session): Optional. The SageMaker Session to use
            to perform the model search. (Default: DEFAULT_JUMPSTART_SAGEMAKER_SESSION).
    """

    region = region or get_region_fallback(
        sagemaker_session=sagemaker_session,
    )
    model_id_version_dict: Dict[str, List[str]] = dict()
    for model_id, version in _generate_jumpstart_model_versions(
        filter=filter,
        region=region,
        list_incomplete_models=list_incomplete_models,
        sagemaker_session=sagemaker_session,
    ):
        if model_id not in model_id_version_dict:
            model_id_version_dict[model_id] = list()
        model_version = Version(version) if _is_valid_version(version) else version
        model_id_version_dict[model_id].append(model_version)

    if not list_versions:
        return sorted(list(model_id_version_dict.keys()))

    if not list_old_models:
        for model_id, versions in model_id_version_dict.items():
            try:
                model_id_version_dict.update({model_id: set([max(versions)])})
            except TypeError:
                versions = [str(v) for v in versions]
                model_id_version_dict.update({model_id: set([max(versions)])})

    model_id_version_set: Set[Tuple[str, str]] = set()
    for model_id in model_id_version_dict:
        for version in model_id_version_dict[model_id]:
            model_id_version_set.add((model_id, str(version)))

    return sorted(list(model_id_version_set), key=cmp_to_key(_compare_model_version_tuples))


def _generate_jumpstart_model_versions(  # pylint: disable=redefined-builtin
    filter: Union[Operator, str] = Constant(BooleanValues.TRUE),
    region: Optional[str] = None,
    list_incomplete_models: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    model_type: Optional[JumpStartModelType] = None,
) -> Generator:
    """Generate models for JumpStart, and optionally apply filters to result.

    Args:
        filter (Union[Operator, str]): Optional. The filter to apply to generate models. This can be
            either an ``Operator`` type filter (e.g. ``And("task == ic", "framework == pytorch")``),
            or simply a string filter which will get serialized into an Identity filter.
            (e.g. ``"task == ic"``). If this argument is not supplied, all models will be generated.
            (Default: Constant(BooleanValues.TRUE)).
        region (str): Optional. The AWS region from which to retrieve JumpStart metadata regarding
            models. (Default: None).
        list_incomplete_models (bool): Optional. If a model does not contain metadata fields
            requested by the filter, and the filter cannot be resolved to a include/not include,
            whether the model should be included. By default, these models are omitted from
            results. (Default: False).
        sagemaker_session (sagemaker.session.Session): Optional. The SageMaker Session
            to use to perform the model search. (Default: DEFAULT_JUMPSTART_SAGEMAKER_SESSION).
    """

    region = region or get_region_fallback(
        sagemaker_session=sagemaker_session,
    )

    prop_models_manifest_list = accessors.JumpStartModelsAccessor._get_manifest(
        region=region,
        s3_client=sagemaker_session.s3_client,
        model_type=JumpStartModelType.PROPRIETARY,
    )
    open_weight_manifest_list = accessors.JumpStartModelsAccessor._get_manifest(
        region=region,
        s3_client=sagemaker_session.s3_client,
        model_type=JumpStartModelType.OPEN_WEIGHTS,
    )
    models_manifest_list = (
        open_weight_manifest_list
        if model_type == JumpStartModelType.OPEN_WEIGHTS
        else (
            prop_models_manifest_list
            if model_type == JumpStartModelType.PROPRIETARY
            else open_weight_manifest_list + prop_models_manifest_list
        )
    )

    if isinstance(filter, str):
        filter = Identity(filter)

    manifest_keys = set(
        open_weight_manifest_list[0].__slots__ + prop_models_manifest_list[0].__slots__
    )

    all_keys: Set[str] = set()

    model_filters: Set[ModelFilter] = set()

    for operator in _model_filter_in_operator_generator(filter):
        model_filter = operator.unresolved_value
        key = model_filter.key
        all_keys.add(key)
        if model_filter.key == SpecialSupportedFilterKeys.MODEL_TYPE and model_filter.value in {
            identifier.value for identifier in ProprietaryModelFilterIdentifiers
        }:
            model_filter.set_value(JumpStartModelType.PROPRIETARY.value)
        model_filters.add(model_filter)

    for key in all_keys:
        if "." in key:
            raise NotImplementedError(f"No support for multiple level metadata indexing ('{key}').")

    metadata_filter_keys = all_keys - SPECIAL_SUPPORTED_FILTER_KEYS

    required_manifest_keys = manifest_keys.intersection(metadata_filter_keys)
    possible_spec_keys = metadata_filter_keys - manifest_keys

    is_task_filter = SpecialSupportedFilterKeys.TASK in all_keys
    is_framework_filter = SpecialSupportedFilterKeys.FRAMEWORK in all_keys
    is_model_type_filter = SpecialSupportedFilterKeys.MODEL_TYPE in all_keys

    def evaluate_model(model_manifest: JumpStartModelHeader) -> Optional[Tuple[str, str]]:

        copied_filter = copy.deepcopy(filter)

        manifest_specs_cached_values: Dict[str, Union[bool, int, float, str, dict, list]] = {}

        model_filters_to_resolved_values: Dict[ModelFilter, BooleanValues] = {}

        for val in required_manifest_keys:
            manifest_specs_cached_values[val] = getattr(model_manifest, val)

        if is_task_filter:
            manifest_specs_cached_values[SpecialSupportedFilterKeys.TASK] = (
                extract_framework_task_model(model_manifest.model_id)[1]
            )

        if is_framework_filter:
            manifest_specs_cached_values[SpecialSupportedFilterKeys.FRAMEWORK] = (
                extract_framework_task_model(model_manifest.model_id)[0]
            )

        if is_model_type_filter:
            manifest_specs_cached_values[SpecialSupportedFilterKeys.MODEL_TYPE] = (
                extract_model_type_filter_representation(model_manifest.spec_key)
            )

        if Version(model_manifest.min_version) > Version(get_sagemaker_version()):
            return None

        _populate_model_filters_to_resolved_values(
            manifest_specs_cached_values,
            model_filters_to_resolved_values,
            model_filters,
        )

        _put_resolved_booleans_into_filter(copied_filter, model_filters_to_resolved_values)

        copied_filter.eval()

        if copied_filter.resolved_value in [BooleanValues.TRUE, BooleanValues.FALSE]:
            if copied_filter.resolved_value == BooleanValues.TRUE:
                return (model_manifest.model_id, model_manifest.version)
            return None

        if copied_filter.resolved_value == BooleanValues.UNEVALUATED:
            raise RuntimeError(
                "Filter expression in unevaluated state after using "
                "values from model manifest. Model ID and version that "
                f"is failing: {(model_manifest.model_id, model_manifest.version)}."
            )
        copied_filter_2 = copy.deepcopy(filter)

        # spec is downloaded to thread's memory. since each thread
        # accesses a unique s3 spec, there is no need to use the JS caching utils.
        # spec only stays in memory for lifecycle of thread.
        model_specs = JumpStartModelSpecs(
            json.loads(
                sagemaker_session.read_s3_file(
                    get_jumpstart_content_bucket(region), model_manifest.spec_key
                )
            )
        )

        for val in possible_spec_keys:
            if hasattr(model_specs, val):
                manifest_specs_cached_values[val] = getattr(model_specs, val)

        _populate_model_filters_to_resolved_values(
            manifest_specs_cached_values,
            model_filters_to_resolved_values,
            model_filters,
        )
        _put_resolved_booleans_into_filter(copied_filter_2, model_filters_to_resolved_values)

        copied_filter_2.eval()

        if copied_filter_2.resolved_value != BooleanValues.UNEVALUATED:
            if copied_filter_2.resolved_value == BooleanValues.TRUE or (
                BooleanValues.UNKNOWN and list_incomplete_models
            ):
                return (model_manifest.model_id, model_manifest.version)
            return None

        raise RuntimeError(
            "Filter expression in unevaluated state after using values from model specs. "
            "Model ID and version that is failing: "
            f"{(model_manifest.model_id, model_manifest.version)}."
        )

    with ThreadPoolExecutor(max_workers=MAX_SEARCH_WORKERS) as executor:
        futures = []
        for header in models_manifest_list:
            futures.append(executor.submit(evaluate_model, header))

        for future in as_completed(futures):
            error = future.exception()
            if error:
                raise error
            result = future.result()
            if result:
                yield result


def get_model_url(
    model_id: str,
    model_version: str,
    region: Optional[str] = None,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
) -> str:
    """Retrieve web url describing pretrained model.

    Args:
        model_id (str): The model ID for which to retrieve the url.
        model_version (str): The model version for which to retrieve the url.
        region (str): Optional. The region from which to retrieve metadata.
            (Default: None)
        sagemaker_session (sagemaker.session.Session): Optional. The SageMaker Session to use
            to retrieve the model url.
    """
    model_type = validate_model_id_and_get_type(
        model_id=model_id,
        model_version=model_version,
        region=region,
        sagemaker_session=sagemaker_session,
    )

    region = region or get_region_fallback(
        sagemaker_session=sagemaker_session,
    )
    model_specs = verify_model_region_and_return_specs(
        region=region,
        model_id=model_id,
        version=model_version,
        sagemaker_session=sagemaker_session,
        scope=JumpStartScriptScope.INFERENCE,
        model_type=model_type,
    )
    return model_specs.url
