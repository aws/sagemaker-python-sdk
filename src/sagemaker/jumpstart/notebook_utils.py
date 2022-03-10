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

from functools import cmp_to_key
from typing import Any, Collection, List, Tuple, Union, Set, Dict
from packaging.version import Version
from sagemaker.jumpstart import accessors
from sagemaker.jumpstart.constants import JUMPSTART_DEFAULT_REGION_NAME
from sagemaker.jumpstart.enums import JumpStartScriptScope
from sagemaker.jumpstart.utils import get_sagemaker_version


def extract_framework_task_model(model_id: str) -> Tuple[str, str, str]:
    """Parse the input model id, return a tuple framework, task, rest-of-id.

    Args:
        model_id (str): The model id for which to extract the framework/task/model.
    """
    _id_parts = model_id.split("-")

    if len(_id_parts) < 3:
        raise ValueError(f"incorrect model id: {model_id}.")

    framework = _id_parts[0]
    task = _id_parts[1]
    name = "-".join(_id_parts[2:])

    return framework, task, name


def _compare_model_version_tuples(  # pylint: disable=too-many-return-statements
    model_version_1: Tuple[str, str] = None, model_version_2: Tuple[str, str] = None
) -> int:
    """Performs comparison of sdk specs paths, in order to sort them in manifest.

    Args:
        model_version_1 (Tuple[str, str]): The first model id and version tuple to compare.
        model_version_2 (Tuple[str, str]): The second model id and version tuple to compare.
    """
    if model_version_1 is None or model_version_2 is None:
        if model_version_2 is not None:
            return -1
        if model_version_1 is not None:
            return 1
        return 0

    version_1 = model_version_1[1]
    model_id_1 = model_version_1[0]

    version_2 = model_version_2[1]
    model_id_2 = model_version_2[0]

    if model_id_1 < model_id_2:
        return -1

    if model_id_2 < model_id_1:
        return 1

    if Version(version_1) < Version(version_2):
        return 1

    if Version(version_2) < Version(version_1):
        return -1

    return 0


def list_jumpstart_frameworks(
    **kwargs: Dict[str, Any],
) -> List[str]:
    """List frameworks actively in use by JumpStart.

    Args:
        kwargs (Dict[str, Any]): kwarg arguments to supply to
            ``list_jumpstart_models``.
    """
    models_list = list_jumpstart_models(**kwargs)
    frameworks = set()
    for model_id, _ in models_list:
        framework, _, _ = extract_framework_task_model(model_id)
        frameworks.add(framework)
    return sorted(list(frameworks))


def list_jumpstart_tasks(
    **kwargs: Dict[str, Any],
) -> List[str]:
    """List tasks actively in use by JumpStart.

    Args:
        kwargs (Dict[str, Any]): kwarg arguments to supply to
            ``list_jumpstart_models``.
    """
    models_list = list_jumpstart_models(**kwargs)
    tasks = set()
    for model_id, _ in models_list:
        _, task, _ = extract_framework_task_model(model_id)
        tasks.add(task)
    return sorted(list(tasks))


def list_jumpstart_scripts(
    **kwargs: Dict[str, Any],
) -> List[str]:
    """List scripts actively in use by JumpStart.

    Note: Using this function will result in slow execution speed, as it requires
    making many http calls and parsing metadata files. To-Do: store script
    information for all models in a single file.

    Check ``sagemaker.jumpstart.enums.JumpStartScriptScope`` for possible types
    of JumpStart scripts.

    Args:
        kwargs (Dict[str, Any]): kwarg arguments to supply to
            ``list_jumpstart_models``.
    """
    models_list = list_jumpstart_models(**kwargs)
    scripts = set()
    for model_id, version in models_list:
        scripts.add(JumpStartScriptScope.INFERENCE.value)
        model_specs = accessors.JumpStartModelsAccessor.get_model_specs(
            region=kwargs.get("region", JUMPSTART_DEFAULT_REGION_NAME),
            model_id=model_id,
            version=version,
        )
        if model_specs.training_supported:
            scripts.add(JumpStartScriptScope.TRAINING.value)

        if scripts == {e.value for e in JumpStartScriptScope}:
            break
    return sorted(list(scripts))


def list_jumpstart_models(
    script_allowlist: Union[str, Collection[str]] = None,
    task_allowlist: Union[str, Collection[str]] = None,
    framework_allowlist: Union[str, Collection[str]] = None,
    model_id_allowlist: Union[str, Collection[str]] = None,
    script_denylist: Union[str, Collection[str]] = None,
    task_denylist: Union[str, Collection[str]] = None,
    framework_denylist: Union[str, Collection[str]] = None,
    model_id_denylist: Union[str, Collection[str]] = None,
    region: str = JUMPSTART_DEFAULT_REGION_NAME,
    accept_unsupported_models: bool = False,
    accept_old_models: bool = False,
    accept_vulnerable_models: bool = True,
    accept_deprecated_models: bool = True,
) -> List[str]:
    """List models in JumpStart, and optionally apply filters to result.

    Args:
        script_allowlist (Union[str, Collection[str]]): Optional. String or
            ``Collection`` storing scripts. All models returned by this function
            must use a script which is specified in this argument. Note: Using this
            filter will result in slow execution speed, as it requires making more
            http calls and parsing many metadata files. To-Do: store script
            information for all models in a single file.
            (Default: None).
        task_allowlist (Union[str, Collection[str]]): Optional. String or
            ``Collection`` storing tasks. All models returned by this function
            must use a task which is specified in this argument.
            (Default: None).
        framework_allowlist (Union[str, Collection[str]]): Optional. String or
            ``Collection`` storing frameworks. All models returned by this function
            must use a frameworks which is specified in this argument.
            (Default: None).
        model_id_allowlist (Union[str, Collection[str]]): Optional. String or
            ``Collection`` storing model ids. All models returned by this function
            must use a model id which is specified in this argument.
            (Default: None).
        script_denylist (Union[str, Collection[str]]): Optional. String or
            ``Collection`` storing scripts. All models returned by this function
            must not use a script which is specified in this argument. Note: Using
            this filter will result in slow execution speed, as it requires making
            more http calls and parsing many metadata files. To-Do: store script
            information for all models in a single file.
            (Default: None).
        task_denylist (Union[str, Collection[str]]): Optional. String or
            ``Collection`` storing tasks. All models returned by this function
            must not use a task which is specified in this argument.
            (Default: None).
        framework_denylist (Union[str, Collection[str]]): Optional. String or
            ``Collection`` storing frameworks. All models returned by this function
            must not use a frameworks which is specified in this argument.
            (Default: None).
        model_id_denylist (Union[str, Collection[str]]): Optional. String or
            ``Collection`` storing scripts. All models returned by this function
            must not use a model id which is specified in this argument.
            (Default: None).
        region (str): Optional. Region to use when fetching JumpStart metadata.
            (Default: ``JUMPSTART_DEFAULT_REGION_NAME``).
        accept_unsupported_models (bool): Optional. Set to True to accept models that
            are not supported with the current SageMaker library version
            (Default: False).
        accept_old_models (bool): Optional. Set to True to accept model and version
            tuples for which a model with the same name and a newer version exists.
            (Default: False).
        accept_vulnerable_models (bool): Optional. Set to False to reject models that
            have a vulnerable inference or training script dependency. Note: accessing
            vulnerability information requires making many http calls and parsing many
            metadata files. To-Do: store vulnerability information for all models in a
            single file, and change default value to False. (Default: True).
        accept_deprecated_models (bool): Optional. Set to False to reject models that
            have been flagged as deprecated. Note: accessing deprecation information
            requires making many http calls and parsing many metadata files. To-Do:
            store deprecation information for all models in a single file, and change
            default value to False. (Default: True).
    """
    bad_script_filter = script_allowlist is not None and script_denylist is not None
    bad_task_filter = task_allowlist is not None and task_denylist is not None
    bad_framework_filter = framework_allowlist is not None and framework_denylist is not None
    bad_model_id_filter = model_id_allowlist is not None and model_id_denylist is not None

    if bad_script_filter or bad_task_filter or bad_framework_filter or bad_model_id_filter:
        raise ValueError(
            (
                "Cannot use an allowlist and denylist at the same time "
                "for a filter (script, task, framework, model id)"
            )
        )

    if isinstance(script_allowlist, str):
        script_allowlist = set([script_allowlist])

    if isinstance(task_allowlist, str):
        task_allowlist = set([task_allowlist])

    if isinstance(framework_allowlist, str):
        framework_allowlist = set([framework_allowlist])

    if isinstance(model_id_allowlist, str):
        model_id_allowlist = set([model_id_allowlist])

    if isinstance(script_denylist, str):
        script_denylist = set([script_denylist])

    if isinstance(task_denylist, str):
        task_denylist = set([task_denylist])

    if isinstance(framework_denylist, str):
        framework_denylist = set([framework_denylist])

    if isinstance(model_id_denylist, str):
        model_id_denylist = set([model_id_denylist])

    models_manifest_list = accessors.JumpStartModelsAccessor.get_manifest(region=region)
    model_id_version_dict: Dict[str, Set[str]] = dict()
    for model_manifest in models_manifest_list:
        model_id = model_manifest.model_id
        model_version = model_manifest.version

        if not accept_unsupported_models and Version(get_sagemaker_version()) < Version(
            model_manifest.min_version
        ):
            continue

        if model_id_allowlist is not None:
            model_id_allowlist = set(model_id_allowlist)
            if model_id not in model_id_allowlist:
                continue
        if model_id_denylist is not None:
            model_id_denylist = set(model_id_denylist)
            if model_id in model_id_denylist:
                continue

        framework, task, _ = extract_framework_task_model(model_id)
        supported_scripts = set([JumpStartScriptScope.INFERENCE.value])

        if task_allowlist is not None:
            task_allowlist = set(task_allowlist)
            if task not in task_allowlist:
                continue
        if task_denylist is not None:
            task_denylist = set(task_denylist)
            if task in task_denylist:
                continue

        if framework_allowlist is not None:
            framework_allowlist = set(framework_allowlist)
            if framework not in framework_allowlist:
                continue
        if framework_denylist is not None:
            framework_denylist = set(framework_denylist)
            if framework in framework_denylist:
                continue

        if script_denylist is not None or script_allowlist is not None:
            model_specs = accessors.JumpStartModelsAccessor.get_model_specs(
                region=region, model_id=model_id, version=model_version
            )
            if model_specs.training_supported:
                supported_scripts.add(JumpStartScriptScope.TRAINING.value)

        if script_allowlist is not None:
            script_allowlist = set(script_allowlist)
            if len(supported_scripts.intersection(script_allowlist)) == 0:
                continue
        if script_denylist is not None:
            script_denylist = set(script_denylist)
            if len(supported_scripts.intersection(script_denylist)) > 0:
                continue
        if not accept_vulnerable_models:
            model_specs = accessors.JumpStartModelsAccessor.get_model_specs(
                region=region, model_id=model_id, version=model_version
            )
            if model_specs.inference_vulnerable or model_specs.training_vulnerable:
                continue
        if not accept_deprecated_models:
            model_specs = accessors.JumpStartModelsAccessor.get_model_specs(
                region=region, model_id=model_id, version=model_version
            )
            if model_specs.deprecated:
                continue

        if model_id not in model_id_version_dict:
            model_id_version_dict[model_id] = set()

        model_id_version_dict[model_id].add(Version(model_version))

    if not accept_old_models:
        model_id_version_dict = {
            model: set([max(versions)]) for model, versions in model_id_version_dict.items()
        }

    model_id_set = set()
    for model_id in model_id_version_dict:
        for version in model_id_version_dict[model_id]:
            model_id_set.add((model_id, str(version)))

    return sorted(list(model_id_set), key=cmp_to_key(_compare_model_version_tuples))
