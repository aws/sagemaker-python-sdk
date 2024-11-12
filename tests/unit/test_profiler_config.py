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
from __future__ import absolute_import

import os
import pytest
import re
import time
import warnings
from packaging import version

from sagemaker import image_uris
import sagemaker.fw_utils as fw
from sagemaker.pytorch import PyTorch
from sagemaker.tensorflow import TensorFlow
from sagemaker.debugger import get_rule_container_image_uri
from sagemaker.debugger.profiler_config import FrameworkProfile
from sagemaker import ProfilerConfig, Profiler

from sagemaker.debugger.metrics_config import (
    StepRange,
    TimeRange,
    DetailedProfilingConfig,
    DataloaderProfilingConfig,
    PythonProfilingConfig,
    HorovodProfilingConfig,
    SMDataParallelProfilingConfig,
)
from sagemaker.debugger.profiler_constants import (
    BASE_FOLDER_DEFAULT,
    CLOSE_FILE_INTERVAL_DEFAULT,
    FILE_OPEN_FAIL_THRESHOLD_DEFAULT,
    MAX_FILE_SIZE_DEFAULT,
    DATALOADER_PROFILING_CONFIG_NAME,
    DATALOADER_PROFILING_START_STEP_DEFAULT,
    DETAILED_PROFILING_CONFIG_NAME,
    DETAILED_PROFILING_START_STEP_DEFAULT,
    SMDATAPARALLEL_PROFILING_CONFIG_NAME,
    SMDATAPARALLEL_PROFILING_START_STEP_DEFAULT,
    HOROVOD_PROFILING_CONFIG_NAME,
    HOROVOD_PROFILING_START_STEP_DEFAULT,
    PROFILING_NUM_STEPS_DEFAULT,
    PYTHON_PROFILING_CONFIG_NAME,
    PYTHON_PROFILING_NUM_STEPS_DEFAULT,
    PYTHON_PROFILING_START_STEP_DEFAULT,
    START_STEP_DEFAULT,
    DETAIL_PROF_PROCESSING_DEFAULT_INSTANCE_TYPE,
    DETAIL_PROF_PROCESSING_DEFAULT_VOLUME_SIZE,
)
from sagemaker.debugger.utils import PythonProfiler, cProfileTimer, ErrorMessages


@pytest.fixture
def custom_local_path():
    return "/tmp/test"


@pytest.fixture
def custom_file_max_size():
    return MAX_FILE_SIZE_DEFAULT * 2


@pytest.fixture
def custom_file_close_interval():
    return CLOSE_FILE_INTERVAL_DEFAULT / 2


@pytest.fixture
def custom_file_open_fail_threshold():
    return FILE_OPEN_FAIL_THRESHOLD_DEFAULT + 10


@pytest.fixture
def custom_start_step():
    return 3


@pytest.fixture
def custom_num_steps():
    return 5


@pytest.fixture
def custom_start_unix_time():
    return int(time.time())


@pytest.fixture
def custom_duration():
    return 30


@pytest.fixture
def custom_metrics_regex():
    return "Dataset::Iterator::GetNext"


@pytest.fixture
def custom_python_profiler():
    return PythonProfiler.PYINSTRUMENT


@pytest.fixture
def custom_cprofile_timer():
    return cProfileTimer.CPU_TIME


@pytest.fixture
def default_framework_profile():
    return FrameworkProfile()


@pytest.fixture
def default_profiler_config():
    return ProfilerConfig()


@pytest.fixture
def framework_profile_with_custom_trace_file_fields(
    custom_local_path,
    custom_file_max_size,
    custom_file_close_interval,
    custom_file_open_fail_threshold,
):
    return FrameworkProfile(
        local_path=custom_local_path,
        file_max_size=custom_file_max_size,
        file_close_interval=custom_file_close_interval,
        file_open_fail_threshold=custom_file_open_fail_threshold,
    )


@pytest.fixture
def custom_detailed_profiling_config(custom_start_step, custom_num_steps):
    return DetailedProfilingConfig(start_step=custom_start_step, num_steps=custom_num_steps)


@pytest.fixture
def custom_dataloader_profiling_config(custom_start_step, custom_metrics_regex):
    return DataloaderProfilingConfig(
        start_step=custom_start_step, metrics_regex=custom_metrics_regex
    )


@pytest.fixture
def custom_python_profiling_config(custom_num_steps, custom_python_profiler):
    return PythonProfilingConfig(num_steps=custom_num_steps, python_profiler=custom_python_profiler)


@pytest.fixture
def custom_python_profiling_config_2(
    custom_start_unix_time, custom_duration, custom_cprofile_timer
):
    return PythonProfilingConfig(
        start_unix_time=custom_start_unix_time,
        duration=custom_duration,
        cprofile_timer=custom_cprofile_timer,
    )


@pytest.fixture
def custom_horovod_profiling_config(custom_start_unix_time):
    return HorovodProfilingConfig(start_unix_time=custom_start_unix_time)


@pytest.fixture
def custom_smdataparallel_profiling_config(custom_duration):
    return SMDataParallelProfilingConfig(duration=custom_duration)


@pytest.fixture
def framework_profile_with_custom_metrics_configs(
    custom_detailed_profiling_config,
    custom_dataloader_profiling_config,
    custom_python_profiling_config,
    custom_horovod_profiling_config,
    custom_smdataparallel_profiling_config,
):
    return FrameworkProfile(
        detailed_profiling_config=custom_detailed_profiling_config,
        dataloader_profiling_config=custom_dataloader_profiling_config,
        python_profiling_config=custom_python_profiling_config,
        horovod_profiling_config=custom_horovod_profiling_config,
        smdataparallel_profiling_config=custom_smdataparallel_profiling_config,
    )


@pytest.fixture
def framework_profile_with_only_custom_python_profiling_config(custom_python_profiling_config_2):
    return FrameworkProfile(python_profiling_config=custom_python_profiling_config_2)


@pytest.fixture
def framework_profile_with_custom_step_range(custom_start_step, custom_num_steps):
    return FrameworkProfile(start_step=custom_start_step, num_steps=custom_num_steps)


@pytest.fixture
def framework_profile_with_custom_time_range(custom_start_unix_time, custom_duration):
    return FrameworkProfile(start_unix_time=custom_start_unix_time, duration=custom_duration)


def _validate_profiling_parameter_conditions(profiling_parameters):
    regex = re.compile(".*")

    for key, val in profiling_parameters.items():
        assert isinstance(key, str)
        assert len(key) <= 256
        assert regex.match(key) is not None
        assert isinstance(val, str)
        assert len(val) <= 256
        assert regex.match(val) is not None


def test_default_profiler_config():
    profiler_config = ProfilerConfig()
    request_dict = profiler_config._to_request_dict()
    assert request_dict.get("S3OutputPath") is None
    assert "ProfilingParameters" not in request_dict


def test_profiler_config_with_default_framework_profile(default_framework_profile):
    profiler_config = ProfilerConfig(framework_profile_params=default_framework_profile)
    request_dict = profiler_config._to_request_dict()
    profiling_parameters = request_dict["ProfilingParameters"]

    _validate_profiling_parameter_conditions(profiling_parameters)
    assert profiling_parameters["LocalPath"] == BASE_FOLDER_DEFAULT
    assert int(profiling_parameters["RotateMaxFileSizeInBytes"]) == MAX_FILE_SIZE_DEFAULT
    assert (
        float(profiling_parameters["RotateFileCloseIntervalInSeconds"])
        == CLOSE_FILE_INTERVAL_DEFAULT
    )
    assert int(profiling_parameters["FileOpenFailThreshold"]) == FILE_OPEN_FAIL_THRESHOLD_DEFAULT

    detailed_profiling_config = eval(profiling_parameters[DETAILED_PROFILING_CONFIG_NAME])
    assert detailed_profiling_config == {
        "StartStep": DETAILED_PROFILING_START_STEP_DEFAULT,
        "NumSteps": PROFILING_NUM_STEPS_DEFAULT,
    }

    dataloader_profiling_config = eval(profiling_parameters[DATALOADER_PROFILING_CONFIG_NAME])
    assert dataloader_profiling_config == {
        "StartStep": DATALOADER_PROFILING_START_STEP_DEFAULT,
        "NumSteps": PROFILING_NUM_STEPS_DEFAULT,
        "MetricsRegex": ".*",
    }

    python_profiling_config = eval(profiling_parameters[PYTHON_PROFILING_CONFIG_NAME])
    assert python_profiling_config == {
        "StartStep": PYTHON_PROFILING_START_STEP_DEFAULT,
        "NumSteps": PYTHON_PROFILING_NUM_STEPS_DEFAULT,
        "ProfilerName": PythonProfiler.CPROFILE.value,
        "cProfileTimer": cProfileTimer.DEFAULT.value,
    }

    horovod_profiling_config = eval(profiling_parameters[HOROVOD_PROFILING_CONFIG_NAME])
    assert horovod_profiling_config == {
        "StartStep": HOROVOD_PROFILING_START_STEP_DEFAULT,
        "NumSteps": PROFILING_NUM_STEPS_DEFAULT,
    }

    smdataparallel_profiling_config = eval(
        profiling_parameters[SMDATAPARALLEL_PROFILING_CONFIG_NAME]
    )
    assert smdataparallel_profiling_config == {
        "StartStep": SMDATAPARALLEL_PROFILING_START_STEP_DEFAULT,
        "NumSteps": PROFILING_NUM_STEPS_DEFAULT,
    }


def test_default_detailed_profiling_config():
    detailed_profiling_config = DetailedProfilingConfig(profile_default_steps=True)
    assert isinstance(detailed_profiling_config.range, StepRange)
    assert detailed_profiling_config.range.start_step == DETAILED_PROFILING_START_STEP_DEFAULT
    assert detailed_profiling_config.range.num_steps == PROFILING_NUM_STEPS_DEFAULT


def test_default_dataloader_metrics_config():
    dataloader_metrics_config = DataloaderProfilingConfig(profile_default_steps=True)
    assert isinstance(dataloader_metrics_config.range, StepRange)
    assert dataloader_metrics_config.range.start_step == DATALOADER_PROFILING_START_STEP_DEFAULT
    assert dataloader_metrics_config.range.num_steps == PROFILING_NUM_STEPS_DEFAULT
    assert dataloader_metrics_config.metrics_regex == ".*"


def test_default_python_profiling_config():
    python_profiling_config = PythonProfilingConfig(profile_default_steps=True)
    assert isinstance(python_profiling_config.range, StepRange)
    assert python_profiling_config.range.start_step == PYTHON_PROFILING_START_STEP_DEFAULT
    assert python_profiling_config.range.num_steps == PYTHON_PROFILING_NUM_STEPS_DEFAULT
    assert python_profiling_config.python_profiler == PythonProfiler.CPROFILE
    assert python_profiling_config.cprofile_timer == cProfileTimer.DEFAULT


def test_default_horovod_profiling_config():
    horovod_profiling_config = HorovodProfilingConfig(profile_default_steps=True)
    assert isinstance(horovod_profiling_config.range, StepRange)
    assert horovod_profiling_config.range.start_step == HOROVOD_PROFILING_START_STEP_DEFAULT
    assert horovod_profiling_config.range.num_steps == PROFILING_NUM_STEPS_DEFAULT


def test_default_smdataparallel_profiling_config():
    smdataparallel_profiling_config = SMDataParallelProfilingConfig(profile_default_steps=True)
    assert isinstance(smdataparallel_profiling_config.range, StepRange)
    assert (
        smdataparallel_profiling_config.range.start_step
        == SMDATAPARALLEL_PROFILING_START_STEP_DEFAULT
    )
    assert smdataparallel_profiling_config.range.num_steps == PROFILING_NUM_STEPS_DEFAULT


def test_profiler_config_with_custom_trace_file_fields(
    framework_profile_with_custom_trace_file_fields,
    custom_local_path,
    custom_file_max_size,
    custom_file_close_interval,
    custom_file_open_fail_threshold,
):
    profiler_config = ProfilerConfig(
        framework_profile_params=framework_profile_with_custom_trace_file_fields
    )
    request_dict = profiler_config._to_request_dict()
    profiling_parameters = request_dict["ProfilingParameters"]

    _validate_profiling_parameter_conditions(profiling_parameters)
    assert profiling_parameters["LocalPath"] == custom_local_path
    assert int(profiling_parameters["RotateMaxFileSizeInBytes"]) == custom_file_max_size
    assert (
        float(profiling_parameters["RotateFileCloseIntervalInSeconds"])
        == custom_file_close_interval
    )
    assert int(profiling_parameters["FileOpenFailThreshold"]) == custom_file_open_fail_threshold

    detailed_profiling_config = eval(profiling_parameters[DETAILED_PROFILING_CONFIG_NAME])
    assert detailed_profiling_config == {
        "StartStep": DETAILED_PROFILING_START_STEP_DEFAULT,
        "NumSteps": PROFILING_NUM_STEPS_DEFAULT,
    }

    dataloader_profiling_config = eval(profiling_parameters[DATALOADER_PROFILING_CONFIG_NAME])
    assert dataloader_profiling_config == {
        "StartStep": DATALOADER_PROFILING_START_STEP_DEFAULT,
        "NumSteps": PROFILING_NUM_STEPS_DEFAULT,
        "MetricsRegex": ".*",
    }

    python_profiling_config = eval(profiling_parameters[PYTHON_PROFILING_CONFIG_NAME])
    assert python_profiling_config == {
        "StartStep": PYTHON_PROFILING_START_STEP_DEFAULT,
        "NumSteps": PYTHON_PROFILING_NUM_STEPS_DEFAULT,
        "ProfilerName": PythonProfiler.CPROFILE.value,
        "cProfileTimer": cProfileTimer.DEFAULT.value,
    }

    horovod_profiling_config = eval(profiling_parameters[HOROVOD_PROFILING_CONFIG_NAME])
    assert horovod_profiling_config == {
        "StartStep": HOROVOD_PROFILING_START_STEP_DEFAULT,
        "NumSteps": PROFILING_NUM_STEPS_DEFAULT,
    }

    smdataparallel_profiling_config = eval(
        profiling_parameters[SMDATAPARALLEL_PROFILING_CONFIG_NAME]
    )
    assert smdataparallel_profiling_config == {
        "StartStep": SMDATAPARALLEL_PROFILING_START_STEP_DEFAULT,
        "NumSteps": PROFILING_NUM_STEPS_DEFAULT,
    }


def test_profiler_config_with_custom_metrics_configs(
    framework_profile_with_custom_metrics_configs,
    framework_profile_with_only_custom_python_profiling_config,
    custom_start_step,
    custom_num_steps,
    custom_start_unix_time,
    custom_duration,
    custom_metrics_regex,
    custom_python_profiler,
    custom_cprofile_timer,
):
    profiler_config = ProfilerConfig(
        framework_profile_params=framework_profile_with_custom_metrics_configs
    )
    request_dict = profiler_config._to_request_dict()
    profiling_parameters = request_dict["ProfilingParameters"]

    _validate_profiling_parameter_conditions(profiling_parameters)
    assert profiling_parameters["LocalPath"] == BASE_FOLDER_DEFAULT
    assert int(profiling_parameters["RotateMaxFileSizeInBytes"]) == MAX_FILE_SIZE_DEFAULT
    assert (
        float(profiling_parameters["RotateFileCloseIntervalInSeconds"])
        == CLOSE_FILE_INTERVAL_DEFAULT
    )
    assert int(profiling_parameters["FileOpenFailThreshold"]) == FILE_OPEN_FAIL_THRESHOLD_DEFAULT

    detailed_profiling_config = eval(profiling_parameters[DETAILED_PROFILING_CONFIG_NAME])
    assert detailed_profiling_config == {
        "StartStep": custom_start_step,
        "NumSteps": custom_num_steps,
    }

    dataloader_profiling_config = eval(profiling_parameters[DATALOADER_PROFILING_CONFIG_NAME])
    assert dataloader_profiling_config == {
        "StartStep": custom_start_step,
        "NumSteps": PROFILING_NUM_STEPS_DEFAULT,
        "MetricsRegex": custom_metrics_regex,
    }

    python_profiling_config = eval(profiling_parameters[PYTHON_PROFILING_CONFIG_NAME])
    assert python_profiling_config == {
        "StartStep": START_STEP_DEFAULT,
        "NumSteps": custom_num_steps,
        "ProfilerName": custom_python_profiler.value,
    }

    horovod_profiling_config = eval(profiling_parameters[HOROVOD_PROFILING_CONFIG_NAME])
    assert horovod_profiling_config == {"StartTimeInSecSinceEpoch": custom_start_unix_time}

    smdataparallel_profiling_config = eval(
        profiling_parameters[SMDATAPARALLEL_PROFILING_CONFIG_NAME]
    )
    assert smdataparallel_profiling_config == {"Duration": custom_duration}

    profiler_config = ProfilerConfig(
        framework_profile_params=framework_profile_with_only_custom_python_profiling_config
    )
    request_dict = profiler_config._to_request_dict()
    profiling_parameters = request_dict["ProfilingParameters"]

    python_profiling_config = eval(profiling_parameters[PYTHON_PROFILING_CONFIG_NAME])
    assert python_profiling_config == {
        "StartTimeInSecSinceEpoch": custom_start_unix_time,
        "Duration": custom_duration,
        "ProfilerName": PythonProfiler.CPROFILE.value,
        "cProfileTimer": custom_cprofile_timer.value,
    }

    for config_name in [
        DETAILED_PROFILING_CONFIG_NAME,
        DATALOADER_PROFILING_CONFIG_NAME,
        HOROVOD_PROFILING_CONFIG_NAME,
        SMDATAPARALLEL_PROFILING_CONFIG_NAME,
    ]:
        assert config_name not in profiling_parameters


def test_custom_detailed_profiling_config(
    custom_detailed_profiling_config, custom_start_step, custom_num_steps
):
    assert isinstance(custom_detailed_profiling_config.range, StepRange)
    assert custom_detailed_profiling_config.range.start_step == custom_start_step
    assert custom_detailed_profiling_config.range.num_steps == custom_num_steps


def test_custom_dataloader_profiling_config(
    custom_dataloader_profiling_config, custom_start_step, custom_metrics_regex
):
    assert isinstance(custom_dataloader_profiling_config.range, StepRange)
    assert custom_dataloader_profiling_config.range.start_step == custom_start_step
    assert custom_dataloader_profiling_config.range.num_steps == PROFILING_NUM_STEPS_DEFAULT
    assert custom_dataloader_profiling_config.metrics_regex == custom_metrics_regex


def test_custom_python_profiling_config(
    custom_python_profiling_config, custom_num_steps, custom_python_profiler
):
    assert isinstance(custom_python_profiling_config.range, StepRange)
    assert custom_python_profiling_config.range.start_step == START_STEP_DEFAULT
    assert custom_python_profiling_config.range.num_steps == custom_num_steps
    assert custom_python_profiling_config.python_profiler == custom_python_profiler
    assert custom_python_profiling_config.cprofile_timer is None


def test_custom_python_profiling_config_2(
    custom_python_profiling_config_2, custom_start_unix_time, custom_duration, custom_cprofile_timer
):
    assert isinstance(custom_python_profiling_config_2.range, TimeRange)
    assert custom_python_profiling_config_2.range.start_unix_time == custom_start_unix_time
    assert custom_python_profiling_config_2.range.duration == custom_duration
    assert custom_python_profiling_config_2.python_profiler == PythonProfiler.CPROFILE
    assert custom_python_profiling_config_2.cprofile_timer == custom_cprofile_timer


def test_custom_horovod_profiling_config(custom_horovod_profiling_config, custom_start_unix_time):
    assert isinstance(custom_horovod_profiling_config.range, TimeRange)
    assert custom_horovod_profiling_config.range.start_unix_time == custom_start_unix_time
    assert custom_horovod_profiling_config.range.duration is None


def test_custom_smdataparallel_profiling_config(
    custom_smdataparallel_profiling_config, custom_duration
):
    assert isinstance(custom_smdataparallel_profiling_config.range, TimeRange)
    assert custom_smdataparallel_profiling_config.range.start_unix_time is None
    assert custom_smdataparallel_profiling_config.range.duration == custom_duration


def test_profiler_config_with_custom_step_range(custom_start_step, custom_num_steps):
    profiler_config = ProfilerConfig(
        framework_profile_params=FrameworkProfile(
            start_step=custom_start_step, num_steps=custom_num_steps
        )
    )
    request_dict = profiler_config._to_request_dict()
    profiling_parameters = request_dict["ProfilingParameters"]

    _validate_profiling_parameter_conditions(profiling_parameters)
    assert profiling_parameters["LocalPath"] == BASE_FOLDER_DEFAULT
    assert int(profiling_parameters["RotateMaxFileSizeInBytes"]) == MAX_FILE_SIZE_DEFAULT
    assert (
        float(profiling_parameters["RotateFileCloseIntervalInSeconds"])
        == CLOSE_FILE_INTERVAL_DEFAULT
    )
    assert int(profiling_parameters["FileOpenFailThreshold"]) == FILE_OPEN_FAIL_THRESHOLD_DEFAULT

    detailed_profiling_config = eval(profiling_parameters[DETAILED_PROFILING_CONFIG_NAME])
    assert detailed_profiling_config == {
        "StartStep": custom_start_step,
        "NumSteps": custom_num_steps,
    }

    dataloader_profiling_config = eval(profiling_parameters[DATALOADER_PROFILING_CONFIG_NAME])
    assert dataloader_profiling_config == {
        "StartStep": custom_start_step,
        "NumSteps": custom_num_steps,
        "MetricsRegex": ".*",
    }

    python_profiling_config = eval(profiling_parameters[PYTHON_PROFILING_CONFIG_NAME])
    assert python_profiling_config == {
        "StartStep": custom_start_step,
        "NumSteps": custom_num_steps,
        "ProfilerName": PythonProfiler.CPROFILE.value,
        "cProfileTimer": cProfileTimer.TOTAL_TIME.value,
    }

    horovod_profiling_config = eval(profiling_parameters[HOROVOD_PROFILING_CONFIG_NAME])
    assert horovod_profiling_config == {
        "StartStep": custom_start_step,
        "NumSteps": custom_num_steps,
    }

    smdataparallel_profiling_config = eval(
        profiling_parameters[SMDATAPARALLEL_PROFILING_CONFIG_NAME]
    )
    assert smdataparallel_profiling_config == {
        "StartStep": custom_start_step,
        "NumSteps": custom_num_steps,
    }


def test_profiler_config_with_custom_time_range(custom_start_unix_time, custom_duration):
    profiler_config = ProfilerConfig(
        framework_profile_params=FrameworkProfile(
            start_unix_time=custom_start_unix_time, duration=custom_duration
        )
    )
    request_dict = profiler_config._to_request_dict()
    profiling_parameters = request_dict["ProfilingParameters"]

    _validate_profiling_parameter_conditions(profiling_parameters)
    assert profiling_parameters["LocalPath"] == BASE_FOLDER_DEFAULT
    assert int(profiling_parameters["RotateMaxFileSizeInBytes"]) == MAX_FILE_SIZE_DEFAULT
    assert (
        float(profiling_parameters["RotateFileCloseIntervalInSeconds"])
        == CLOSE_FILE_INTERVAL_DEFAULT
    )
    assert int(profiling_parameters["FileOpenFailThreshold"]) == FILE_OPEN_FAIL_THRESHOLD_DEFAULT

    detailed_profiling_config = eval(profiling_parameters[DETAILED_PROFILING_CONFIG_NAME])
    assert detailed_profiling_config == {
        "StartTimeInSecSinceEpoch": custom_start_unix_time,
        "Duration": custom_duration,
    }

    dataloader_profiling_config = eval(profiling_parameters[DATALOADER_PROFILING_CONFIG_NAME])
    assert dataloader_profiling_config == {
        "StartTimeInSecSinceEpoch": custom_start_unix_time,
        "Duration": custom_duration,
        "MetricsRegex": ".*",
    }

    python_profiling_config = eval(profiling_parameters[PYTHON_PROFILING_CONFIG_NAME])
    assert python_profiling_config == {
        "StartTimeInSecSinceEpoch": custom_start_unix_time,
        "Duration": custom_duration,
        "ProfilerName": PythonProfiler.CPROFILE.value,
        "cProfileTimer": cProfileTimer.TOTAL_TIME.value,
    }

    horovod_profiling_config = eval(profiling_parameters[HOROVOD_PROFILING_CONFIG_NAME])
    assert horovod_profiling_config == {
        "StartTimeInSecSinceEpoch": custom_start_unix_time,
        "Duration": custom_duration,
    }

    smdataparallel_profiling_config = eval(
        profiling_parameters[SMDATAPARALLEL_PROFILING_CONFIG_NAME]
    )
    assert smdataparallel_profiling_config == {
        "StartTimeInSecSinceEpoch": custom_start_unix_time,
        "Duration": custom_duration,
    }


def test_validation():
    with pytest.raises(AssertionError, match=ErrorMessages.INVALID_LOCAL_PATH.value):
        FrameworkProfile(local_path=10)

    with pytest.raises(AssertionError, match=ErrorMessages.INVALID_FILE_MAX_SIZE.value):
        FrameworkProfile(file_max_size=100.5)

    with pytest.raises(AssertionError, match=ErrorMessages.INVALID_FILE_CLOSE_INTERVAL.value):
        FrameworkProfile(file_close_interval=-1)

    with pytest.raises(AssertionError, match=ErrorMessages.INVALID_FILE_OPEN_FAIL_THRESHOLD.value):
        FrameworkProfile(file_open_fail_threshold=2.3)

    with pytest.raises(AssertionError, match=ErrorMessages.INVALID_START_STEP.value):
        FrameworkProfile(start_step=6.8)

    with pytest.raises(AssertionError, match=ErrorMessages.INVALID_NUM_STEPS.value):
        FrameworkProfile(num_steps=0)

    with pytest.raises(AssertionError, match=ErrorMessages.INVALID_START_UNIX_TIME.value):
        FrameworkProfile(start_unix_time=99999999999999)

    with pytest.raises(AssertionError, match=ErrorMessages.INVALID_DURATION.value):
        FrameworkProfile(duration=0)

    with pytest.raises(AssertionError, match=ErrorMessages.FOUND_BOTH_STEP_AND_TIME_FIELDS.value):
        FrameworkProfile(start_step=5, duration=20)

    with pytest.raises(AssertionError, match=ErrorMessages.INVALID_METRICS_REGEX.value):
        DataloaderProfilingConfig(metrics_regex="*")

    with pytest.raises(AssertionError, match=ErrorMessages.INVALID_METRICS_REGEX.value):
        DataloaderProfilingConfig(metrics_regex=3)

    with pytest.raises(AssertionError, match=ErrorMessages.INVALID_PYTHON_PROFILER.value):
        PythonProfilingConfig(python_profiler="bad_python_profiler", profile_default_steps=True)

    with pytest.raises(AssertionError, match=ErrorMessages.INVALID_CPROFILE_TIMER.value):
        PythonProfilingConfig(cprofile_timer="bad_cprofile_timer")


def test_detail_profiler_processing_url():
    url = get_rule_container_image_uri("DetailedProfilerProcessingJobConfig", "us-west-2")
    assert url.endswith("detailed-profiler-processing:latest")


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SCRIPT_PATH = os.path.join(DATA_DIR, "dummy_script.py")
INSTANCE_COUNT = 1
ROLE = "Dummy"
REGION = "us-west-2"
INSTANCE_TYPE = "ml.p3.2xlarge"


def _check_framework_profile_deprecation_warning(framework_version, framework_name, warn_list):
    """Check the collected warnings for a framework fromfile DeprecationWarning"""

    thresh = version.parse("2.12") if framework_name == "tensorflow" else version.parse("2.0")
    actual = version.parse(framework_version)

    if actual >= thresh:
        # should find a Framework profiling deprecation warning
        for w in warn_list:
            if issubclass(w.category, DeprecationWarning):
                if "Framework profiling" in str(w.message):
                    return
        assert 0  # Should have found a deprecation and exited above


def test_create_pytorch_estimator_with_framework_profile(
    sagemaker_session,
    pytorch_inference_version,
    pytorch_inference_py_version,
    default_framework_profile,
):
    profiler_config = ProfilerConfig(framework_profile_params=default_framework_profile)

    with warnings.catch_warnings(record=True) as warn_list:
        warnings.simplefilter("always")
        framework_version = pytorch_inference_version
        pytorch = PyTorch(
            entry_point=SCRIPT_PATH,
            framework_version=framework_version,
            py_version=pytorch_inference_py_version,
            role=ROLE,
            sagemaker_session=sagemaker_session,
            instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE,
            base_job_name="job",
            profiler_config=profiler_config,
        )

        _check_framework_profile_deprecation_warning(
            framework_version, pytorch._framework_name, warn_list
        )


def test_create_pytorch_estimator_w_image_with_framework_profile(
    sagemaker_session,
    pytorch_inference_version,
    pytorch_inference_py_version,
    gpu_pytorch_instance_type,
    default_framework_profile,
):
    image_uri = image_uris.retrieve(
        "pytorch",
        REGION,
        version=pytorch_inference_version,
        py_version=pytorch_inference_py_version,
        instance_type=gpu_pytorch_instance_type,
        image_scope="inference",
    )

    profiler_config = ProfilerConfig(framework_profile_params=default_framework_profile)

    with warnings.catch_warnings(record=True) as warn_list:
        warnings.simplefilter("always")
        pytorch = PyTorch(
            entry_point=SCRIPT_PATH,
            role=ROLE,
            sagemaker_session=sagemaker_session,
            instance_count=INSTANCE_COUNT,
            instance_type=gpu_pytorch_instance_type,
            image_uri=image_uri,
            profiler_config=profiler_config,
        )

        framework_version = None
        _, _, image_tag, _ = fw.framework_name_from_image(image_uri)

        if image_tag is not None:
            framework_version = fw.framework_version_from_tag(image_tag)

        if framework_version is not None:
            _check_framework_profile_deprecation_warning(
                framework_version, pytorch._framework_name, warn_list
            )


def test_create_tf_estimator_with_framework_profile_212(
    sagemaker_session,
    default_framework_profile,
):
    profiler_config = ProfilerConfig(framework_profile_params=default_framework_profile)

    with warnings.catch_warnings(record=True) as warn_list:
        warnings.simplefilter("always")
        framework_version = "2.12"
        tf = TensorFlow(
            entry_point=SCRIPT_PATH,
            role=ROLE,
            framework_version=framework_version,
            py_version="py39",
            sagemaker_session=sagemaker_session,
            instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE,
            profiler_config=profiler_config,
        )

        _check_framework_profile_deprecation_warning(
            framework_version, tf._framework_name, warn_list
        )


def test_create_tf_estimator_w_image_with_framework_profile(
    sagemaker_session,
    tensorflow_inference_version,
    tensorflow_inference_py_version,
    default_framework_profile,
):
    image_uri = image_uris.retrieve(
        "tensorflow",
        REGION,
        version=tensorflow_inference_version,
        py_version=tensorflow_inference_py_version,
        instance_type=INSTANCE_TYPE,
        image_scope="inference",
    )

    profiler_config = ProfilerConfig(framework_profile_params=default_framework_profile)

    with warnings.catch_warnings(record=True) as warn_list:
        warnings.simplefilter("always")
        tf = TensorFlow(
            entry_point=SCRIPT_PATH,
            role=ROLE,
            sagemaker_session=sagemaker_session,
            instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE,
            image_uri=image_uri,
            profiler_config=profiler_config,
        )

        framework_version = None
        _, _, image_tag, _ = fw.framework_name_from_image(image_uri)

        if image_tag is not None:
            framework_version = fw.framework_version_from_tag(image_tag)

        if framework_version is not None:
            _check_framework_profile_deprecation_warning(
                framework_version, tf._framework_name, warn_list
            )


def test_create_pytorch_estimator_with_profile_processing(
    sagemaker_session,
    pytorch_training_version,
    pytorch_training_py_version,
    default_framework_profile,
):

    profiler_config = ProfilerConfig(
        system_monitor_interval_millis=500, profile_params=Profiler(cpu_profiling_duration="3600")
    )
    assert (
        profiler_config.profile_params.instanceType == DETAIL_PROF_PROCESSING_DEFAULT_INSTANCE_TYPE
    )
    assert (
        profiler_config.profile_params.volumeSizeInGB == DETAIL_PROF_PROCESSING_DEFAULT_VOLUME_SIZE
    )

    profiler_config = ProfilerConfig(
        system_monitor_interval_millis=500,
        profile_params=Profiler(
            cpu_profiling_duration="3600",
        ),
    )

    pytorch = PyTorch(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,  # gpu_pytorch_instance_type,
        framework_version=pytorch_training_version,
        py_version=pytorch_training_py_version,
        profiler_config=profiler_config,
    )
    pytorch._prepare_profiler_for_training()

    assert pytorch.profiler_rules is not None
    for rule in pytorch.profiler_rules:
        if rule.name is not None and rule.name.startswith("DetailedProfilerProcessingJobConfig"):
            if rule.image_uri.endswith("detailed-profiler-processing:latest"):
                return

    assert 0  # Should not happen. A rule should have been built with return above
