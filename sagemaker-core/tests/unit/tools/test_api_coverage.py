import json
import os
import pytest

from sagemaker.core.tools.constants import API_COVERAGE_JSON_FILE_PATH, SERVICE_JSON_FILE_PATH
from sagemaker.core.tools.resources_extractor import ResourcesExtractor


class TestAPICoverage:
    @pytest.mark.skipif(
        not os.path.exists(API_COVERAGE_JSON_FILE_PATH) or not os.path.exists(SERVICE_JSON_FILE_PATH),
        reason="API coverage file or service JSON files not found - this test requires source files"
    )
    def test_api_coverage(self):
        with open(API_COVERAGE_JSON_FILE_PATH, "r") as file:
            coverage_json = json.load(file)
            previous_supported_apis = coverage_json["SupportedAPIs"]
            previous_unsupported_apis = coverage_json["UnsupportedAPIs"]
        resources_extractor = ResourcesExtractor()
        current_supported_apis = len(resources_extractor.actions_under_resource)
        current_unsupported_apis = len(resources_extractor.actions)
        # Check the numbers of current and previous apis being the same here
        # to ensure that developers update api_coverage.json when updating codegen
        assert current_supported_apis == previous_supported_apis
        assert current_unsupported_apis == previous_unsupported_apis
