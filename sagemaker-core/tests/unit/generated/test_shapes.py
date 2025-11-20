import ast
import unittest

from pydantic import BaseModel, ValidationError

import os
from sagemaker.core.shapes import Base, AdditionalS3DataSource
from sagemaker.core.utils.utils import Unassigned

# Use the installed package location
import sagemaker.core.shapes
FILE_NAME = os.path.join(os.path.dirname(os.path.abspath(sagemaker.core.shapes.__file__)), "shapes.py")


class TestGeneratedShape(unittest.TestCase):
    def test_generated_shapes_have_pydantic_enabled(self):
        # This test ensures that all main shapes inherit Base which inherits BaseModel, thereby forcing pydantic validiation
        assert issubclass(Base, BaseModel)
        assert (
            self._fetch_number_of_classes_in_file_not_inheriting_a_class(FILE_NAME, "Base") == 1
        )  # 1 Because Base class itself does not inherit

    def test_pydantic_validation_for_generated_class_success(self):
        additional_s3_data_source = AdditionalS3DataSource(
            s3_data_type="filestring", s3_uri="s3/uri"
        )
        assert isinstance(additional_s3_data_source.s3_data_type, str)
        assert isinstance(additional_s3_data_source.s3_uri, str)
        assert isinstance(additional_s3_data_source.compression_type, Unassigned)

    def test_pydantic_validation_for_generated_class_success_with_optional_attributes_provided(
        self,
    ):
        additional_s3_data_source = AdditionalS3DataSource(
            s3_data_type="filestring", s3_uri="s3/uri", compression_type="zip"
        )
        assert isinstance(additional_s3_data_source.s3_data_type, str)
        assert isinstance(additional_s3_data_source.s3_uri, str)
        assert isinstance(additional_s3_data_source.compression_type, str)

    def test_pydantic_validation_for_generated_class_throws_error_for_incorrect_input(
        self,
    ):
        with self.assertRaises(ValidationError):
            AdditionalS3DataSource(s3_data_type="str", s3_uri=12)

    def _fetch_number_of_classes_in_file_not_inheriting_a_class(
        self, filepath: str, base_class_name: str
    ):
        count = 0
        with open(filepath, "r") as file:
            tree = ast.parse(file.read(), filename=filepath)
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    if not any(base_class.id == base_class_name for base_class in node.bases):
                        count = count + 1
        return count
