import json
import unittest
from unittest.mock import patch

from sagemaker.train.common_utils.data_utils import is_multimodal_data

PATCH_LOAD = "sagemaker.train.common_utils.data_utils.load_file_content"


def _jsonl_lines(records):
    """Return a list of JSON strings, one per record, as load_file_content would yield."""
    return [json.dumps(r) for r in records]


class TestIsMultimodalData(unittest.TestCase):
    """Test multimodal data detection functionality."""

    @patch(PATCH_LOAD)
    def test_detects_image_content(self, mock_load):
        """Test detection of image content in messages."""
        data = [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"text": "What's in this image?"},
                            {"image": {"format": "png", "source": {"bytes": "..."}}},
                        ],
                    }
                ]
            }
        ]
        mock_load.return_value = iter(_jsonl_lines(data))
        self.assertTrue(is_multimodal_data("s3://bucket/data.jsonl"))

    @patch(PATCH_LOAD)
    def test_detects_video_content(self, mock_load):
        """Test detection of video content in messages."""
        data = [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"text": "Describe this video"},
                            {
                                "video": {
                                    "format": "mp4",
                                    "source": {"s3Location": {"uri": "s3://..."}},
                                }
                            },
                        ],
                    }
                ]
            }
        ]
        mock_load.return_value = iter(_jsonl_lines(data))
        self.assertTrue(is_multimodal_data("s3://bucket/data.jsonl"))

    @patch(PATCH_LOAD)
    def test_detects_document_content(self, mock_load):
        """Test detection of document content in messages."""
        data = [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"text": "Summarize this document"},
                            {
                                "document": {
                                    "format": "pdf",
                                    "name": "doc.pdf",
                                    "source": {"bytes": "..."},
                                }
                            },
                        ],
                    }
                ]
            }
        ]
        mock_load.return_value = iter(_jsonl_lines(data))
        self.assertTrue(is_multimodal_data("s3://bucket/data.jsonl"))

    @patch(PATCH_LOAD)
    def test_text_only_returns_false(self, mock_load):
        """Test that text-only data returns False."""
        data = [
            {
                "messages": [
                    {"role": "user", "content": [{"text": "Hello"}]},
                    {"role": "assistant", "content": [{"text": "Hi there"}]},
                ]
            }
        ]
        mock_load.return_value = iter(_jsonl_lines(data))
        self.assertFalse(is_multimodal_data("s3://bucket/data.jsonl"))

    @patch(PATCH_LOAD)
    def test_early_return_on_multimodal_found(self, mock_load):
        """Test that function returns True as soon as multimodal content is found."""
        data = [
            {"messages": [{"role": "user", "content": [{"image": {"format": "png"}}]}]},
            {"messages": [{"role": "user", "content": [{"text": "Hello"}]}]},
        ]
        mock_load.return_value = iter(_jsonl_lines(data))
        self.assertTrue(is_multimodal_data("s3://bucket/data.jsonl"))

    @patch(PATCH_LOAD)
    def test_handles_malformed_json(self, mock_load):
        """Test that malformed JSON lines are skipped gracefully."""
        mock_load.return_value = iter(
            [
                '{"messages": [{"role": "user", "content": [{"text": "Hello"}]}]}',
                "invalid json",
                '{"messages": [{"role": "user", "content": [{"image": {"format": "png"}}]}]}',
            ]
        )
        self.assertTrue(is_multimodal_data("s3://bucket/data.jsonl"))

    @patch(PATCH_LOAD)
    def test_handles_empty_file(self, mock_load):
        """Test that empty file returns False."""
        mock_load.return_value = iter([])
        self.assertFalse(is_multimodal_data("s3://bucket/data.jsonl"))

    @patch(PATCH_LOAD)
    def test_handles_s3_exception(self, mock_load):
        """Test that S3 exceptions are handled gracefully."""
        mock_load.side_effect = Exception("S3 error")
        self.assertFalse(is_multimodal_data("s3://bucket/data.jsonl"))

    @patch(PATCH_LOAD)
    def test_no_false_positive_on_string_content_items(self, mock_load):
        """String content items containing 'image'/'video'/'document' must not trigger detection."""
        data = [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            "describe this document",  # plain string, not a dict
                            "image of a cat",
                        ],
                    }
                ]
            }
        ]
        mock_load.return_value = iter(_jsonl_lines(data))
        self.assertFalse(is_multimodal_data("s3://bucket/data.jsonl"))

    @patch(PATCH_LOAD)
    def test_handles_null_content_field(self, mock_load):
        """Records with content: null must not raise TypeError."""
        data = [
            {"messages": [{"role": "user", "content": None}]},
            {"messages": [{"role": "user", "content": [{"image": {"format": "png"}}]}]},
        ]
        mock_load.return_value = iter(_jsonl_lines(data))
        self.assertTrue(is_multimodal_data("s3://bucket/data.jsonl"))

    @patch(PATCH_LOAD)
    def test_handles_null_messages_field(self, mock_load):
        """Records with messages: null must not raise TypeError — scan continues to next record."""
        data = [
            {"messages": None},
            {"messages": [{"role": "user", "content": [{"image": {"format": "png"}}]}]},
        ]
        mock_load.return_value = iter(_jsonl_lines(data))
        self.assertTrue(is_multimodal_data("s3://bucket/data.jsonl"))

    @patch(PATCH_LOAD)
    def test_handles_string_message_items(self, mock_load):
        """String items in messages list must not raise AttributeError — scan continues."""
        data = [
            {
                "messages": [
                    "not a dict",
                    {"role": "user", "content": [{"image": {"format": "png"}}]},
                ]
            },
        ]
        mock_load.return_value = iter(_jsonl_lines(data))
        self.assertTrue(is_multimodal_data("s3://bucket/data.jsonl"))

    # --- .json format tests ---

    @patch(PATCH_LOAD)
    def test_json_format_detects_multimodal(self, mock_load):
        """A .json array file with multimodal content is detected correctly."""
        data = [
            {"messages": [{"role": "user", "content": [{"text": "hello"}]}]},
            {"messages": [{"role": "user", "content": [{"image": {"format": "png"}}]}]},
        ]
        # load_file_content for .json yields lines of the serialized JSON
        mock_load.return_value = iter([json.dumps(data)])
        self.assertTrue(is_multimodal_data("s3://bucket/data.json"))

    @patch(PATCH_LOAD)
    def test_json_format_text_only_returns_false(self, mock_load):
        """A .json array file with text-only content returns False."""
        data = [{"messages": [{"role": "user", "content": [{"text": "hello"}]}]}]
        mock_load.return_value = iter([json.dumps(data)])
        self.assertFalse(is_multimodal_data("s3://bucket/data.json"))

    @patch(PATCH_LOAD)
    def test_json_format_single_object(self, mock_load):
        """A .json file containing a single object (not array) is handled."""
        data = {"messages": [{"role": "user", "content": [{"video": {"format": "mp4"}}]}]}
        mock_load.return_value = iter([json.dumps(data)])
        self.assertTrue(is_multimodal_data("s3://bucket/data.json"))

    @patch(PATCH_LOAD)
    def test_load_file_content_called_with_correct_args_jsonl(self, mock_load):
        """Verifies load_file_content is called with utf-8-sig encoding for .jsonl."""
        mock_load.return_value = iter([])
        is_multimodal_data("s3://bucket/data.jsonl")
        mock_load.assert_called_once_with(
            "s3://bucket/data.jsonl", extension=".jsonl", encoding="utf-8-sig"
        )

    @patch(PATCH_LOAD)
    def test_load_file_content_called_with_correct_args_json(self, mock_load):
        """Verifies load_file_content is called with utf-8 encoding for .json."""
        mock_load.return_value = iter([json.dumps([])])
        is_multimodal_data("s3://bucket/data.json")
        mock_load.assert_called_once_with(
            "s3://bucket/data.json", extension=".json", encoding="utf-8"
        )


class TestValidateDataPathExists(unittest.TestCase):
    """Tests for validate_data_path_exists utility."""

    def _make_session(self):
        from unittest.mock import Mock
        session = Mock()
        self.s3 = Mock()
        session.boto_session.client.return_value = self.s3
        from botocore.exceptions import ClientError
        self.s3.exceptions.ClientError = ClientError
        return session

    def test_object_exists(self):
        from sagemaker.train.common_utils.data_utils import validate_data_path_exists

        session = self._make_session()
        self.s3.list_objects_v2.return_value = {"KeyCount": 1}

        validate_data_path_exists("s3://my-bucket/data/train.jsonl", session, label="training")
        self.s3.list_objects_v2.assert_called_once_with(
            Bucket="my-bucket", Prefix="data/train.jsonl", MaxKeys=1
        )

    def test_prefix_exists(self):
        from sagemaker.train.common_utils.data_utils import validate_data_path_exists

        session = self._make_session()
        self.s3.list_objects_v2.return_value = {"KeyCount": 3}

        validate_data_path_exists("s3://my-bucket/data/prefix/", session, label="training")

    def test_path_does_not_exist_raises(self):
        from sagemaker.train.common_utils.data_utils import validate_data_path_exists

        session = self._make_session()
        self.s3.list_objects_v2.return_value = {"KeyCount": 0}

        with self.assertRaises(ValueError) as ctx:
            validate_data_path_exists(
                "s3://my-bucket/bad/path.jsonl", session, label="training dataset"
            )
        self.assertIn("does not exist", str(ctx.exception))

    def test_access_denied_warns_not_raises(self):
        from sagemaker.train.common_utils.data_utils import validate_data_path_exists
        from botocore.exceptions import ClientError

        session = self._make_session()
        self.s3.list_objects_v2.side_effect = ClientError(
            {"Error": {"Code": "403", "Message": "Forbidden"}}, "ListObjectsV2"
        )

        # Should not raise — caller may lack access but execution role might have it
        validate_data_path_exists("s3://locked-bucket/data.jsonl", session, label="data")

    def test_unrecognized_format_raises(self):
        from sagemaker.train.common_utils.data_utils import validate_data_path_exists
        from unittest.mock import Mock

        session = Mock()
        with self.assertRaises(ValueError) as ctx:
            validate_data_path_exists(
                "arn:aws:sagemaker:us-east-1:123:dataset/foo", session
            )
        self.assertIn("Invalid", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
