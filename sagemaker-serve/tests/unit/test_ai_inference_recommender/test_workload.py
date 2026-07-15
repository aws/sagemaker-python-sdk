# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
"""Unit tests for Workload."""
from __future__ import absolute_import

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from sagemaker.serve.ai_inference_recommender import Secret, Workload


class TestWorkloadSynthetic:
    def test_default_params(self):
        wl = Workload.synthetic(tokenizer="meta-llama/Llama-3.2-1B")
        assert wl.parameters == {
            "tokenizer": "meta-llama/Llama-3.2-1B",
            "concurrency": 1,
            "request_count": 100,
            "prompt_input_tokens_mean": 256,
            "prompt_input_tokens_stddev": 0.0,
            "output_tokens_mean": 256,
            "output_tokens_stddev": 0.0,
            "streaming": True,
        }
        assert wl.secrets == {}

    def test_overridden_params(self):
        wl = Workload.synthetic(
            tokenizer="meta-llama/Llama-3.2-1B",
            concurrency=10,
            request_count=500,
            prompt_input_tokens_mean=550,
            output_tokens_mean=150,
            streaming=False,
        )
        assert wl.parameters["concurrency"] == 10
        assert wl.parameters["request_count"] == 500
        assert wl.parameters["prompt_input_tokens_mean"] == 550
        assert wl.parameters["output_tokens_mean"] == 150
        assert wl.parameters["streaming"] is False

    def test_kwargs_passthrough_for_image_workloads(self):
        wl = Workload.synthetic(
            tokenizer="t",
            image_width_mean=640,
            image_height_mean=480,
            image_format="png",
        )
        assert wl.parameters["image_width_mean"] == 640
        assert wl.parameters["image_height_mean"] == 480
        assert wl.parameters["image_format"] == "png"

    def test_kwargs_passthrough_for_video_workloads(self):
        wl = Workload.synthetic(
            tokenizer="t",
            video_width=640,
            video_height=480,
            video_fps=4,
            video_duration=5.0,
        )
        assert wl.parameters["video_width"] == 640
        assert wl.parameters["video_fps"] == 4

    def test_kwargs_passthrough_for_multi_lora(self):
        wl = Workload.synthetic(tokenizer="t", model_selection_strategy="round_robin")
        assert wl.parameters["model_selection_strategy"] == "round_robin"

    def test_hf_token_as_secret(self):
        secret = Secret(arn="arn:aws:secretsmanager:us-east-1:123:secret:hf-AbCdEf")
        wl = Workload.synthetic(tokenizer="t", hf_token=secret)
        assert wl.secrets == {"hf_token": secret}
        assert "hf_token" not in wl.parameters

    def test_hf_token_as_arn_string(self):
        arn = "arn:aws:secretsmanager:us-east-1:123:secret:my-hf-AbCdEf"
        wl = Workload.synthetic(tokenizer="t", hf_token=arn)
        assert wl.secrets == {"hf_token": arn}

    def test_plaintext_hf_token_string_rejected(self):
        # A raw token (not an ARN) must be rejected so it is never persisted
        # in cleartext; the error points at Secret.from_string.
        with pytest.raises(ValueError, match="Secret.from_string"):
            Workload.synthetic(tokenizer="t", hf_token="hf_thisIsARawTokenNotAnArn")

    def test_sonnet_is_alias_for_synthetic(self):
        wl_sonnet = Workload.sonnet(tokenizer="meta-llama/Llama-3.2-1B")
        wl_synth = Workload.synthetic(tokenizer="meta-llama/Llama-3.2-1B")
        assert wl_sonnet.parameters == wl_synth.parameters
        assert wl_sonnet.secrets == wl_synth.secrets
        assert wl_sonnet.tooling == wl_synth.tooling


class TestWorkloadToInline:
    def test_envelope(self):
        wl = Workload.synthetic(tokenizer="meta-llama/Llama-3.2-1B")
        decoded = json.loads(wl.to_inline())
        assert decoded["benchmark"] == {"type": "aiperf"}
        assert decoded["tooling"] == {"api_standard": "openai"}
        assert decoded["parameters"]["tokenizer"] == "meta-llama/Llama-3.2-1B"
        assert "secrets" not in decoded

    def test_secrets_flatten_to_arns(self):
        secret = Secret(arn="arn:aws:secretsmanager:us-east-1:123:secret:hf-AbCdEf")
        wl = Workload.synthetic(tokenizer="t", hf_token=secret)
        decoded = json.loads(wl.to_inline())
        assert decoded["secrets"] == {"hf_token": secret.arn}

    def test_arn_string_secret_passes_through(self):
        arn = "arn:aws:secretsmanager:us-east-1:123:secret:my-hf-AbCdEf"
        wl = Workload.synthetic(tokenizer="t", hf_token=arn)
        decoded = json.loads(wl.to_inline())
        assert decoded["secrets"] == {"hf_token": arn}

    def test_inline_is_valid_json(self):
        wl = Workload.synthetic(tokenizer="t")
        assert json.loads(wl.to_inline())["parameters"]["tokenizer"] == "t"


class TestWorkloadFromDataset:
    def _wl(self, **overrides) -> Workload:
        kwargs = dict(
            s3_uri="s3://my-bucket/datasets/traffic/",
            custom_dataset_type="openai-chat",
            tokenizer="meta-llama/Llama-3.2-1B",
        )
        kwargs.update(overrides)
        return Workload.from_dataset(**kwargs)

    def test_parameters_carry_custom_dataset_type(self):
        wl = self._wl()
        assert wl.parameters["custom_dataset_type"] == "openai-chat"
        assert wl.parameters["tokenizer"] == "meta-llama/Llama-3.2-1B"
        assert wl.parameters["concurrency"] == 1
        assert wl.parameters["request_count"] == 100
        assert wl.parameters["streaming"] is True
        # input_file is not surfaced as a public kwarg; AIPerf reads from
        # the mounted directory by default.
        assert "input_file" not in wl.parameters

    def test_token_distribution_defaults_match_synthetic(self):
        wl = self._wl()
        assert wl.parameters["prompt_input_tokens_mean"] == 256
        assert wl.parameters["prompt_input_tokens_stddev"] == 0.0
        assert wl.parameters["output_tokens_mean"] == 256
        assert wl.parameters["output_tokens_stddev"] == 0.0

    def test_dataset_channel_recorded_with_default_name(self):
        wl = self._wl()
        assert len(wl.dataset_channels) == 1
        channel = wl.dataset_channels[0]
        assert channel.channel_name == "dataset"
        assert channel.s3_uri == "s3://my-bucket/datasets/traffic/"

    def test_channel_name_override(self):
        wl = self._wl(channel_name="traffic")
        assert wl.dataset_channels[0].channel_name == "traffic"
        assert wl.dataset_channels[0].s3_uri == "s3://my-bucket/datasets/traffic/"

    def test_inline_payload_omits_s3_uri(self):
        wl = self._wl()
        decoded = json.loads(wl.to_inline())
        assert "s3://" not in json.dumps(decoded)

    def test_optional_custom_dataset_type_omitted_when_none(self):
        wl = self._wl(custom_dataset_type=None)
        assert "custom_dataset_type" not in wl.parameters

    def test_input_file_passes_through_via_params(self):
        wl = self._wl(input_file="/opt/ml/input/data/dataset/requests.jsonl")
        assert wl.parameters["input_file"] == "/opt/ml/input/data/dataset/requests.jsonl"

    def test_kwargs_passthrough(self):
        wl = self._wl(request_rate=2.5, num_conversations=4)
        assert wl.parameters["request_rate"] == 2.5
        assert wl.parameters["num_conversations"] == 4

    def test_hf_token_as_secret(self):
        secret = Secret(arn="arn:aws:secretsmanager:us-east-1:123:secret:hf-AbCdEf")
        wl = self._wl(hf_token=secret)
        assert wl.secrets == {"hf_token": secret}

    def test_rejects_non_s3_uri(self):
        with pytest.raises(ValueError, match="s3://"):
            self._wl(s3_uri="https://example.com/data")


class TestWorkloadTemplate:
    def _wl(self, **overrides) -> Workload:
        kwargs = dict(
            template_s3_uri="s3://my-bucket/templates/endpoint_template.jinja",
            response_field="generated_text",
            tokenizer="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            concurrency=4,
            request_count=100,
            prompt_input_tokens_mean=100,
            output_tokens_mean=50,
            streaming=False,
        )
        kwargs.update(overrides)
        return Workload.template(**kwargs)

    def test_template_channel_recorded_with_default_name(self):
        wl = self._wl()
        assert len(wl.dataset_channels) == 1
        channel = wl.dataset_channels[0]
        assert channel.channel_name == "template"
        assert channel.s3_uri == "s3://my-bucket/templates/endpoint_template.jinja"

    def test_extra_inputs_wires_payload_template_and_response_field(self):
        wl = self._wl()
        assert wl.parameters["extra_inputs"] == (
            "payload_template:/opt/ml/input/data/template/endpoint_template.jinja "
            "response_field:generated_text"
        )

    def test_payload_template_path_tracks_channel_name(self):
        wl = self._wl(channel_name="tmpl")
        assert wl.dataset_channels[0].channel_name == "tmpl"
        assert (
            "payload_template:/opt/ml/input/data/tmpl/endpoint_template.jinja"
            in wl.parameters["extra_inputs"]
        )

    def test_response_field_omitted_when_none(self):
        wl = self._wl(response_field=None)
        assert (
            wl.parameters["extra_inputs"]
            == "payload_template:/opt/ml/input/data/template/endpoint_template.jinja"
        )
        assert "response_field" not in wl.parameters["extra_inputs"]

    def test_extra_inputs_appends_additional_pairs(self):
        wl = self._wl(extra_inputs="ignore_eos:true")
        assert wl.parameters["extra_inputs"].endswith("ignore_eos:true")
        assert "payload_template:" in wl.parameters["extra_inputs"]
        assert "response_field:generated_text" in wl.parameters["extra_inputs"]

    def test_token_distribution_and_traffic_params_carried(self):
        wl = self._wl()
        assert wl.parameters["concurrency"] == 4
        assert wl.parameters["request_count"] == 100
        assert wl.parameters["prompt_input_tokens_mean"] == 100
        assert wl.parameters["output_tokens_mean"] == 50
        assert wl.parameters["streaming"] is False
        assert wl.parameters["tokenizer"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    def test_tokenizer_omitted_when_none(self):
        wl = self._wl(tokenizer=None)
        assert "tokenizer" not in wl.parameters

    def test_kwargs_passthrough(self):
        wl = self._wl(request_rate=2.5, benchmark_duration=60)
        assert wl.parameters["request_rate"] == 2.5
        assert wl.parameters["benchmark_duration"] == 60

    def test_hf_token_as_secret(self):
        secret = Secret(arn="arn:aws:secretsmanager:us-east-1:123:secret:hf-AbCdEf")
        wl = self._wl(hf_token=secret)
        assert wl.secrets == {"hf_token": secret}
        assert "hf_token" not in wl.parameters

    def test_inline_payload_omits_s3_uri(self):
        wl = self._wl()
        decoded = json.loads(wl.to_inline())
        assert "s3://" not in json.dumps(decoded)
        # The template path AIPerf reads is the in-container mount, not S3.
        assert decoded["parameters"]["extra_inputs"].startswith(
            "payload_template:/opt/ml/input/data/"
        )

    def test_rejects_non_s3_uri(self):
        with pytest.raises(ValueError, match="s3://"):
            self._wl(template_s3_uri="https://example.com/template.jinja")

    def test_rejects_prefix_without_filename(self):
        with pytest.raises(ValueError, match="not a prefix"):
            self._wl(template_s3_uri="s3://my-bucket/templates/")

    def test_requires_exactly_one_of_template_or_s3_uri(self):
        # Neither provided.
        with pytest.raises(ValueError, match="exactly one"):
            Workload.template(response_field="generated_text")
        # Both provided.
        with pytest.raises(ValueError, match="exactly one"):
            Workload.template(
                request_template="{{ text }}",
                template_s3_uri="s3://my-bucket/templates/t.jinja",
            )

    @patch("sagemaker.core.s3.S3Uploader.upload_string_as_file_body")
    def test_inline_template_string_uploaded(self, mock_upload):
        uploaded_uri = (
            "s3://default-bucket/ai-inference-recommender/templates/abc/endpoint_template.jinja"
        )
        mock_upload.return_value = uploaded_uri
        session = MagicMock()
        wl = Workload.template(
            request_template='{"inputs": {{ text|tojson }}}',
            sagemaker_session=session,
            response_field="generated_text",
        )
        # Inline string was uploaded via the SDK uploader (no boto3 in caller code).
        assert mock_upload.called
        assert wl.dataset_channels[0].s3_uri == uploaded_uri
        # The container mount path tracks the uploaded filename.
        assert (
            "payload_template:/opt/ml/input/data/template/endpoint_template.jinja"
            in wl.parameters["extra_inputs"]
        )

    @pytest.mark.parametrize("path_kind", ["absolute", "relative", "dot_relative", "tilde"])
    @patch("sagemaker.core.s3.S3Uploader.upload")
    def test_local_path_template_uploaded(self, mock_upload, path_kind, tmp_path, monkeypatch):
        """A local template path is uploaded (not treated as inline) for every
        common path form: absolute, relative, ./relative, and ~-prefixed."""
        local = tmp_path / "my_template.jinja"
        local.write_text('{"inputs": {{ text|tojson }}}')

        if path_kind == "absolute":
            arg = str(local)
        elif path_kind == "relative":
            monkeypatch.chdir(tmp_path)
            arg = "my_template.jinja"
        elif path_kind == "dot_relative":
            monkeypatch.chdir(tmp_path)
            arg = os.path.join(".", "my_template.jinja")
        else:  # tilde — home-relative; os.path.isfile won't see it without expansion
            monkeypatch.setenv("HOME", str(tmp_path))
            monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", str(tmp_path), 1))
            arg = "~/my_template.jinja"

        uploaded_uri = (
            "s3://default-bucket/ai-inference-recommender/templates/xyz/my_template.jinja"
        )
        mock_upload.return_value = uploaded_uri
        session = MagicMock()
        wl = Workload.template(
            request_template=arg,
            sagemaker_session=session,
            response_field="generated_text",
        )
        # The path was recognized as a file and uploaded (NOT sent as inline text).
        assert mock_upload.called, f"{path_kind} path was not uploaded as a file"
        # The expanded local path (no leading ~) is what gets uploaded.
        passed = mock_upload.call_args.kwargs["local_path"]
        assert "~" not in passed
        assert os.path.isfile(passed)
        assert wl.dataset_channels[0].s3_uri == uploaded_uri
        assert (
            "payload_template:/opt/ml/input/data/template/my_template.jinja"
            in wl.parameters["extra_inputs"]
        )

    @patch("sagemaker.core.s3.S3Uploader.upload")
    @patch("sagemaker.core.s3.S3Uploader.upload_string_as_file_body")
    def test_inline_string_not_mistaken_for_path(self, mock_inline, mock_file_upload):
        """A Jinja2 string that isn't an existing file is uploaded as inline
        content, not via the file-upload path."""
        mock_inline.return_value = (
            "s3://b/ai-inference-recommender/templates/z/endpoint_template.jinja"
        )
        Workload.template(
            request_template='{"inputs": {{ text|tojson }}}',
            sagemaker_session=MagicMock(),
        )
        assert mock_inline.called
        assert not mock_file_upload.called
