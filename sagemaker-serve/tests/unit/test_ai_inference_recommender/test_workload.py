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
