SUPPORTED_OPTIMIZATION_CONFIGURATIONS = {
    "trt": {
        "supported_instance_families": ["p4d", "p4de", "p5", "g5", "g6"],
        "compilation": True,
        "quantization": {
            "awq": True,
            "fp8": True,
            "gptq": False,
            "smooth_quant": True
        },
        "speculative_decoding": False,
        "sharding": False
    },
    "vllm": {
        "supported_instance_families": ["p4d", "p4de", "p5", "g5", "g6"],
        "compilation": False,
        "quantization": {
            "awq": True,
            "fp8": True,
            "gptq": False,
            "smooth_quant": False
        },
        "speculative_decoding": True,
        "sharding": True
    },
    "neuron": {
        "supported_instance_families": ["inf2", "trn1", "trn1n"],
        "compilation": True,
        "quantization": {
            "awq": False,
            "fp8": False,
            "gptq": False,
            "smooth_quant": False
        },
        "speculative_decoding": False,
        "sharding": False
    }
}
