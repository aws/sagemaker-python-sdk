import unittest
from sagemaker.serve.spec.inference_spec import InferenceSpec


class ConcreteInferenceSpec(InferenceSpec):
    def load(self, model_dir):
        return "loaded_model"
    
    def invoke(self, input_object, model):
        return f"invoked with {input_object}"


class TestInferenceSpec(unittest.TestCase):
    def test_abstract_methods_must_be_implemented(self):
        with self.assertRaises(TypeError):
            InferenceSpec()

    def test_concrete_implementation(self):
        spec = ConcreteInferenceSpec()
        model = spec.load("/path/to/model")
        self.assertEqual(model, "loaded_model")
        
        result = spec.invoke("test_input", model)
        self.assertEqual(result, "invoked with test_input")

    def test_optional_methods(self):
        spec = ConcreteInferenceSpec()
        # These methods are optional and should not raise errors
        spec.preprocess("input")
        spec.postprocess("output")
        spec.prepare()
        spec.get_model()


if __name__ == "__main__":
    unittest.main()
