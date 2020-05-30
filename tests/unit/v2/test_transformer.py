import ast
import unittest

from tests.unit.v2.utils import get_sample_file
from tools.compatibility.v2.ast_transformer import ASTTransformer
import pasta


class TransformerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.transformer_class = ASTTransformer()

    def test_simple_transform(self):
        sample = get_sample_file('simple.txt')
        rewrite = self.transformer_class.visit(
            ast.parse(
                sample
            )
        )

        expected = """TensorFlow(entry_point='foo.py', framework_version='1.11.0')
sagemaker.tensorflow.TensorFlow(framework_version='1.11.0')
m = MXNet(framework_version='1.2.0')
sagemaker.mxnet.MXNet(framework_version='1.2.0')\n"""

        self.assertEqual(pasta.dump(rewrite), expected)


if __name__ == '__main__':
    unittest.main()


