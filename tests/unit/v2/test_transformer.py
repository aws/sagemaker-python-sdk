from __future__ import absolute_import

import ast
from sagemaker.tools.compatibility.v2.ast_transformer import ASTTransformer
import pasta


def test_code_needs_transform():
    simple = """
TensorFlow(entry_point="foo.py")
sagemaker.tensorflow.TensorFlow()
m = MXNet()
sagemaker.mxnet.MXNet()
"""

    transformer_class = ASTTransformer()
    rewrite = transformer_class.visit(ast.parse(simple))
    expected = """TensorFlow(entry_point='foo.py', framework_version='1.11.0')
sagemaker.tensorflow.TensorFlow(framework_version='1.11.0')
m = MXNet(framework_version='1.2.0')
sagemaker.mxnet.MXNet(framework_version='1.2.0')\n"""

    assert pasta.dump(rewrite) == expected


def test_code_does_not_need_transform():
    simple = """TensorFlow(entry_point='foo.py', framework_version='1.11.0')
sagemaker.tensorflow.TensorFlow(framework_version='1.11.0')
m = MXNet(framework_version='1.2.0')
sagemaker.mxnet.MXNet(framework_version='1.2.0')\n"""
    transformer_class = ASTTransformer()
    rewrite = transformer_class.visit(ast.parse(simple))
    expected = """TensorFlow(entry_point='foo.py', framework_version='1.11.0')
sagemaker.tensorflow.TensorFlow(framework_version='1.11.0')
m = MXNet(framework_version='1.2.0')
sagemaker.mxnet.MXNet(framework_version='1.2.0')\n"""

    assert pasta.dump(rewrite) == expected
