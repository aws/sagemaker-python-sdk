"""Namespace package for SageMaker."""

__path__ = __import__("pkgutil").extend_path(__path__, __name__)

# Register the fallback finder that gives actionable migration guidance for v2
# modules removed in v3. Doing it here (in the namespace package init, which
# runs on any ``import sagemaker.*``) ensures the guidance is active even when a
# removed module is the very first sagemaker import in the process (e.g.
# ``import sagemaker.estimator``), before ``sagemaker.core`` is otherwise loaded.
# Fully guarded so it can never interfere with importing the package.
try:
    from sagemaker.core.deprecations import register_removed_module_finder as _register

    _register()
except Exception:  # pylint: disable=W0703  # noqa: E722
    pass
