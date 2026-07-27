"""Helper module to verify cross-file imports work from S3 source_dir."""


def get_greeting(name: str) -> str:
    """Return a greeting string."""
    return f"Hello from S3 source_dir, {name}!"
