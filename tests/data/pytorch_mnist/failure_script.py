def estimator_fn(run_config, params):
    """For use with integration tests expecting failures."""
    raise Exception('This failure is expected.')
