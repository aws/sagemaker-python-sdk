from sagemaker.core.utils.exceptions import TimeoutExceededError


class TestTimeoutExceededError:
    def test_default_message(self):
        """Default message should match original behavior."""
        err = TimeoutExceededError(resource_type="TrainingJob", status="InProgress")
        assert str(err) == (
            "Timeout exceeded while waiting for TrainingJob. "
            "Final Resource State: InProgress. "
            "Increase the timeout and try again."
        )

    def test_custom_message(self):
        """Custom message should replace the default."""
        err = TimeoutExceededError(
            resource_type="EvaluationJob",
            status="Executing",
            message="Your evaluation job is still running. Call .refresh() to check its current status.",
        )
        assert str(err) == (
            "Timeout exceeded while waiting for EvaluationJob. "
            "Final Resource State: Executing. "
            "Your evaluation job is still running. Call .refresh() to check its current status."
        )

    def test_default_params(self):
        """No args should use defaults without crashing."""
        err = TimeoutExceededError()
        assert "(Unkown)" in str(err)
        assert "Increase the timeout and try again." in str(err)
