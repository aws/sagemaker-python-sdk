from smdebug.rules.rule import Rule


class CustomGradientRule(Rule):
    def __init__(self, base_trial, threshold=10.0):
        super().__init__(base_trial)
        self.threshold = float(threshold)

    def set_required_tensors(self, step):
        for tname in self.base_trial.tensor_names(collection="gradients"):
            self.req_tensors.add(tname, steps=[step])

    def invoke_at_step(self, step):
        return False
