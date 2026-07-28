from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice


@task
def boolq_pt():
    return Task(
        dataset=json_dataset("boolq_data.json"),
        solver=[multiple_choice()],
        scorer=choice(),
    )
