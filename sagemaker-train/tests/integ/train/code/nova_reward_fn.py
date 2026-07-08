from typing import List
import json
import random

from dataclasses import asdict, dataclass

@dataclass
class RewardOutput:
    """Reward service."""

    id: str
    aggregate_reward_score: float

def lambda_handler(event, context):

    scores: List[RewardOutput] = []

    samples = event

    for sample in samples:
        # Extract the ground truth key. In the current dataset it's answer
        print("Sample: ", json.dumps(sample, indent=2))
        ground_truth = sample["reference_answer"]
        
        idx = "no id"
        # print(sample)
        if not "id" in sample:
            print(f"ID is None/empty for sample: {sample}")
        else:
            idx = sample["id"]

        ro = RewardOutput(id=idx, aggregate_reward_score=0.0)

        if not "messages" in sample:
            print(f"Messages is None/empty for id: {idx}")
            # scores.append(RewardOutput(id="0", aggregate_reward_score=0.0))
            continue
        
        # Extract answer from ground truth dict
        if ground_truth is None:
            print(f"No answer found in ground truth for id: {idx}")
            # scores.append(RewardOutput(id="0", aggregate_reward_score=0.0))
            continue
        
        # Get completion from last message (assistant message)
        last_message = sample["messages"][-1]
        # completion_text = last_message["content"]
        
        if last_message["role"] != "assistant":
            print(f"Last message is not from assistant for id: {idx}")
            # scores.append(RewardOutput(id="0", aggregate_reward_score=0.0))
            continue

        if not "content" in last_message:
            print(f"Completion text is empty for id: {idx}")
            # scores.append(RewardOutput(id="0", aggregate_reward_score=0.0))
            continue

        random_score = random.uniform(0.0, 1.0)
        ro = RewardOutput(id=idx, aggregate_reward_score=random_score)

        print(f"Response for id: {idx} is {ro}")
        scores.append(ro)

    return [asdict(score) for score in scores]
