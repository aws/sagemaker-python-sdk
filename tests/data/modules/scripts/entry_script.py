import json
import os
import time


def main():
    hps = json.loads(os.environ["SM_HPS"])
    assert hps != None
    print(f"Hyperparameters: {hps}")

    print("Running pseudo training script")
    for epochs in range(hps["epochs"]):
        print(f"Epoch: {epochs}")
        time.sleep(1)
    print("Finished running pseudo training script")


if __name__ == "__main__":
    main()
