import json
import os
import subprocess
import sys


def main():
    driver_config = json.loads(os.environ["SM_DISTRIBUTED_CONFIG"])
    process_count_per_node = driver_config["process_count_per_node"]
    assert process_count_per_node != None

    hps = json.loads(os.environ["SM_HPS"])
    assert hps != None
    assert isinstance(hps, dict)

    source_dir = os.environ["SM_SOURCE_DIR"]
    assert source_dir == "/opt/ml/input/data/code"
    sm_drivers_dir = os.environ["SM_DRIVER_DIR"]
    assert sm_drivers_dir == "/opt/ml/input/data/sm_drivers/drivers"

    entry_script = os.environ["SM_ENTRY_SCRIPT"]
    assert entry_script != None

    python = sys.executable

    command = [python, entry_script]
    print(f"Running command: {command}")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    print("Running custom driver script")
    main()
    print("Finished running custom driver script")
