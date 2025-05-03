import json


def main():
    """
    Every run of the pytest-cov command creates the report of coverages per class into a json file - i.e coverage.json
    This file is being parsed to print the coverage for resources.py which will be used as a tracking metric
    """
    json_file = "coverage.json"

    with open(json_file, "r") as f:
        data = json.load(f)
        print(data["files"]["src/sagemaker/core/resources.py"]["summary"]["percent_covered"])


if __name__ == "__main__":
    main()
