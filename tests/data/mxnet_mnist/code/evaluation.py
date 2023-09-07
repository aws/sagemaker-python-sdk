import json
import pathlib

if __name__ == "__main__":
    # use static value for model evaluation metrics
    report_dict = {"metrics": {"f1": {"value": 0.7}, "mse": {"value": 5.8}}}

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
