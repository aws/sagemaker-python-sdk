import argparse
import json
import os
import re

HYPERPARAMETERS_FILE_PATH = "/opt/ml/input/config/hyperparameters.json"


def set_intelligent_params(path: str) -> None:
    """
    Set intelligent parameters for all python files under the given path.
    For python code with comment sm_hyper_param or sm_hp_{variable_name}, the value will be found in
        /opt/ml/input/config/hyperparameters.json, and this function will rewrite lines with these comments.

    Args:
        path (str): The folder path to set intellingent parameters
    """
    with open(HYPERPARAMETERS_FILE_PATH, "r") as f:
        hyperparameters = json.load(f)
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                rewrite_file(file_path, hyperparameters)


def rewrite_file(file_path: str, hyperparameters: dict) -> None:
    """
    Rewrite a single python file with intelligent parameters.

    Args:
        file_path (str): The file path to rewrite
        hyperparameters (dict): The hyperparameter names and values
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for i in range(len(lines)):
        lines[i] = rewrite_line(lines[i], hyperparameters)
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def rewrite_line(line: str, hyperparameters: dict) -> None:
    """
    Rewrite a single line of python code with intelligent parameters.

    Args:
        line (str): The python code to rewrite
        hyperparameters (dict): The hyperparameter names and values
    """
    # Remove strings from the line to avoid = and # in strings
    line_without_strings = re.sub(r'".*?"', '""', line.strip())
    line_without_strings = re.sub(r"'.*?'", '""', line_without_strings)

    # Match lines with format "a = 1 # comment"
    assignment_pattern = r"^[a-zA-Z_][a-zA-Z0-9_]*\s*=.*#.*"
    if re.match(assignment_pattern, line_without_strings):
        indent = (len(line) - len(line.lstrip())) * " "
        variable = line_without_strings.split("=")[0].strip()
        comment = line_without_strings.split("#")[-1].strip()
        value = get_parameter_value(variable, comment, hyperparameters)
        if value is None:
            return line
        if isinstance(value, str):
            new_line = f'{indent}{variable} = "{value}" # set by intelligent parameters\n'
        else:
            new_line = f"{indent}{variable} = {str(value)} # set by intelligent parameters\n"
        return new_line
    return line


def get_parameter_value(variable: str, comment: str, hyperparameters: dict) -> None:
    """
    Get the parameter value by the variable name and comment.

    Args:
        variable (str): The variable name
        comment (str): The comment string in the python code
        hyperparameters (dict): The hyperparameter names and values
    """
    if comment == "sm_hyper_param":
        # Get the hyperparameter value by the variable name
        return hyperparameters.get(variable, None)
    if comment.startswith("sm_hp_"):
        # Get the hyperparameter value by the suffix of comment
        return hyperparameters.get(comment[6:], None)
    # Get the hyperparameter value from environment variables
    if comment.startswith("sm_"):
        return os.environ.get(comment.upper(), None)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intelligent parameters")
    parser.add_argument(
        "-p", "--path", help="The folder path to set intellingent parameters", required=True
    )

    args = parser.parse_args()

    set_intelligent_params(args.path)
