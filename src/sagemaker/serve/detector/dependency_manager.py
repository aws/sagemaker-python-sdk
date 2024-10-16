import logging
import subprocess
import sys
import re
from pathlib import Path

_SUPPORTED_SUFFIXES = [".txt"]
PKL_FILE_NAME = "serve.pkl"

logger = logging.getLogger(__name__)


def capture_dependencies(dependencies: dict, work_dir: Path, capture_all: bool = False):
    """Capture dependencies and print output."""
    print(f"Capturing dependencies: {dependencies}, work_dir: {work_dir}, capture_all: {capture_all}")
    
    path = work_dir.joinpath("requirements.txt")
    if "auto" in dependencies and dependencies["auto"]:
        command = [
            sys.executable,
            Path(__file__).parent.joinpath("pickle_dependencies.py"),
            "--pkl_path",
            work_dir.joinpath(PKL_FILE_NAME),
            "--dest",
            path,
        ]

        if capture_all:
            command.append("--capture_all")
        
        print(f"Running subprocess with command: {command}")

        subprocess.run(
            command,
            env={"SETUPTOOLS_USE_DISTUTILS": "stdlib"},
            check=True,
        )

        with open(path, "r") as f:
            autodetect_depedencies = f.read().splitlines()
        autodetect_depedencies.append("sagemaker[huggingface]>=2.199")
        print(f"Auto-detected dependencies: {autodetect_depedencies}")
    else:
        autodetect_depedencies = ["sagemaker[huggingface]>=2.199"]
        print(f"No auto-detection, using default dependencies: {autodetect_depedencies}")

    module_version_dict = _parse_dependency_list(autodetect_depedencies)
    print(f"Parsed auto-detected dependencies: {module_version_dict}")

    if "requirements" in dependencies:
        module_version_dict = _process_customer_provided_requirements(
            requirements_file=dependencies["requirements"], module_version_dict=module_version_dict
        )
        print(f"After processing customer-provided requirements: {module_version_dict}")
    
    if "custom" in dependencies:
        module_version_dict = _process_custom_dependencies(
            custom_dependencies=dependencies.get("custom"), module_version_dict=module_version_dict
        )
        print(f"After processing custom dependencies: {module_version_dict}")
    
    with open(path, "w") as f:
        for module, version in module_version_dict.items():
            f.write(f"{module}{version}\n")
    print(f"Final dependencies written to {path}")


def _process_custom_dependencies(custom_dependencies: list, module_version_dict: dict):
    """Process custom dependencies and print output."""
    print(f"Processing custom dependencies: {custom_dependencies}")
    
    custom_module_version_dict = _parse_dependency_list(custom_dependencies)
    print(f"Parsed custom dependencies: {custom_module_version_dict}")
    
    module_version_dict.update(custom_module_version_dict)
    print(f"Updated module_version_dict with custom dependencies: {module_version_dict}")
    
    return module_version_dict


def _process_customer_provided_requirements(requirements_file: str, module_version_dict: dict):
    """Process customer-provided requirements and print output."""
    print(f"Processing customer-provided requirements from file: {requirements_file}")
    
    requirements_file = Path(requirements_file)
    if not requirements_file.is_file() or not _is_valid_requirement_file(requirements_file):
        raise Exception(f"Path: {requirements_file} to requirements.txt doesn't exist")
    
    logger.debug("Packaging provided requirements.txt from %s", requirements_file)
    with open(requirements_file, "r") as f:
        custom_dependencies = f.read().splitlines()
    
    print(f"Customer-provided dependencies: {custom_dependencies}")

    module_version_dict.update(_parse_dependency_list(custom_dependencies))
    print(f"Updated module_version_dict with customer-provided requirements: {module_version_dict}")
    
    return module_version_dict


def _is_valid_requirement_file(path):
    """Check if the requirements file is valid and print result."""
    print(f"Validating requirement file: {path}")
    
    for suffix in _SUPPORTED_SUFFIXES:
        if path.name.endswith(suffix):
            print(f"File {path} is valid with suffix {suffix}")
            return True
    
    print(f"File {path} is not valid")
    return False


def _parse_dependency_list(depedency_list: list) -> dict:
    """Parse the dependency list and print output."""
    print(f"Parsing dependency list: {depedency_list}")
    
    pattern = r"^([\w.-]+)(@[^,\n]+|((?:[<>=!~]=?[\w.*-]+,?)+)?)$"
    module_version_dict = {}

    for dependency in depedency_list:
        if dependency.startswith("#"):
            continue
        match = re.match(pattern, dependency)
        if match:
            package = match.group(1)
            url_or_version = match.group(2) if match.group(2) else ""
            module_version_dict.update({package: url_or_version})
        else:
            module_version_dict.update({dependency: ""})
    
    print(f"Parsed module_version_dict: {module_version_dict}")
    return module_version_dict
