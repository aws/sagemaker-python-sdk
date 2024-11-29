import logging
import subprocess
import sys
import re
from pathlib import Path
from typing import List
import argparse
import email.parser
import email.policy
import json
import inspect
import itertools
import subprocess
import tqdm
import ast

import cloudpickle
import boto3

pipcmd = [sys.executable, "-m", "pip", "--disable-pip-version-check"]


def get_all_files_for_installed_packages_pip(packages: List[str]):
    """Get all files for installed packages using pip."""
    print(f"Fetching files for installed packages: {packages}")
    proc = subprocess.Popen(pipcmd + ["show", "-f"] + packages, stdout=subprocess.PIPE)
    with proc.stdout:
        lines = []
        for line in iter(proc.stdout.readline, b""):
            if line == b"---\n":
                print(f"Package details: {lines}")
                yield lines
                lines = []
            else:
                lines.append(line)
    yield lines
    proc.wait(timeout=10)


def get_all_files_for_installed_packages(packages: List[str]):
    """Get all files for installed packages."""
    print(f"Processing installed packages: {packages}")
    ret = {}
    for rawmsg in get_all_files_for_installed_packages_pip(packages):
        parser = email.parser.BytesParser(policy=email.policy.default)
        msg = parser.parsebytes(b"".join(iter(rawmsg)))
        if not msg.get("Files"):
            continue
        ret[msg.get("Name")] = {
            Path(msg.get("Location")).joinpath(x) for x in msg.get("Files").split()
        }
        print(f"Package {msg.get('Name')} with files: {ret[msg.get('Name')]}")
    return ret


def batched(iterable, n):
    """Batch data into tuples of length n."""
    print(f"Batching data into groups of {n}")
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while True:
        batch = tuple(itertools.islice(it, n))
        if not batch:
            break
        print(f"Batch: {batch}")
        yield batch


def get_all_installed_packages():
    """Get all installed packages."""
    """from local env"""
    print("Fetching all installed packages...")
    proc = subprocess.run(pipcmd + ["list", "--format", "json"], stdout=subprocess.PIPE, check=True)
    all_packages = json.loads(proc.stdout)
    print(f"All installed packages: {all_packages}")
    return all_packages


def map_package_names_to_files(package_names: List[str]):
    """Map package names to their files."""
    print(f"Mapping package names to files for: {package_names}")
    m = {}
    batch_size = 20
    with tqdm.tqdm(total=len(package_names), desc="Scanning for dependencies", ncols=100) as pbar:
        for pkg_names in batched(package_names, batch_size):
            m.update(get_all_files_for_installed_packages(list(pkg_names)))
            pbar.update(batch_size)
            print(f"Processed batch: {pkg_names}")
    print(f"Package name to file map: {m}")
    return m


def get_currently_used_packages():
    """Get currently used packages."""
    print("Fetching currently used packages...")
    all_installed_packages = get_all_installed_packages()
    package_to_file_names = map_package_names_to_files([x["name"] for x in all_installed_packages])
    # print(f"package_to_file_names: {package_to_file_names}")

    currently_used_files = {
        Path(m.__file__)
        for m in sys.modules.values()
        if inspect.ismodule(m) and hasattr(m, "__file__") and m.__file__
    }

    print(f"Currently used files: {currently_used_files}")

    currently_used_packages = set()
    for file in currently_used_files:
        for package in package_to_file_names:
            if file in package_to_file_names[package]:
                print(f"file: {file}")
                print(f"package: {package}")
                currently_used_packages.add(package)
                
    # for module in sys.modules.values():
    #     if inspect.ismodule(module):
    #         for _, obj in inspect.getmembers(module):
    #             if inspect.ismethod(obj) or inspect.isfunction(obj):
    #                 source_code = inspect.getsource(obj)
    #                 import_nodes = [node for node in ast.walk(ast.parse(source_code)) if isinstance(node, ast.Import)]
    #                 for import_node in import_nodes:
    #                     for alias in import_node.names:
    #                         package_name = alias.name.split('.')[0]
    #                         if package_name in package_to_file_names:
    #                             currently_used_packages.add(package_name)

    print(f"Currently used packages: {currently_used_packages}")
    return currently_used_packages


def get_requirements_for_pkl_file(pkl_path: Path, dest: Path):
    """Get requirements for a pickled file."""
    print(f"Loading pickled file from {pkl_path}")
    with open(pkl_path, mode="rb") as file:
        cloudpickle.load(file)

    currently_used_packages = get_currently_used_packages()
    print(f"Currently used packages after loading pkl: {currently_used_packages}")

    with open(dest, mode="w+") as out:
        for x in get_all_installed_packages():
            name = x["name"]
            version = x["version"]
            # skip only for dev
            if name == "boto3":
                boto3_version = boto3.__version__
                out.write(f"boto3=={boto3_version}\n")
                print(f"Added boto3=={boto3_version} to requirements")
            elif name in currently_used_packages:
                out.write(f"{name}=={version}\n")
                print(f"Added {name}=={version} to requirements")


def get_all_requirements(dest: Path):
    """Get all installed requirements."""
    print(f"Getting all requirements and saving to {dest}")
    all_installed_packages = get_all_installed_packages()

    with open(dest, mode="w+") as out:
        for package_info in all_installed_packages:
            name = package_info.get("name")
            version = package_info.get("version")
            out.write(f"{name}=={version}\n")
            print(f"Added {name}=={version} to requirements2")


def parse_args():
    """Parse command-line arguments."""
    print("Parsing command-line arguments...")
    parser = argparse.ArgumentParser(
        prog="pkl_requirements", description="Generates a requirements.txt for a cloudpickle file"
    )
    parser.add_argument("--pkl_path", required=True, help="path of the pkl file")
    parser.add_argument("--dest", required=True, help="path of the destination requirements.txt")
    parser.add_argument(
        "--capture_all",
        action="store_true",
        help="capture all dependencies in current environment",
    )
    args = parser.parse_args()
    print(f"Arguments parsed: {args}")
    return (Path(args.pkl_path), Path(args.dest), args.capture_all)


def main():
    """Main function to execute the script."""
    print("Starting the main function...")
    pkl_path, dest, capture_all = parse_args()
    if capture_all:
        print(f"Capturing all requirements to {dest}")
        get_all_requirements(dest)
    else:
        print(f"Capturing requirements for pkl file {pkl_path} to {dest}")
        get_requirements_for_pkl_file(pkl_path, dest)


if __name__ == "__main__":
    main()

'''
capture_all is being set to False. Hence, we are getting reqs for pkl file. 
Then we get currently used pkgs. Get all installed pkgs. 
'''