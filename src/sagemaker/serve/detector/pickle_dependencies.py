"""Load a pickled object to detect the dependencies it requires"""

from __future__ import absolute_import
from pathlib import Path
from typing import List
import argparse
import email.parser
import email.policy
import json
import inspect
import itertools
import subprocess
import sys
import tqdm

# non native imports. Ideally add as little as possible here
# because it will add to requirements.txt
import cloudpickle
import boto3

pipcmd = [sys.executable, "-m", "pip", "--disable-pip-version-check"]


def get_all_files_for_installed_packages_pip(packages: List[str]):
    """Placeholder docstring"""
    proc = subprocess.Popen(pipcmd + ["show", "-f"] + packages, stdout=subprocess.PIPE)
    with proc.stdout:
        lines = []
        for line in iter(proc.stdout.readline, b""):
            if line == b"---\n":
                yield lines
                lines = []
            else:
                lines.append(line)
    yield lines
    proc.wait(timeout=10)  # wait for the subprocess to exit


def get_all_files_for_installed_packages(packages: List[str]):
    """Placeholder docstring"""
    ret = {}
    for rawmsg in get_all_files_for_installed_packages_pip(packages):
        parser = email.parser.BytesParser(policy=email.policy.default)
        msg = parser.parsebytes(b"".join(iter(rawmsg)))
        if not msg.get("Files"):
            continue
        ret[msg.get("Name")] = {
            Path(msg.get("Location")).joinpath(x) for x in msg.get("Files").split()
        }

    return ret


def batched(iterable, n):
    """Batch data into tuples of length n. The last batch may be shorter."""
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while True:
        batch = tuple(itertools.islice(it, n))
        if not batch:
            break
        yield batch


def get_all_installed_packages():
    """Placeholder docstring"""
    proc = subprocess.run(pipcmd + ["list", "--format", "json"], stdout=subprocess.PIPE, check=True)
    return json.loads(proc.stdout)


def map_package_names_to_files(package_names: List[str]):
    """Placeholder docstring"""
    m = {}
    batch_size = 20
    with tqdm.tqdm(total=len(package_names), desc="Scanning for dependencies", ncols=100) as pbar:
        for pkg_names in batched(package_names, batch_size):
            m.update(get_all_files_for_installed_packages(list(pkg_names)))
            pbar.update(batch_size)
    return m


def get_currently_used_packages():
    """Placeholder docstring"""
    all_installed_packages = get_all_installed_packages()
    package_to_file_names = map_package_names_to_files([x["name"] for x in all_installed_packages])

    currently_used_files = {
        Path(m.__file__)
        for m in sys.modules.values()
        if inspect.ismodule(m) and hasattr(m, "__file__") and m.__file__
    }

    currently_used_packages = set()
    for file in currently_used_files:
        for package in package_to_file_names:
            if file in package_to_file_names[package]:
                currently_used_packages.add(package)
    return currently_used_packages


def get_requirements_for_pkl_file(pkl_path: Path, dest: Path):
    """Placeholder docstring"""
    with open(pkl_path, mode="rb") as file:
        cloudpickle.load(file)

    currently_used_packages = get_currently_used_packages()

    with open(dest, mode="w+") as out:
        for x in get_all_installed_packages():
            name = x["name"]
            version = x["version"]
            # skip only for dev
            if name == "boto3":
                boto3_version = boto3.__version__
                out.write(f"boto3=={boto3_version}\n")
            elif name in currently_used_packages:
                out.write(f"{name}=={version}\n")


def get_all_requirements(dest: Path):
    """Placeholder docstring"""
    all_installed_packages = get_all_installed_packages()

    with open(dest, mode="w+") as out:
        for package_info in all_installed_packages:
            name = package_info.get("name")
            version = package_info.get("version")

            out.write(f"{name}=={version}\n")


def parse_args():
    """Placeholder docstring"""
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
    return (Path(args.pkl_path), Path(args.dest), args.capture_all)


def main():
    """Placeholder docstring"""
    pkl_path, dest, capture_all = parse_args()
    if capture_all:
        get_all_requirements(dest)
    else:
        get_requirements_for_pkl_file(pkl_path, dest)


if __name__ == "__main__":
    main()
