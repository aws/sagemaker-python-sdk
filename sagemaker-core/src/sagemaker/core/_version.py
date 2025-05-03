import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the root directory of the project
root_dir = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

version_file_path = os.path.join(root_dir, "VERSION")

with open(version_file_path) as version_file:
    __version__ = version_file.read().strip()
