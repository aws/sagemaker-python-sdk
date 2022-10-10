import os

from setuptools import find_packages, setup

# Declare your non-python data files:
# Files underneath configuration/ will be copied into the build preserving the
# subdirectory structure if they exist.
data_files = []
for root, dirs, files in os.walk("configuration"):
    data_files.append(
        (os.path.relpath(root, "configuration"), [os.path.join(root, f) for f in files])
    )

setup(
    name="SageMakerTitanWorkflowsPythonLambda",
    version="1.0",
    # declare your packages
    packages=find_packages(where="src", exclude=("test",)),
    package_dir={"": "src"},
    # include data files
    data_files=data_files,
    # set up the shebang
    options={
        # make sure the right shebang is set for the scripts - use the
        # environment default Python
        "build_scripts": {
            "executable": '/apollo/sbin/envroot "$ENVROOT/bin/python"',
        },
    },
    # Use the pytest brazilpython runner. Provided by BrazilPython-Pytest.
    test_command="brazilpython_pytest",
    # Use custom sphinx command which adds an index.html that's compatible with
    # code.amazon.com links.
    # doc_command='amazon_doc_utils_build_sphinx',
    check_format=True,  # Enable build-time format checking
    test_mypy=False,  # Enable type checking
    test_flake8=False,  # Enable linting at build time
)
