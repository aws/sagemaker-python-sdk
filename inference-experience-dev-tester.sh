trap "exit" INT

eval "$(conda shell.zsh hook)"

conda activate $1

echo Installing tox if necessary
pip install --upgrade tox

echo Running unit testing in Python3.8 Conda Environement

if "$4";
then
    echo Clearing .tox cache for Python3.8
    rm -r .tox/py38
fi

tox -e py38 -- tests/unit/sagemaker/serve/.

if "$3";
then
    echo Running Python3.8 Integration Tests
    tox -e py38 -- tests/integ/sagemaker/serve/.
fi

conda deactivate

conda activate $2

echo Installing tox if necessary
pip install --upgrade tox

echo Running unit testing in Python3.10 Conda Environment

if "$4";
then
    echo Clearing .tox cache for Python3.10
    rm -r .tox/py10
fi

tox -e py310 -- tests/unit/sagemaker/serve/. 

if "$3";
then
    echo Running Python3.10 Integration Tests
    tox -e py310 -- tests/integ/sagemaker/serve/.
fi

conda deactivate

echo Coverage report after testing:

coverage report -i --fail-under=75 --include "*/serve/*" --omit '*in_process*,*interceptors*,*__init__*,*build_model*,*function_pointers*'
