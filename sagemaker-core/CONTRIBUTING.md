# Contribution guidelines for sagemaker-code-gen

## Setting up Enviornment using Pyenv
* Set up prerequisites following guide here -  https://github.com/pyenv/pyenv/wiki#suggested-build-environment

* Install Pyenv
```
curl https://pyenv.run | bash
```

* Add the following to  ~/.zshrc to load Pyenv automatically
```
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

* Install Python Version and setup virtual-env
```
pyenv install 3.10.14
pyenv virtualenv 3.10.14 py3.10.14
pyenv activate py3.10.14
```

* Install dependencies required for CodeGen and set PYTHONPATH
```
pip install ".[codegen]"
source .env
```

## Run CodeGen
* To generate all CodeGen code run the below
```
python src/sagemaker/core/tools/codegen.py
```

## Testing
* To check for regressions in existing flows, make sure to run: `pytest tst`. For new unit test coverage added make sure `pytest tst` validates them. 
```
pytest tst
```
* Use Pylint to detect errors and improve code quality. For code style errors use `black` to format the files.
```
black .
pylint **/*.py
```

## Building Distribution
* To build a distribution of SageMakerCore run below
```
pip install --upgrade build
python -m build
```