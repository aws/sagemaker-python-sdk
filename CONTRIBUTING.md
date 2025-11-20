# Contributing Guidelines

Thank you for your interest in contributing to our project. Whether it's a bug report, new feature, correction, or additional
documentation, we greatly value feedback and contributions from our community.

Please read through this document before submitting any issues or pull requests to ensure we have all the necessary
information to effectively respond to your bug report or contribution.


## Table of Contents

* [Report Bugs/Feature Requests](#report-bugsfeature-requests)
* [Contribute via Pull Requests (PRs)](#contribute-via-pull-requests-prs)
  * [Set up Your Development Environment *[Optional, but Recommended]*](#set-up-your-development-environment-optional-but-recommended)
  * [Pull Down the Code](#pull-down-the-code)
  * [Run the Unit Tests](#run-the-unit-tests)
  * [Run the Integration Tests](#run-the-integration-tests)
  * [Make and Test Your Change](#make-and-test-your-change)
  * [Lint Your Change](#lint-your-change)
  * [Commit Your Change](#commit-your-change)
  * [Send a Pull Request](#send-a-pull-request)
* [Documentation Guidelines](#documentation-guidelines)
  * [Overviews](#overviews)
  * [API References (docstrings)](#api-references-docstrings)
  * [Build and Test Documentation](#build-and-test-documentation)
* [Find Contributions to Work On](#find-contributions-to-work-on)
* [Code of Conduct](#code-of-conduct)
* [Security Issue Notifications](#security-issue-notifications)
* [Licensing](#licensing)

## Report Bugs/Feature Requests

We welcome you to use the GitHub issue tracker to report bugs or suggest features.

When filing an issue, please check [existing open](https://github.com/aws/sagemaker-python-sdk/issues) and [recently closed](https://github.com/aws/sagemaker-python-sdk/issues?utf8=%E2%9C%93&q=is%3Aissue%20is%3Aclosed%20) issues to make sure somebody else hasn't already
reported the issue. Please try to include as much information as you can. Details like these are incredibly useful:

* A reproducible test case or series of steps.
* The version of our code being used.
* Any modifications you've made relevant to the bug.
* A description of your environment or deployment.


## Contribute via Pull Requests (PRs)

Contributions via pull requests are much appreciated.

Before sending us a pull request, please ensure that:

* You are working against the latest source on the *master* branch.
* You check the existing open and recently merged pull requests to make sure someone else hasn't already addressed the problem.
* You open an issue to discuss any significant work - we would hate for your time to be wasted.


### Set up Your Development Environment *[Optional, but Recommended]*

1. Set up the Cloud9 environment:
   1. Instance type: You'll need at least 4 GB of RAM to avoid running into memory issues. We recommend at least a t3.medium to run the unit tests. A larger host will reduce the chance of encountering resource limits.
   1. Follow the instructions at [Creating a Cloud9 EC2 Environment](https://docs.aws.amazon.com/cloud9/latest/user-guide/create-environment.html#create-environment-main) to set up a Cloud9 EC2 environment.
1. Expand the storage of the EC2 instance from 10GB to 20GB:
   1. Because you'll need a minimum of 11GB of disk storage on the EC2 instance to run the repository's unit tests, you'll need to expand your EC2 volume size. We recommend at least 20GB. A larger volume will reduce the chance of encountering resource limits.
   1. Follow the instructions at [Modifying an EBS Volume Using Elastic Volumes (Console)](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/requesting-ebs-volume-modifications.html#modify-ebs-volume) to increase the EBS volume size associated with the newly created EC2 instance.
   1. Wait 5-10min for the new EBS volume increase to finalize.
   1. Allow EC2 to claim the additional space by stopping and then starting your EC2 host.
2. Set up a venv to manage dependencies:
   1. `python -m venv ~/.venv/myproject-env` to create the venv
   2. `source ~/.venv/myproject-env/bin/activate` to activate the venv
   3. `deactivate` to exit the venv


### Pull Down the Code

1. If you do not already have one, create a GitHub account by following the prompts at [Join Github](https://github.com/join).
1. Create a fork of this repository on GitHub. You should end up with a fork at `https://github.com/<username>/sagemaker-python-sdk`.
   1. Follow the instructions at [Fork a Repo](https://help.github.com/en/articles/fork-a-repo) to fork a GitHub repository.
1. Clone your fork of the repository: `git clone https://github.com/<username>/sagemaker-python-sdk` where `<username>` is your github username.


### Run the Unit Tests

1. Install tox using `pip install tox`
1. cd into the github project sagemaker-python-sdk folder: `cd sagemaker-python-sdk` or `cd /environment/sagemaker-python-sdk`
1. Install coverage using `pip install '.[test]'`
1. Run the following tox command and verify that all code checks and unit tests pass: `tox tests/unit`
1. You can also run a single test with the following command: `tox -e py310 -- -s -vv <path_to_file><file_name>::<test_function_name>`
1. You can run coverage via runcvoerage env : `tox -e runcoverage -- tests/unit` or `tox -e py310 -- tests/unit --cov=sagemaker --cov-append --cov-report xml`
  * Note that the coverage test will fail if you only run a single test, so make sure to surround the command with `export IGNORE_COVERAGE=-` and `unset IGNORE_COVERAGE`
  * Example: `export IGNORE_COVERAGE=- ; tox -e py310 -- -s -vv tests/unit/test_estimator.py::test_sagemaker_model_s3_uri_invalid ; unset IGNORE_COVERAGE`


### Run the Integration Tests

Our CI system runs integration tests (the ones in the `tests/integ` directory), in parallel, for every Pull Request.
You should only worry about manually running any new integration tests that you write, or integration tests that test an area of code that you've modified.

1. Follow the instructions at [Set Up the AWS Command Line Interface (AWS CLI)](https://docs.aws.amazon.com/polly/latest/dg/setup-aws-cli.html).
1. To run a test, specify the test file and method you want to run per the following command: `tox -e py310 -- -s -vv <path_to_file><file_name>::<test_function_name>`
   * Note that the coverage test will fail if you only run a single test, so make sure to surround the command with `export IGNORE_COVERAGE=-` and `unset IGNORE_COVERAGE`
   * Example: `export IGNORE_COVERAGE=- ; tox -e py310 -- -s -vv tests/integ/test_tf_script_mode.py::test_mnist ; unset IGNORE_COVERAGE`

If you are writing or modifying a test that creates a SageMaker job (training, tuner, or transform) or endpoint, it's important to assign a concurrency-friendly `job_name` (or `endpoint_name`), or your tests may fail randomly due to name collisions. We have a helper method `sagemaker.utils.unique_name_from_base(base, max_length)` that makes test-friendly names. You can find examples of how to use it [here](https://github.com/aws/sagemaker-python-sdk/blob/3816a5658d3737c9767e01bc8d37fc3ed5551593/tests/integ/test_tfs.py#L37) and
[here](https://github.com/aws/sagemaker-python-sdk/blob/3816a5658d3737c9767e01bc8d37fc3ed5551593/tests/integ/test_tuner.py#L616), or by searching for "unique\_name\_from\_base" in our test code.


### Make and Test Your Change

1. Create a new git branch:
     ```shell
     git checkout -b my-fix-branch master
     ```
1. Make your changes, **including unit tests** and, if appropriate, integration tests.
   1. Include unit tests when you contribute new features or make bug fixes, as they help to:
      1. Prove that your code works correctly.
      1. Guard against future breaking changes to lower the maintenance cost.
   1. Please focus on the specific change you are contributing. If you also reformat all the code, it will be hard for us to focus on your change.
1. Run all the unit tests as per [Run the Unit Tests](#run-the-unit-tests), and verify that all checks and tests pass.
   1. Note that this also runs tools that may be necessary for the automated build to pass (ex: code reformatting by 'black').
1. If your changes include documentation changes, please see the [Documentation Guidelines](#documentation-guidelines).
1. If you include integration tests, do not mark them as canaries if they will not run in all regions.

### Lint Your Change

Before submitting, ensure your code meets our quality and style guidelines. Run:
```shell
tox -e flake8,pylint,docstyle,black-check,twine --parallel all
```
Address any errors or warnings before opening a pull request.

### Commit Your Change

We use commit messages to update the project version number and generate changelog entries, so it's important for them to follow the right format. Valid commit messages include a prefix, separated from the rest of the message by a colon and a space. Here are a few examples:

```
feature: support VPC config for hyperparameter tuning
fix: fix flake8 errors
documentation: add MXNet documentation
```

Valid prefixes are listed in the table below.

| Prefix          | Use for...                                                                                     |
|----------------:|:-----------------------------------------------------------------------------------------------|
| `breaking`      | Incompatible API changes.                                                                      |
| `deprecation`   | Deprecating an existing API or feature, or removing something that was previously deprecated.  |
| `feature`       | Adding a new feature.                                                                          |
| `fix`           | Bug fixes.                                                                                     |
| `change`        | Any other code change.                                                                         |
| `documentation` | Documentation changes.                                                                         |

Some of the prefixes allow abbreviation ; `break`, `feat`, `depr`, and `doc` are all valid. If you omit a prefix, the commit will be treated as a `change`.

For the rest of the message, use imperative style and keep things concise but informative. See [How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/) for guidance.


### Send a Pull Request

GitHub provides additional document on [Creating a Pull Request](https://help.github.com/articles/creating-a-pull-request/).

Please remember to:
* Use commit messages (and PR titles) that follow the guidelines under [Commit Your Change](#commit-your-change).
* Send us a pull request, answering any default questions in the pull request interface.
* Pay attention to any automated CI failures reported in the pull request, and stay involved in the conversation.


## Documentation Guidelines

We use reStructuredText (RST) for most of our documentation. For a quick primer on the syntax,
see [the Sphinx documentation](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html).

In this repository, we have two main categories of documentation: overviews and API references.
"How to" tutorials are housed in the [Amazon SageMaker Examples repository](https://github.com/awslabs/amazon-sagemaker-examples).
Overviews and API references are discussed in more detail below.

Here are some general guidelines to follow when writing either kind of documentation:
* Use present tense.
  * 👍 "The estimator fits a model."
  * 👎 "The estimator will fit a model."
* When referring to an AWS product, use its full name in the first invocation.
  (This applies only to prose; use what makes sense when it comes to writing code, etc.)
  * 👍 "Amazon S3"
  * 👎 "s3"
* Provide links to other ReadTheDocs pages, AWS documentation, etc. when helpful.
  Try to not duplicate documentation when you can reference it instead.
  * Use meaningful text in a link.
    * 👍 You can learn more about [hyperparameter tuning with SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-how-it-works.html) in the SageMaker docs.
    * 👎 Read more about it [here](#).


### Overviews

This section refers to documentation that discusses a specific topic or feature to
help the reader deepen their understanding, and may include short snippets of how to do specific tasks.
Examples include "[Amazon SageMaker Debugger](https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_debugger.html)"
and "[Use MXNet with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/using_mxnet.html)."

The goal of these documents is to explain basic usage.
This includes the general purpose of the topic or feature,
and common ways to use the SageMaker Python SDK in that context.

This type of documentation should not be a step-by-step tutorial.
That is better suited for the [example notebooks](https://github.com/awslabs/amazon-sagemaker-examples).
Instead, keep the content focused on the unique aspects of the feature.
For example, if one is writing specifically about deploying models,
there is no need to also include instructions on how to train a model first.
In this case, consider linking to existing documentation about training models and any other prerequisites.

Lastly, in addition to the general guidelines listed above:
* Use the imperative mood for headings.
  * 👍 "Prepare a Training Script"
  * 👎 "Preparing a Training Script"
* Don’t refer to features as "new" - they might be at the time of writing, but they won’t always be!

### API References (docstrings)

The API references are generated from docstrings.
A docstring is the comment in the source code that describes a module, class, function, or variable.

```python
def foo():
    """This comment is a docstring for the function foo."""
```

We use [Google-style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
There should be a docstring for every public module, class, and function.
For functions, make sure your docstring covers all of the arguments, exceptions, and any other relevant information.
When possible, link to classes and functions, e.g. use ":class:~\`sagemaker.session.Session\`" over just "Session."

If a parameter of a function has a default value, please note what the default is.
If that default value is `None`, it can also be helpful to explain what happens when the parameter is `None`.
If `**kwargs` is part of the function signature, link to the parent class(es) or method(s) so that the reader knows where to find the available parameters.

For an example file with docstrings, see [the `processing` module](https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/processing.py).

To have a class's docstrings included in the API reference, it needs to be included in one of the files in the `doc/` folder.
For example, see the [Processing API reference](https://github.com/aws/sagemaker-python-sdk/blob/master/doc/processing.rst).


### Build and Test Documentation

To build the Sphinx docs, run the following command in the `doc/` directory:

```shell
# Initial setup, only required for the first run
pip install -r requirements.txt
pip install -e ../
```

```shell
make html
```

You can then find the generated HTML files in `doc/_build/html/`.

To check both the README and API documentation for build errors, you can run the following:

```shell
tox -e twine,sphinx
```


## Find Contributions to Work On

Looking at the existing issues is a great way to find something to contribute on. As our projects, by default, use the default GitHub issue labels ((enhancement/bug/duplicate/help wanted/invalid/question/wontfix), looking at any ['help wanted'](https://github.com/aws/sagemaker-python-sdk/labels/help%20wanted) issues is a great place to start.


## Code of Conduct

This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact
opensource-codeofconduct@amazon.com with any additional questions or comments.


## Security Issue Notifications

If you discover a potential security issue in this project we ask that you notify AWS/Amazon Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public github issue.


## Licensing

See the [LICENSE](https://github.com/aws/sagemaker-python-sdk/blob/master/LICENSE) file for our project's licensing. We will ask you to confirm the licensing of your contribution.

We may ask you to sign a [Contributor License Agreement (CLA)](http://en.wikipedia.org/wiki/Contributor_License_Agreement) for larger changes.
