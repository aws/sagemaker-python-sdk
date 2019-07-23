# Contributing Guidelines

Thank you for your interest in contributing to our project. Whether it's a bug report, new feature, correction, or additional
documentation, we greatly value feedback and contributions from our community.

Please read through this document before submitting any issues or pull requests to ensure we have all the necessary
information to effectively respond to your bug report or contribution.


## Reporting Bugs/Feature Requests

We welcome you to use the GitHub issue tracker to report bugs or suggest features.

When filing an issue, please check [existing open](https://github.com/aws/sagemaker-python-sdk/issues), or [recently closed](https://github.com/aws/sagemaker-python-sdk/issues?utf8=%E2%9C%93&q=is%3Aissue%20is%3Aclosed%20), issues to make sure somebody else hasn't already
reported the issue. Please try to include as much information as you can. Details like these are incredibly useful:

* A reproducible test case or series of steps
* The version of our code being used
* Any modifications you've made relevant to the bug
* A description of your environment or deployment

## Setting up your development environment [optional, but recommended]

* Set up the Cloud9 environment:
  * Instance type: You'll need at least 4 GB of RAM to avoid running into memory issues. We recommend at least a t3.medium to run the unit tests. Larger hosts will reduce the chance of encountering resource limits.
  * Follow the instructions at [Creating a Cloud9 EC2 Environment](https://docs.aws.amazon.com/cloud9/latest/user-guide/create-environment.html#create-environment-main) to set up a Cloud9 EC2 environment
* Expand the storage of the EC2 instance from 10GB to 20GB
  * Because you'll need a minimum of 11GB of disk storage on the EC2 instance to run the package's unit tests, you'll need to expand your EC2 volume size. We recommend at least 20GB. A larger volume will reduce the chance of encountering resource limits. 
  * Follow the instructions at [Modifying an EBS Volume Using Elastic Volumes (Console)](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/requesting-ebs-volume-modifications.html#modify-ebs-volume) to increase the EBS volume size associated with the newly created EC2 instance.
  * Wait 5-10min for the new EBS volume increase to take effect.
  * Allow EC2 to claim the additional space by stopping and then starting your EC2 host.
* Create a fork of this package on GitHub. You should end up with a fork at `https://github.com/<username>/sagemaker-python-sdk`
  * Follow the instructions at [Fork a repo](https://help.github.com/en/articles/fork-a-repo) to fork a GitHub repository.
* In the Cloud9 UI, pull down this package by clicking on "Clone from Github" or running the following command in the Cloud9 terminal: `git clone https://github.com/<username>/sagemaker-python-sdk` where <username> is your github username.
* Install tox using `pip install tox`
* Install coverage using `pip install .[test]`
* cd into the sagemaker-python-sdk package: `cd sagemaker-python-sdk` or `cd /environment/sagemaker-python-sdk`
* Run the following tox command and verify that all unit tests pass: `tox tests/unit`

## Contributing via Pull Requests
Contributions via pull requests are much appreciated. 

You can use the following commands to setup your developing and testing environment after you fork sagemaker-python-sdk repository:

```bash
git clone git@github.com:<your-github-username>/sagemaker-python-sdk.git
cd sagemaker-python-sdk
pip install -U .
pip install -U .[test]
```

Before sending us a pull request, please ensure that:

1. You are working against the latest source on the *master* branch.
2. You check existing open, and recently merged, pull requests to make sure someone else hasn't addressed the problem already.
3. You open an issue to discuss any significant work - we would hate for your time to be wasted.

To send us a pull request, please:

1. Fork the repository.
2. Modify the source; please focus on the specific change you are contributing. If you also reformat all the code, it will be hard for us to focus on your change.
3. Include unit tests when you contribute new features or make bug fixes, as they help to a) prove that your code works correctly, and b) guard against future breaking changes to lower the maintenance cost.
4. Ensure local tests pass.
5. Use commit messages (and PR titles) that follow [these guidelines](#commit-message-guidelines).
6. Send us a pull request, answering any default questions in the pull request interface.
7. Pay attention to any automated CI failures reported in the pull request, and stay involved in the conversation.

GitHub provides additional document on [forking a repository](https://help.github.com/articles/fork-a-repo/) and
[creating a pull request](https://help.github.com/articles/creating-a-pull-request/).

### Commit message guidelines

We use commit messages to update the project version number and generate changelog entries, so it's important for them to follow the right format. Valid commit messages include a prefix, separated from the rest of the message by a colon and a space. Here are a few examples:

```
feature: support VPC config for hyperparameter tuning
fix: fix flake8 errors
documentation: add MXNet documentation
```

Valid prefixes are listed in the table below.

| Prefix          | Use for...                                                                                     |
|-----------------|------------------------------------------------------------------------------------------------|
| `breaking`      | Incompatible API changes.                                                                      |
| `deprecation`   | Deprecating an existing API or feature, or removing something that was previously deprecated.  |
| `feature`       | Adding a new feature.                                                                          |
| `fix`           | Bug fixes.                                                                                     |
| `change`        | Any other code change.                                                                         |
| `documentation` | Documentation changes.                                                                         |

Some of the prefixes allow abbreviation -- `break`, `feat`, `depr`, and `doc` are all valid. If you omit a prefix, the commit will be treated as a `change`.

For the rest of the message, use imperative style and keep things concise but informative. See [How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/) for guidance.

### Integration tests

Our CI system runs integration tests (the ones in the `tests/integ` directory) in parallel. If you are writing or modifying a test that creates a SageMaker job (training, tuner, or transform) or an endpoint, it's important to assign a concurrency-friendly `job_name` (or `endpoint_name`), or your tests may fail randomly due to name collisions. We have a helper method `sagemaker.utils.unique_name_from_base(base, max_length)` that makes test-friendly names. You can find examples of how to use it [here](https://github.com/aws/sagemaker-python-sdk/blob/3816a5658d3737c9767e01bc8d37fc3ed5551593/tests/integ/test_tfs.py#L37) and 
[here](https://github.com/aws/sagemaker-python-sdk/blob/3816a5658d3737c9767e01bc8d37fc3ed5551593/tests/integ/test_tuner.py#L616), or by searching for "unique\_name\_from\_base" in our test code.

## Finding contributions to work on
Looking at the existing issues is a great way to find something to contribute on. As our projects, by default, use the default GitHub issue labels ((enhancement/bug/duplicate/help wanted/invalid/question/wontfix), looking at any ['help wanted'](https://github.com/aws/sagemaker-python-sdk/labels/help%20wanted) issues is a great place to start.


## Code of Conduct
This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact
opensource-codeofconduct@amazon.com with any additional questions or comments.


## Security issue notifications
If you discover a potential security issue in this project we ask that you notify AWS/Amazon Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public github issue.


## Licensing

See the [LICENSE](https://github.com/aws/sagemaker-python-sdk/blob/master/LICENSE) file for our project's licensing. We will ask you to confirm the licensing of your contribution.

We may ask you to sign a [Contributor License Agreement (CLA)](http://en.wikipedia.org/wiki/Contributor_License_Agreement) for larger changes.
