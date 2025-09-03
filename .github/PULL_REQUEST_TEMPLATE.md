*Issue #, if available:*

*Description of changes:*

*Testing done:*

## Merge Checklist

_Put an `x` in the boxes that apply. You can also fill these out after creating the PR. If you're unsure about any of them, don't hesitate to ask. We're here to help! This is simply a reminder of what we are going to look for before merging your pull request._

#### General

- [ ] I have read the [CONTRIBUTING](https://github.com/aws/sagemaker-python-sdk/blob/master/CONTRIBUTING.md) doc
- [ ] I certify that the changes I am introducing will be backward compatible, and I have discussed concerns about this, if any, with the Python SDK team
- [ ] I used the commit message format described in [CONTRIBUTING](https://github.com/aws/sagemaker-python-sdk/blob/master/CONTRIBUTING.md#committing-your-change)
- [ ] I have passed the region in to all S3 and STS clients that I've initialized as part of this change.
- [ ] I have updated any necessary documentation, including [READMEs](https://github.com/aws/sagemaker-python-sdk/blob/master/README.rst) and [API docs](https://github.com/aws/sagemaker-python-sdk/tree/master/doc) (if appropriate)

#### Tests

- [ ] I have added tests that prove my fix is effective or that my feature works (if appropriate)
- [ ] I have added unit and/or integration tests as appropriate to ensure backward compatibility of the changes
- [ ] I have checked that my tests are not configured for a specific region or account (if appropriate)
- [ ] I have used [`unique_name_from_base`](https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/utils.py#L77) to create resource names in integ tests (if appropriate)
- [ ] If adding any dependency in requirements.txt files, I have spell checked and ensured they exist in PyPi

By submitting this pull request, I confirm that my contribution is made under the terms of the Apache 2.0 license.
