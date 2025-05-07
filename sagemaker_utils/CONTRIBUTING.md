# Contributing Guidelines

Thank you for your interest in contributing to our project. Whether it's a bug report, new feature, correction, or additional
documentation, we greatly value feedback and contributions from our community.

Please read through this document before submitting any issues or pull requests to ensure we have all the necessary
information to effectively respond to your bug report or contribution.

This namespace - sagemaker utils mainly has all of the common utils for other sagemaker namespaces.  

## Setting up envirnonment . 

``pip intall .``
#### Running tests 

```pytest -vv sagemaker_utils/tests```

* Can run unit and integ tests separately like 
  * ```pytest -vv sagemaker_utils/tests/unit```
  * ```pytest -vv sagemaker_utils/tests/integ```

* Please make sure all the tests in unit and integ folders pass before posting the PR 
* Please make for any change that is made , approrpiate unit and integ tests are added .