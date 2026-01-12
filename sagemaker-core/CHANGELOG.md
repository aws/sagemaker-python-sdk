# Changelog

## v2.3.1 (2025-01-12)

### Bug Fixes
* ProcessingJob fix - Remove tags in Processor while Job creation

## v2.3.0 (2025-12-19)

### Feature
* AWS_Batch: queueing of training jobs with ModelTrainer

## v2.2.0 (2025-12-18)

### Bug Fixes
* Add xgboost 3.0-5 to release
* Fix get_child_process_ids parsing issue
* Fix pip installation issues

## v2.1.1 (2025-12-10)

### Bug fixes

 * Fixed Unit and integ tests

## v2.0.1 (2025-11-21)

 * Fixed telemetry for PySDK V3 and SageMaker Core V2 usage

## v2.0 (2025-11-19)

 * Released SageMaker Core 2.0 as part of Python SDK 3.0 release

## v1.0.26 (2025-03-25)

 * Daily Sync with Botocore v1.37.19 on 2025/03/25 (#267)
 * UNIT TEST FIXES FROM BOTOCORE GENERATION  (#263)
 * Add CODEOWNERS file (#240)

## v1.0.25 (2025-02-27)

 * Daily Sync with Botocore v1.37.2 on 2025/02/27 (#251)

## v1.0.24 (2025-02-21)

 * Daily Sync with Botocore v1.36.25 on 2025/02/21 (#247)

## v1.0.23 (2025-02-20)

 * Daily Sync with Botocore v1.36.24 on 2025/02/20 (#246)

## v1.0.22 (2025-02-14)

 * Daily Sync with Botocore v1.36.20 on 2025/02/14 (#245)

## v1.0.21 (2025-02-05)

 * Daily Sync with Botocore v1.36.13 on 2025/02/05 (#239)

## v1.0.20 (2025-02-03)

 * Daily Sync with Botocore v1.36.11 on 2025/02/03 (#238)

## v1.0.19 (2025-01-20)

 * Daily Sync with Botocore v1.36.2 on 2025/01/20 (#236)

## v1.0.18 (2025-01-17)

 * Daily Sync with Botocore v1.36.1 on 2025/01/17 (#235)
 * Latest Botocore changes and unit test updates (#233)
 * fix: fetch version dynamically for useragent string and fix workflows (#230)
 * Use OIDC Role in workflows (#229)

## v1.0.17 (2024-12-04)

 * Daily Sync with Botocore v1.35.75 on 2024/12/04 (#227)
 * Add support for map-in-list, list-in-list structures (#224)

## v1.0.16 (2024-11-25)

 * Daily Sync with Botocore v1.35.68 on 2024/11/25 (#223)

## v1.0.15 (2024-11-19)

 * fix: update pydantic dep version (#222)

## v1.0.14 (2024-11-15)

 * Daily Sync with Botocore v1.35.62 on 2024/11/15 (#221)
 * Support BatchDeleteClusterNodes from sagemaker (#220)

## v1.0.13 (2024-11-01)

 * Daily Sync with Botocore v1.35.53 on 2024/11/01 (#219)

## v1.0.12 (2024-10-31)

 * Daily Sync with Botocore v1.35.52 on 2024/10/31 (#218)

## v1.0.11 (2024-10-30)

 * Daily Sync with Botocore v1.35.51 on 2024/10/30 (#217)

## v1.0.10 (2024-10-03)

 * Daily Sync with Botocore v1.35.32 on 2024/10/03 (#215)
 * fix: set rich panel to transient (#214)
 * Feature: Add wait with logs to subset of Job types (#201)

## v1.0.9 (2024-09-27)

 * Daily Sync with Botocore v1.35.28 on 2024/09/27 (#210)

## v1.0.8 (2024-09-25)

 * Daily Sync with Botocore v1.35.26 on 2024/09/25 (#209)
 * Support BatchGetMetrics from sagemaker-metrics (#207)

## v1.0.7 (2024-09-23)

 * Daily Sync with Botocore v1.35.24 on 2024/09/23 (#206)

## v1.0.6 (2024-09-20)

 * Daily Sync with Botocore v1.35.23 on 2024/09/20 (#205)
 * Update all PR Checks to have collab check (#202)

## v1.0.5 (2024-09-16)

 * Daily Sync with Botocore v1.35.19 on 2024/09/16 (#200)
 * Issue template (#196)
 * Collab check (#195)
 * Support APIs from sagemaker-featurestore-runtime and sagemaker-metrics (#181)

## v1.0.4 (2024-09-10)

 * Daily Sync with Botocore v1.35.15 on 2024/09/10 (#182)

## v1.0.3 (2024-09-06)

 * Daily Sync with Botocore v1.35.13 on 2024/09/06 (#180)
 * Add test to check API coverage (#165)
 * Update README.rst (#178)

## v1.0.2 (2024-09-04)

 * Daily Sync with Botocore v1.35.11 on 2024/09/04 (#179)
 * Add serialization for all methods (#177)
 * Add forbid extra for pydantic BaseModel (#173)
 * Add black check (#174)

## v1.0.1 (2024-08-30)

 * fix: SMD pydantic issue (#170)
 * feat: Add get started notebook (#160)
 * update notebooks (#168)
 * fix pyproject.toml (#167)

## v0.1.10 (2024-08-28)


## v0.1.9 (2024-08-28)

 * Update counting method of botocore api coverage (#159)
 * Example notebook for tracking local pytorch experiment (#158)
 * Add gen AI examples (#155)
 * Fix _serialize_args() for dict parameters (#157)

## v0.1.8 (2024-08-21)

 * Daily Sync with Botocore v1.35.2 on 2024/08/21 (#153)

## v0.1.7 (2024-08-13)

 * Daily Sync with Botocore v1.34.159 on 2024/08/13 (#150)
 * feat: add param validation with pydantic validate_call (#149)
 * Update create-release.yml
 * Support textual rich logging for wait methods (#146)
 * Refactor Package structure (#144)
 * Separate environment variable for Sagemaker Core (#147)
 * Add styling for textual rich logging (#145)
 * Replace all Sagemaker V2 Calls (#142)
 * Daily Sync with Botocore v1.34.153 on 2024/08/05 (#143)
 * Update auto-approve.yml
 * Use textual rich logging handler for all loggers (#138)
 * Update auto-approve.yml
 * Add user agent to Sagemaker Core (#140)
 * Switch to sagemaker-bot account (#137)
 * Metrics for boto API coverage (#136)
 * Fix volume_size_in_g_b attribute in example notebooks (#130)

## v0.1.6 (2024-07-25)

 * Add private preview feedback for denesting simplifications (#128)
 * Put Metrics only for Daily Sync API (#125)

## v0.1.5 (2024-07-22)

 * Daily Sync with Botocore v1.34.145 on 2024/07/22 (#127)

## v0.1.4 (2024-07-22)

 * Cleanup Resources created by Integration tests (#120)
 * Enable Botocore sync workflow (#92)

## v0.1.3 (2024-07-18)

 * Daily Sync with Botocore v1.34.143 on 2024/07/11 (#91)
 * Update license classifier (#119)
 * Metrics (#118)
 * Support wait_for_delete method (#114)

## v0.1.2 (2024-07-08)

 * Add additional methods to the unit test framework (#83)
 * Integration tests (#82)
 * Add exception and return type docstring for additional methods (#58)
 * Support SagemakerServicecatalogPortfolio resource (#49)
 * Support last few additional methods (#52)
 * Integration tests (#53)
 * Fix Intelligent defaults decorator conversion (#51)
 * Fix for Issues that came up in integration tests (#50)
 * Resource Unit test Framework and tests (#46)
 * Support resources by create method and move methods under EdgeDeploymentStage to EdgeDeploymentPlan (#48)
 * Support resources that have the List operation but do not have the Describe operation (#45)
 * Fix class method (#44)
 * Update docstring for additional methods (#43)
 * Add Python 3.11 and 3.12 to PR checks (#42)
 * change: update s3 bucket in notebooks and add cell to delete resources (#41)
 * Fix pascal_to_snake for consecutive capitalized characters (#38)
 * Intelligent Defaults with Snake cased arguments (#40)

## v0.1.1 (2024-06-14)

 * Rollback CHANGELOG.md
 * Rollback VERSION
 * prepare release v0.1.2
 * prepare release v0.1.1
 * Add resource class docstring (#37)
 * Create CHANGELOG.md (#39)

## v0.1.0 (2024-06-14)

 * Initial release of SageMaker Core
