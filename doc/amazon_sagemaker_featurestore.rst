##############################
Amazon SageMaker Feature Store
##############################

.. rubric:: **Create Feature Groups**
   :name: bCe9CAXalwH

This guide will show you how to create and use
`Amazon SageMaker Feature Store <https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store-getting-started.html>`__.
The example code in this guide covers using the SageMaker Python SDK. The
underlying APIs are available for developers using other languages.

.. rubric:: Features
   :name: bCe9CAtWHPP

Prior to using a feature store you will typically load your dataset, run
transformations, and set up your features for ingestion. This step has a
lot of variation and is highly dependent on your data. The example code
in the following code blocks will often make reference to an example
notebook, \ `Fraud Detection with Amazon SageMaker Feature Store
<https://sagemaker-examples.readthedocs.io/en/latest/sagemaker-featurestore/sagemaker_featurestore_fraud_detection_python_sdk.html>`__.
It is recommended that you run this notebook
in SageMaker Studio and use the code from there, as the code in this
guide is conceptual and not fully functional if copied.

.. rubric:: Feature store data types and schema
   :name: bCe9CAr4kIT

Feature Store supported types are ``String``, ``Fractional``, and
``Integral``. The default type is set to ``String``. This means that, if
a column in your dataset is not a ``float`` or ``long`` type, it will
default to ``String`` in your feature store.


You may use a schema to describe your data’s columns and data types. You
pass this schema into FeatureDefinitions, a required parameter for a
FeatureGroup. However, for Python developers, the SageMaker Python SDK
has automatic data type detection when you use the
``load_feature_definitions`` function.

.. rubric:: Feature store setup
   :name: bCe9CAgy6IH

To start using Feature Store, first create a SageMaker session, boto3
session, and a Feature Store session. Also, setup the bucket you will
use for your features; this is your Offline Store. The following will
use the SageMaker default bucket and add a custom prefix to it.

.. note::

   The role that you use requires these managed
   policies:\ ``AmazonSageMakerFullAccess``\ and\ ``AmazonSageMakerFeatureStoreAccess``\ .


.. code:: python

   import boto3
   import sagemaker
   from sagemaker.session import Session

   boto_session = boto3.Session(region_name=region)
   role = sagemaker.get_execution_role()
   sagemaker_session = sagemaker.Session()
   region = sagemaker_session.boto_region_name
   default_bucket = sagemaker_session.default_bucket()
   prefix = 'sagemaker-featurestore'
   offline_feature_store_bucket = 's3://*{}*/*{}*'.format(default_bucket, prefix)

   sagemaker_client = boto_session.client(service_name='sagemaker', region_name=region)
   featurestore_runtime = boto_session.client(service_name='sagemaker-featurestore-runtime', region_name=region)

   feature_store_session = Session(
       boto_session=boto_session,
       sagemaker_client=sagemaker_client,
       sagemaker_featurestore_runtime_client=featurestore_runtime
   )

.. rubric:: Load datasets and partition data into feature groups
   :name: bCe9CA31y9f

You will load your data into data frames for each of your features. You
will use these data frames after you setup the feature group. In the
fraud detection example, you can see these steps in the following code.

.. code:: python

   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   import io

   fraud_detection_bucket_name = 'sagemaker-featurestore-fraud-detection'
   identity_file_key = 'sampled_identity.csv'
   transaction_file_key = 'sampled_transactions.csv'

   identity_data_object = s3_client.get_object(Bucket=fraud_detection_bucket_name, Key=identity_file_key)
   transaction_data_object = s3_client.get_object(Bucket=fraud_detection_bucket_name, Key=transaction_file_key)

   identity_data = pd.read_csv(io.BytesIO(identity_data_object['Body'].read()))
   transaction_data = pd.read_csv(io.BytesIO(transaction_data_object['Body'].read()))

   identity_data = identity_data.round(5)
   transaction_data = transaction_data.round(5)

   identity_data = identity_data.fillna(0)
   transaction_data = transaction_data.fillna(0)

   # Feature transformations for this dataset are applied before ingestion into FeatureStore.
   # One hot encode card4, card6
   encoded_card_bank = pd.get_dummies(transaction_data['card4'], prefix = 'card_bank')
   encoded_card_type = pd.get_dummies(transaction_data['card6'], prefix = 'card_type')

   transformed_transaction_data = pd.concat([transaction_data, encoded_card_type, encoded_card_bank], axis=1)
   transformed_transaction_data = transformed_transaction_data.rename(columns={"card_bank_american express": "card_bank_american_express"})

.. rubric:: Feature group setup
   :name: bCe9CARx8h9

Name your feature groups and customize the feature names with a unique
name, and setup each feature group with the ``FeatureGroup`` class.

.. code:: python

   from sagemaker.feature_store.feature_group import FeatureGroup
   feature_group_name = "some string for a name"
   feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=feature_store_session)

For example, in the fraud detection example, the two feature groups are
“identity” and “transaction”. In the following code you can see how the
names are customized with a timestamp, then each group is setup by
passing in the name and the session.

.. code:: python

   import time
   from time import gmtime, strftime, sleep
   from sagemaker.feature_store.feature_group import FeatureGroup

   identity_feature_group_name = 'identity-feature-group-' + strftime('%d-%H-%M-%S', gmtime())
   transaction_feature_group_name = 'transaction-feature-group-' + strftime('%d-%H-%M-%S', gmtime())

   identity_feature_group = FeatureGroup(name=identity_feature_group_name, sagemaker_session=feature_store_session)
   transaction_feature_group = FeatureGroup(name=transaction_feature_group_name, sagemaker_session=feature_store_session)

.. rubric:: Record identifier and event time feature
   :name: bCe9CA17VV7

Next, you will need a record identifier name and an event time feature
name. This will match the column of the corresponding features in your
data. For example, in the fraud detection example, the column of
interest is “TransactionID”. “EventTime” can be appended to your data
when no timestamp is available. In the following code, you can see how
these variables are set, and then ``EventTime`` is appended to both
feature’s data.

.. code:: python

   record_identifier_name = "TransactionID"
   event_time_feature_name = "EventTime"
   current_time_sec = int(round(time.time()))
   identity_data[event_time_feature_name] = pd.Series([current_time_sec]*len(identity_data), dtype="float64")
   transformed_transaction_data[event_time_feature_name] = pd.Series([current_time_sec]*len(transaction_data), dtype="float64")

.. rubric:: Feature definitions
   :name: bCe9CA4yUcO

You can now load the feature definitions by passing a data frame
containing the feature data. In the following code for the fraud
detection example, the identity feature and transaction feature are each
loaded by using ``load_feature_definitions``, and this function
automatically detects the data type of each column of data. For
developers using a schema rather than automatic detection, refer to the
`Creating Feature Groups with Data Wrangler example <https://docs.aws.amazon.com/sagemaker/latest/dg/data-wrangler-data-export.html#data-wrangler-data-export-feature-store>`__ for
code that shows loading the schema, mapping it and adding as a
``FeatureDefinition`` that is used when you create the ``FeatureGroup``.
This example also covers a boto3 implementation, instead of using the
SageMaker Python SDK.

.. code:: python

   identity_feature_group.load_feature_definitions(data_frame=identity_data); # output is suppressed
   transaction_feature_group.load_feature_definitions(data_frame=transformed_transaction_data); # output is suppressed

.. rubric:: Create a feature group
   :name: bCe9CAwMEgY

The last step for creating the feature group is to use the
``create`` function. The following code shows all of the available
parameters. The online store is not created by default, so you must set
this as \ ``True`` if you want to enable it. The ``s3_uri`` is the
location of your offline store.

.. code:: python

   # create a FeatureGroup
   feature_group.create(
       description = "Some info about the feature group",
       feature_group_name = feature_group_name,
       record_identifier_name = record_identifier_name,
       event_time_feature_name = event_time_feature_name,
       feature_definitions = feature_definitions,
       role_arn = role,
       s3_uri = offline_feature_store_bucket,
       enable_online_store = True,
       ttl_duration = None,
       online_store_kms_key_id = None,
       offline_store_kms_key_id = None,
       disable_glue_table_creation = False,
       data_catalog_config = None,
       tags = ["tag1","tag2"])

The following code from the fraud detection example shows a minimal
``create`` call for each of the two features groups being created.

.. code:: python

   identity_feature_group.create(
       s3_uri=offline_feature_store_bucket,
       record_identifier_name=record_identifier_name,
       event_time_feature_name=event_time_feature_name,
       role_arn=role,
       enable_online_store=True
   )

   transaction_feature_group.create(
       s3_uri=offline_feature_store_bucket,
       record_identifier_name=record_identifier_name,
       event_time_feature_name=event_time_feature_name,
       role_arn=role,
       enable_online_store=True
   )

Creating a feature group takes time as the data is loaded. You will
need to wait until it is created before you can use it. You can
check status using the following method. Note that it can take
approximately 10-15 minutes to provision an online ``FeatureGroup``
with the ``InMemory`` ``StorageType``.

.. code:: python

    status = feature_group.describe().get("FeatureGroupStatus")

While it is creating you will get a ``Creating`` as a response. When
this has finished successfully the response will be ``Created``. The
other possible statuses are ``CreateFailed``, ``Deleting``, or
``DeleteFailed``.

.. rubric:: Describe a feature group
   :name: bCe9CA2TNON

You can retrieve information about your feature group with the
``describe`` function.

.. code:: python

   feature_group.describe()

.. rubric:: List feature groups
   :name: bCe9CA2wPF2

You can list all of your feature groups with the
``list_feature_groups`` function.

.. code:: python

   sagemaker_client.list_feature_groups()

.. rubric:: Put records in a feature group
   :name: bCe9CAymRdA

You can use the ``ingest`` function to load your feature data. You pass
in a data frame of feature data, set the number of workers, and choose
to wait for it to return or not. The following example demonstrates
using the ``ingest`` function.

.. code:: python

   feature_group.ingest(
       data_frame=feature_data, max_workers=3, wait=True
   )

For each feature group you have, run the ``ingest`` function on the
feature data you want to load.

.. rubric:: Get records from a feature group
   :name: bCe9CA25xj5

You can use the ``get_record`` function to retrieve the data for a
specific feature by its record identifier. The following example uses an
example identifier to retrieve the record.

.. code:: python

   record_identifier_value = str(2990130)
   featurestore_runtime.get_record(FeatureGroupName=transaction_feature_group_name, RecordIdentifierValueAsString=record_identifier_value)

You can use the ``batch_get_record`` function to retrieve multiple records simultaneously from your feature store. The following example uses this API to retrieve a batch of records.

.. code:: python

   record_identifier_values = ["573291", "109382", "828400", "124013"]
   featurestore_runtime.batch_get_record(Identifiers=[{"FeatureGroupName": transaction_feature_group_name, "RecordIdentifiersValueAsString": record_identifier_values}])

An example response from the fraud detection example:

.. code:: python

   ...
   'Record': [{'FeatureName': 'TransactionID', 'ValueAsString': '2990130'},
     {'FeatureName': 'isFraud', 'ValueAsString': '0'},
     {'FeatureName': 'TransactionDT', 'ValueAsString': '152647'},
     {'FeatureName': 'TransactionAmt', 'ValueAsString': '75.0'},
     {'FeatureName': 'ProductCD', 'ValueAsString': 'H'},
     {'FeatureName': 'card1', 'ValueAsString': '4577'},
   ...

.. rubric:: Hive DDL commands
   :name: bCe9CA30nHn

The SageMaker Python SDK’s FeatureStore class also provides the
functionality to generate Hive DDL commands. The schema of the table is
generated based on the feature definitions. Columns are named after
feature name and data-type are inferred based on feature type.

.. code:: python

   print(feature_group.as_hive_ddl())

An example output:

.. code:: python

   CREATE EXTERNAL TABLE IF NOT EXISTS sagemaker_featurestore.identity-feature-group-27-19-33-00 (
     TransactionID INT
     id_01 FLOAT
     id_02 FLOAT
     id_03 FLOAT
     id_04 FLOAT
    ...

.. rubric:: Build a Training Dataset
   :name: bCe9CAVnDLV

Feature Store automatically builds a Amazon Glue Data Catalog when
Feature Groups are created and can optionally be turned off. The
following we show how to create a single training dataset with feature
values from both identity and transaction feature groups created above.
Also, the following shows how to run an Amazon Athena query to join data
stored in the Offline Store from both identity and transaction feature
groups.


To start, create an Athena query using\ ``athena_query()``\ for both
identity and transaction feature groups. The ``table_name`` is the Glue
table that is auto-generated by Feature Store.

.. code:: python

   identity_query = identity_feature_group.athena_query()
   transaction_query = transaction_feature_group.athena_query()

   identity_table = identity_query.table_name
   transaction_table = transaction_query.table_name

.. rubric:: Writing and Executing your Athena Query
   :name: bCe9CArSR5J

You will write your query using SQL on these feature groups, and then
execute the query with the ``.run()`` command and specify your S3 bucket
location for the data set to be saved there.

.. code:: python

   # Athena query
   query_string = 'SELECT * FROM "'+transaction_table+'" LEFT JOIN "'+identity_table+'" ON "'+transaction_table+'".transactionid = "'+identity_table+'".transactionid'

   # run Athena query. The output is loaded to a Pandas dataframe.
   dataset = pd.DataFrame()
   identity_query.run(query_string=query_string, output_location='s3://'+default_s3_bucket_name+'/query_results/')
   identity_query.wait()
   dataset = identity_query.as_dataframe()

From here you can train a model using this data set and then perform
inference.

.. rubric:: Using the Offline Store SDK: Getting Started
   :name: bCe9CA61b79

The Feature Store Offline SDK provides the ability to quickly and easily
build ML-ready datasets for use by ML model training or pre-processing.
The SDK makes it easy to build datasets from SQL join, point-in-time accurate
join, and event range time frames, all without the need to write any SQL code.
This functionality is accessed via the DatasetBuilder class which is the
primary entry point for the SDK functionality.

.. code:: python

   from sagemaker.feature_store.feature_store import FeatureStore

   feature_store = FeatureStore(sagemaker_session=feature_store_session)

.. code:: python

   base_feature_group = identity_feature_group
   target_feature_group = transaction_feature_group

You can create dataset using `create_dataset` of feature store API.
`base` can either be a feature group or a pandas dataframe.

.. code:: python

   result_df, query = feature_store.create_dataset(
      base=base_feature_group,
      output_path=f"s3://{s3_bucket_name}"
   ).to_dataframe()

If you want to join other feature group, you can specify extra
feature group using `with_feature_group` method.

.. code:: python

   dataset_builder = feature_store.create_dataset(
      base=base_feature_group,
      output_path=f"s3://{s3_bucket_name}"
   ).with_feature_group(target_feature_group, record_identifier_name)

   result_df, query = dataset_builder.to_dataframe()

.. rubric:: Using the Offline Store SDK: Configuring the DatasetBuilder
   :name: bCe9CA61b80

How the DatasetBuilder produces the resulting dataframe can be configured
in various ways.

By default the Python SDK will exclude all deleted and duplicate records.
However if you need either of them in returned dataset, you can call
`include_duplicated_records` or `include_deleted_records` when creating
dataset builder.

.. code:: python

   dataset_builder.include_duplicated_records()
   dataset_builder.include_deleted_records()

The DatasetBuilder provides `with_number_of_records_from_query_results` and
`with_number_of_recent_records_by_record_identifier` methods to limit the
number of records returned for the offline snapshot.

`with_number_of_records_from_query_results` will limit the number of records
in the output. For example, when N = 100, only 100 records are going to be
returned in either the csv or dataframe.

.. code:: python

   dataset_builder.with_number_of_records_from_query_results(number_of_records=N)

On the other hand, `with_number_of_recent_records_by_record_identifier` is
used to deal with records which have the same identifier. They are going
to be sorted according to `event_time` and return at most N recent records
in the output.

.. code:: python

   dataset_builder.with_number_of_recent_records_by_record_identifier(number_of_recent_records=N)

Since these functions return the dataset builder, these functions can
be chained.

.. code:: python

   dataset_builder
      .with_number_of_records_from_query_results(number_of_records=N)
      .include_duplicated_records()
      .with_number_of_recent_records_by_record_identifier(number_of_recent_records=N)
      .to_dataframe()

There are additional configurations that can be made for various use cases,
such as time travel and point-in-time join. These are outlined in the
Feature Store `DatasetBuilder API Reference
<https://sagemaker.readthedocs.io/en/stable/api/prep_data/feature_store.html#dataset-builder>`__.

.. rubric:: Delete a feature group
   :name: bCe9CA61b78

You can delete a feature group with the ``delete`` function. Note that it
can take approximately 10-15 minutes to delete an online ``FeatureGroup``
with the ``InMemory`` ``StorageType``.

.. code:: python

   feature_group.delete()

The following code example is from the fraud detection example.

.. code:: python

   identity_feature_group.delete()
   transaction_feature_group.delete()

