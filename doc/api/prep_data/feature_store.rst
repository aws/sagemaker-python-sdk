Feature Store APIs
------------------

Feature Group
*************

.. autoclass:: sagemaker.feature_store.feature_group.FeatureGroup
    :members:
    :show-inheritance:

.. autoclass:: sagemaker.feature_store.feature_group.AthenaQuery
    :members:
    :show-inheritance:

.. autoclass:: sagemaker.feature_store.feature_group.IngestionManagerPandas
    :members:
    :show-inheritance:


Feature Definition
******************

.. autoclass:: sagemaker.feature_store.feature_definition.FeatureDefinition
    :members:
    :show-inheritance:

.. autoclass:: sagemaker.feature_store.feature_definition.FractionalFeatureDefinition
    :members:
    :show-inheritance:

.. autoclass:: sagemaker.feature_store.feature_definition.IntegralFeatureDefinition
    :members:
    :show-inheritance:

.. autoclass:: sagemaker.feature_store.feature_definition.StringFeatureDefinition
    :members:
    :show-inheritance:

.. autoclass:: sagemaker.feature_store.feature_definition.FeatureTypeEnum
    :members:
    :show-inheritance:

.. autoclass:: sagemaker.feature_store.feature_definition.CollectionTypeEnum
    :members:
    :show-inheritance:

.. autoclass:: sagemaker.feature_store.feature_definition.CollectionType
    :members:
    :show-inheritance:

.. autoclass:: sagemaker.feature_store.feature_definition.ListCollectionType
    :members:
    :show-inheritance:

.. autoclass:: sagemaker.feature_store.feature_definition.SetCollectionType
    :members:
    :show-inheritance:

.. autoclass:: sagemaker.feature_store.feature_definition.VectorCollectionType
    :members:
    :show-inheritance:


Inputs
******

.. autoclass:: sagemaker.feature_store.inputs.Config
    :members:
    :show-inheritance:

.. autoclass:: sagemaker.feature_store.inputs.DataCatalogConfig
    :members:
    :show-inheritance:

.. autoclass:: sagemaker.feature_store.inputs.OfflineStoreConfig
    :members:
    :show-inheritance:

.. autoclass:: sagemaker.feature_store.inputs.ThroughputConfig
    :members:
    :show-inheritance:

.. autoclass:: sagemaker.feature_store.inputs.ThroughputConfigUpdate
    :members:
    :show-inheritance:

.. autoclass:: sagemaker.feature_store.inputs.OnlineStoreConfig
    :members:
    :show-inheritance:

.. autoclass:: sagemaker.feature_store.inputs.OnlineStoreSecurityConfig
    :members:
    :show-inheritance:

.. autoclass:: sagemaker.feature_store.inputs.TtlDuration
    :members:
    :show-inheritance:

.. autoclass:: sagemaker.feature_store.inputs.S3StorageConfig
    :members:
    :show-inheritance:

.. autoclass:: sagemaker.feature_store.inputs.FeatureValue
    :members:
    :show-inheritance:

.. autoclass:: sagemaker.feature_store.inputs.TableFormatEnum
    :members:
    :show-inheritance:

.. autoclass:: sagemaker.feature_store.inputs.OnlineStoreStorageTypeEnum
    :members:
    :show-inheritance:

.. autoclass:: sagemaker.feature_store.inputs.ThroughputModeEnum
    :members:
    :show-inheritance:

.. autoclass:: sagemaker.feature_store.inputs.ResourceEnum
    :members:
    :show-inheritance:

.. autoclass:: sagemaker.feature_store.inputs.SearchOperatorEnum
    :members:
    :show-inheritance:

.. autoclass:: sagemaker.feature_store.inputs.SortOrderEnum
    :members:
    :show-inheritance:

.. autoclass:: sagemaker.feature_store.inputs.FilterOperatorEnum
    :members:
    :show-inheritance:

.. autoclass:: sagemaker.feature_store.inputs.Filter
    :members:
    :show-inheritance:

.. autoclass:: sagemaker.feature_store.inputs.Identifier
    :members:
    :show-inheritance:

.. autoclass:: sagemaker.feature_store.inputs.FeatureParameter
    :members:
    :show-inheritance:


Dataset Builder
***************

.. autoclass:: sagemaker.feature_store.dataset_builder.DatasetBuilder
    :members:
    :show-inheritance:


Feature Store
*************

.. autoclass:: sagemaker.feature_store.feature_store.FeatureStore
    :members:
    :show-inheritance:


@feature_processor Decorator
****************************

.. autodecorator:: sagemaker.feature_store.feature_processor.feature_processor


Feature Processor Data Source
*****************************

.. autoclass:: sagemaker.feature_store.feature_processor.FeatureGroupDataSource
    :members:
    :show-inheritance:

.. autoclass:: sagemaker.feature_store.feature_processor.CSVDataSource
    :members:
    :show-inheritance:

.. autoclass:: sagemaker.feature_store.feature_processor.ParquetDataSource
    :members:
    :show-inheritance:

.. autoclass:: sagemaker.feature_store.feature_processor.PySparkDataSource
    :members:
    :show-inheritance:


Feature Processor Scheduler and Triggers
****************************************

.. automethod:: sagemaker.feature_store.feature_processor.to_pipeline

.. automethod:: sagemaker.feature_store.feature_processor.schedule

.. automethod:: sagemaker.feature_store.feature_processor.execute

.. automethod:: sagemaker.feature_store.feature_processor.delete_schedule

.. automethod:: sagemaker.feature_store.feature_processor.describe

.. automethod:: sagemaker.feature_store.feature_processor.list_pipelines

.. automethod:: sagemaker.feature_store.feature_processor.put_trigger

.. automethod:: sagemaker.feature_store.feature_processor.enable_trigger

.. automethod:: sagemaker.feature_store.feature_processor.disable_trigger

.. automethod:: sagemaker.feature_store.feature_processor.delete_trigger

