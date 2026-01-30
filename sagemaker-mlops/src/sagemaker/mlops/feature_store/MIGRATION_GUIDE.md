# SageMaker FeatureStore V2 to V3 Migration Guide

## Overview

V3 uses **sagemaker-core** as the foundation, which provides:
- Pydantic-based shapes with automatic serialization
- Resource classes that manage boto clients internally
- No need for explicit Session management in most cases

## File Mapping

| V2 File | V3 File | Notes |
|---------|---------|-------|
| `feature_group.py` | Re-exported from `sagemaker_core.main.resources` | No wrapper class needed |
| `feature_store.py` | Re-exported from `sagemaker_core.main.resources` | `FeatureStore.search()` available |
| `feature_definition.py` | `feature_definition.py` | Helper factories retained |
| `feature_utils.py` | `feature_utils.py` | Standalone functions |
| `inputs.py` | `inputs.py` | Enums only (shapes from core) |
| `dataset_builder.py` | `dataset_builder.py` | Converted to dataclass |
| N/A | `athena_query.py` | Extracted from feature_group.py |
| N/A | `ingestion_manager_pandas.py` | Extracted from feature_group.py |

---

## FeatureGroup Operations

### Create FeatureGroup

**V2:**
```python
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.session import Session

session = Session()
fg = FeatureGroup(name="my-fg", sagemaker_session=session)
fg.load_feature_definitions(data_frame=df)
fg.create(
    s3_uri="s3://bucket/prefix",
    record_identifier_name="id",
    event_time_feature_name="ts",
    role_arn=role,
    enable_online_store=True,
)
```

**V3:**
```python
from sagemaker.mlops.feature_store import (
    FeatureGroup,
    OnlineStoreConfig,
    OfflineStoreConfig,
    S3StorageConfig,
    load_feature_definitions_from_dataframe,
)

feature_defs = load_feature_definitions_from_dataframe(df)

FeatureGroup.create(
    feature_group_name="my-fg",
    feature_definitions=feature_defs,
    record_identifier_feature_name="id",
    event_time_feature_name="ts",
    role_arn=role,
    online_store_config=OnlineStoreConfig(enable_online_store=True),
    offline_store_config=OfflineStoreConfig(
        s3_storage_config=S3StorageConfig(s3_uri="s3://bucket/prefix")
    ),
)
```

### Get/Describe FeatureGroup

**V2:**
```python
fg = FeatureGroup(name="my-fg", sagemaker_session=session)
response = fg.describe()
```

**V3:**
```python
from sagemaker.mlops.feature_store import FeatureGroup

fg = FeatureGroup.get(feature_group_name="my-fg")
# fg is now a typed object with attributes:
# fg.feature_group_name, fg.feature_definitions, fg.offline_store_config, etc.
```

### Delete FeatureGroup

**V2:**
```python
fg.delete()
```

**V3:**
```python
FeatureGroup(feature_group_name="my-fg").delete()
# or
fg = FeatureGroup.get(feature_group_name="my-fg")
fg.delete()
```

### Update FeatureGroup

**V2:**
```python
fg.update(
    feature_additions=[FeatureDefinition("new_col", FeatureTypeEnum.STRING)],
    throughput_config=ThroughputConfigUpdate(mode=ThroughputModeEnum.ON_DEMAND),
)
```

**V3:**
```python
from sagemaker.mlops.feature_store import FeatureGroup, ThroughputConfig

fg = FeatureGroup.get(feature_group_name="my-fg")
fg.update(
    feature_additions=[{"FeatureName": "new_col", "FeatureType": "String"}],
    throughput_config=ThroughputConfig(throughput_mode="OnDemand"),
)
```

---

## Record Operations

### Put Record

**V2:**
```python
from sagemaker.feature_store.inputs import FeatureValue

fg.put_record(
    record=[
        FeatureValue(feature_name="id", value_as_string="123"),
        FeatureValue(feature_name="name", value_as_string="John"),
    ],
    target_stores=[TargetStoreEnum.ONLINE_STORE],
)
```

**V3:**
```python
from sagemaker.mlops.feature_store import FeatureGroup, FeatureValue

FeatureGroup(feature_group_name="my-fg").put_record(
    record=[
        FeatureValue(feature_name="id", value_as_string="123"),
        FeatureValue(feature_name="name", value_as_string="John"),
    ],
    target_stores=["OnlineStore"],  # strings, not enums
)
```

### Get Record

**V2:**
```python
response = fg.get_record(record_identifier_value_as_string="123")
```

**V3:**
```python
response = FeatureGroup(feature_group_name="my-fg").get_record(
    record_identifier_value_as_string="123"
)
```

### Delete Record

**V2:**
```python
fg.delete_record(
    record_identifier_value_as_string="123",
    event_time="2024-01-15T00:00:00Z",
    deletion_mode=DeletionModeEnum.SOFT_DELETE,
)
```

**V3:**
```python
FeatureGroup(feature_group_name="my-fg").delete_record(
    record_identifier_value_as_string="123",
    event_time="2024-01-15T00:00:00Z",
    deletion_mode="SoftDelete",  # string, not enum
)
```

### Batch Get Record

**V2:**
```python
from sagemaker.feature_store.feature_store import FeatureStore
from sagemaker.feature_store.inputs import Identifier

fs = FeatureStore(sagemaker_session=session)
response = fs.batch_get_record(
    identifiers=[
        Identifier(feature_group_name="my-fg", record_identifiers_value_as_string=["123", "456"])
    ]
)
```

**V3:**
```python
from sagemaker.mlops.feature_store import FeatureGroup

response = FeatureGroup(feature_group_name="my-fg").batch_get_record(
    identifiers=[
        {"FeatureGroupName": "my-fg", "RecordIdentifiersValueAsString": ["123", "456"]}
    ]
)
```

---

## DataFrame Ingestion

**V2:**
```python
fg.ingest(data_frame=df, max_workers=4, max_processes=2, wait=True)
```

**V3:**
```python
from sagemaker.mlops.feature_store import ingest_dataframe

manager = ingest_dataframe(
    feature_group_name="my-fg",
    data_frame=df,
    max_workers=4,
    max_processes=2,
    wait=True,
)
# Access failed rows: manager.failed_rows
```

---

## Athena Query

**V2:**
```python
query = fg.athena_query()
query.run(query_string="SELECT * FROM ...", output_location="s3://...")
query.wait()
df = query.as_dataframe()
```

**V3:**
```python
from sagemaker.mlops.feature_store import create_athena_query

query = create_athena_query("my-fg", session)
query.run(query_string="SELECT * FROM ...", output_location="s3://...")
query.wait()
df = query.as_dataframe()
```

---

## Hive DDL Generation

**V2:**
```python
ddl = fg.as_hive_ddl(database="mydb", table_name="mytable")
```

**V3:**
```python
from sagemaker.mlops.feature_store import as_hive_ddl

ddl = as_hive_ddl("my-fg", database="mydb", table_name="mytable")
```

---

## Feature Definitions

**V2:**
```python
fg.load_feature_definitions(data_frame=df)
# Modifies fg.feature_definitions in place
```

**V3:**
```python
from sagemaker.mlops.feature_store import load_feature_definitions_from_dataframe

defs = load_feature_definitions_from_dataframe(df)
# Returns list, doesn't modify any object
```

### Using Helper Factories

**V2 & V3 (same):**
```python
from sagemaker.mlops.feature_store import (
    FractionalFeatureDefinition,
    IntegralFeatureDefinition,
    StringFeatureDefinition,
    VectorCollectionType,
)

defs = [
    IntegralFeatureDefinition("id"),
    StringFeatureDefinition("name"),
    FractionalFeatureDefinition("embedding", VectorCollectionType(128)),
]
```

---

## Search

**V2:**
```python
from sagemaker.feature_store.feature_store import FeatureStore
from sagemaker.feature_store.inputs import Filter, ResourceEnum

fs = FeatureStore(sagemaker_session=session)
response = fs.search(
    resource=ResourceEnum.FEATURE_GROUP,
    filters=[Filter(name="FeatureGroupName", value="my-prefix", operator=FilterOperatorEnum.CONTAINS)],
)
```

**V3:**
```python
from sagemaker.mlops.feature_store import FeatureStore, Filter, SearchExpression

response = FeatureStore.search(
    resource="FeatureGroup",
    search_expression=SearchExpression(
        filters=[Filter(name="FeatureGroupName", value="my-prefix", operator="Contains")]
    ),
)
```

---

## Feature Metadata

**V2:**
```python
fg.describe_feature_metadata(feature_name="my-feature")
fg.update_feature_metadata(feature_name="my-feature", description="Updated desc")
fg.list_parameters_for_feature_metadata(feature_name="my-feature")
```

**V3:**
```python
from sagemaker.mlops.feature_store import FeatureMetadata

# Get metadata
metadata = FeatureMetadata.get(feature_group_name="my-fg", feature_name="my-feature")
print(metadata.description)
print(metadata.parameters)

# Update metadata
metadata.update(description="Updated desc")
```

---

## Dataset Builder

**V2:**
```python
from sagemaker.feature_store.feature_store import FeatureStore

fs = FeatureStore(sagemaker_session=session)
builder = fs.create_dataset(
    base=fg,
    output_path="s3://bucket/output",
)
builder.with_feature_group(other_fg, target_feature_name_in_base="id")
builder.point_in_time_accurate_join()
df, query = builder.to_dataframe()
```

**V3:**
```python
from sagemaker.mlops.feature_store import create_dataset, FeatureGroup

fg = FeatureGroup.get(feature_group_name="my-fg")
other_fg = FeatureGroup.get(feature_group_name="other-fg")

builder = create_dataset(
    base=fg,
    output_path="s3://bucket/output",
    session=session,
)
builder.with_feature_group(other_fg, target_feature_name_in_base="id")
builder.point_in_time_accurate_join()
df, query = builder.to_dataframe()
```

---

## Config Objects (Shapes)

**V2:**
```python
from sagemaker.feature_store.inputs import (
    OnlineStoreConfig,
    OfflineStoreConfig,
    S3StorageConfig,
    TtlDuration,
)

config = OnlineStoreConfig(enable_online_store=True, ttl_duration=TtlDuration(unit="Hours", value=24))
config.to_dict()  # Manual serialization required
```

**V3:**
```python
from sagemaker.mlops.feature_store import (
    OnlineStoreConfig,
    OfflineStoreConfig,
    S3StorageConfig,
    TtlDuration,
)

config = OnlineStoreConfig(enable_online_store=True, ttl_duration=TtlDuration(unit="Hours", value=24))
# No to_dict() needed - Pydantic handles serialization automatically
```

---

## Key Differences Summary

| Aspect | V2 | V3 |
|--------|----|----|
| **Session** | Required for most operations | Optional - core manages clients |
| **FeatureGroup** | Wrapper class with session | Direct core resource class |
| **Shapes** | `@attr.s` with `to_dict()` | Pydantic with auto-serialization |
| **Enums** | `TargetStoreEnum.ONLINE_STORE.value` | Just use strings: `"OnlineStore"` |
| **Methods** | Instance methods on FeatureGroup | Standalone functions + core methods |
| **Ingestion** | `fg.ingest(df)` | `ingest_dataframe(name, df)` |
| **Athena** | `fg.athena_query()` | `create_athena_query(name, session)` |
| **DDL** | `fg.as_hive_ddl()` | `as_hive_ddl(name)` |
| **Feature Defs** | `fg.load_feature_definitions(df)` | `load_feature_definitions_from_dataframe(df)` |
| **Imports** | Multiple modules | Single `__init__.py` re-exports all |

---

## Missing in V3 (Intentionally)

These V2 features are **not wrapped** because core provides them directly:

- `FeatureGroup.create()` - use `FeatureGroup.create()` from core
- `FeatureGroup.delete()` - use `FeatureGroup(...).delete()` from core
- `FeatureGroup.describe()` - use `FeatureGroup.get()` from core (returns typed object)
- `FeatureGroup.update()` - use `FeatureGroup(...).update()` from core
- `FeatureGroup.put_record()` - use `FeatureGroup(...).put_record()` from core
- `FeatureGroup.get_record()` - use `FeatureGroup(...).get_record()` from core
- `FeatureGroup.delete_record()` - use `FeatureGroup(...).delete_record()` from core
- `FeatureGroup.batch_get_record()` - use `FeatureGroup(...).batch_get_record()` from core
- `FeatureStore.search()` - use `FeatureStore.search()` from core
- `FeatureStore.list_feature_groups()` - use `FeatureGroup.get_all()` from core
- All config shapes (`OnlineStoreConfig`, etc.) - re-exported from core

---

## Import Cheatsheet

```python
# V3 - Everything from one place
from sagemaker.mlops.feature_store import (
    # Resources (from core)
    FeatureGroup,
    FeatureStore,
    FeatureMetadata,
    
    # Shapes (from core)
    OnlineStoreConfig,
    OfflineStoreConfig,
    S3StorageConfig,
    DataCatalogConfig,
    TtlDuration,
    FeatureValue,
    FeatureParameter,
    ThroughputConfig,
    Filter,
    SearchExpression,
    
    # Enums (local)
    TargetStoreEnum,
    OnlineStoreStorageTypeEnum,
    TableFormatEnum,
    DeletionModeEnum,
    ThroughputModeEnum,
    
    # Feature Definition helpers (local)
    FeatureDefinition,
    FractionalFeatureDefinition,
    IntegralFeatureDefinition,
    StringFeatureDefinition,
    VectorCollectionType,
    
    # Utility functions (local)
    create_athena_query,
    as_hive_ddl,
    load_feature_definitions_from_dataframe,
    ingest_dataframe,
    create_dataset,
    
    # Classes (local)
    DatasetBuilder,
)
```
