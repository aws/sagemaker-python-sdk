"""Enums for FeatureStore operations."""
from enum import Enum

class TargetStoreEnum(Enum):
    """Store types for put_record."""
    ONLINE_STORE = "OnlineStore"
    OFFLINE_STORE = "OfflineStore"

class OnlineStoreStorageTypeEnum(Enum):
    """Storage types for online store."""
    STANDARD = "Standard"
    IN_MEMORY = "InMemory"

class TableFormatEnum(Enum):
    """Offline store table formats."""
    GLUE = "Glue"
    ICEBERG = "Iceberg"

class ResourceEnum(Enum):
    """Resource types for search."""
    FEATURE_GROUP = "FeatureGroup"
    FEATURE_METADATA = "FeatureMetadata"

class SearchOperatorEnum(Enum):
    """Search operators."""
    AND = "And"
    OR = "Or"

class SortOrderEnum(Enum):
    """Sort orders."""
    ASCENDING = "Ascending"
    DESCENDING = "Descending"

class FilterOperatorEnum(Enum):
    """Filter operators."""
    EQUALS = "Equals"
    NOT_EQUALS = "NotEquals"
    GREATER_THAN = "GreaterThan"
    GREATER_THAN_OR_EQUAL_TO = "GreaterThanOrEqualTo"
    LESS_THAN = "LessThan"
    LESS_THAN_OR_EQUAL_TO = "LessThanOrEqualTo"
    CONTAINS = "Contains"
    EXISTS = "Exists"
    NOT_EXISTS = "NotExists"
    IN = "In"

class DeletionModeEnum(Enum):
    """Deletion modes for delete_record."""
    SOFT_DELETE = "SoftDelete"
    HARD_DELETE = "HardDelete"

class ExpirationTimeResponseEnum(Enum):
    """ExpiresAt response toggle."""
    DISABLED = "Disabled"
    ENABLED = "Enabled"

class ThroughputModeEnum(Enum):
    """Throughput modes for feature group."""
    ON_DEMAND = "OnDemand"
    PROVISIONED = "Provisioned"