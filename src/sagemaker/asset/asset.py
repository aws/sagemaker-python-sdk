# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""This module contains code related to the DataZone SageMaker Asset."""
from __future__ import print_function, absolute_import

from typing import Optional, Dict, List
from enum import Enum
import attr


class AssetTypeIdentifier(Enum):
    """AssetTypeIdentifier class.

    Enumeration for various ML and data asset types.

    """

    SageMakerFeatureGroupAssetType = "amazon.datazone.SageMakerFeatureGroupAssetType"
    SageMakerModelPackageGroupAssetType = "amazon.datazone.SageMakerModelPackageGroupAssetType"
    DataZoneGlueAssetType = "amazon.datazone.GlueTableAssetType"

    @staticmethod
    def value_of(s):
        """Converts a string to a value of the enumeration.

        Args:
            s: An enumeration value as string.

        Raises:
            ValueError: If the combination of arguments specified is not supported.
        """
        for e in AssetTypeIdentifier:
            if e.value == s:
                return e

        raise ValueError("Argument does not match any AssetTypeIdentifiers")


@attr.s
class Asset(object):
    """An asset class representing a DataZone SageMaker ML asset.

    Attributes:
        name (str): The name of the asset.
        type_id (AssetTypeIdentifier): The DataZone type identifier for the asset.
        external_id (str): The external id of the asset, which equals to resource ARN.
        id (str): The id of the asset. We wil update the asset id once asset is created.
        forms_input (Optional[List[Dict]]): The metadata forms of the asset.
    """

    name: str = attr.ib()
    type_id: AssetTypeIdentifier = attr.ib()
    external_id: str = attr.ib()
    id: str = attr.ib(default=None)
    forms_input: Optional[List[Dict]] = attr.ib(default=None)


class ChangeSetAction(Enum):
    """A change set action class."""

    PUBLISH = "PUBLISH"
    UNPUBLISH = "UNPUBLISH"


@attr.s
class Listing(object):
    """A listing class representing a published DataZone SageMaker ML asset.

    Attributes:
        id (str): The id of the listing.
        revision (str): The revision of the listing.
        status (str): The status of the listing
    """

    id: str = attr.ib()
    revision: str = attr.ib()
    status: str = attr.ib(default=None)
