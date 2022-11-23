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
"""Unit tests for ClusterConfig module"""

from __future__ import absolute_import

from sagemaker.workflow.cluster_config import (
    Application,
    AutoScalingPolicy,
    BootstrapActionConfig,
    CloudWatchAlarmDefinition,
    ClusterConfig,
    ComputeLimits,
    Configuration,
    EbsBlockDeviceConfig,
    EbsConfiguration,
    InstanceFleetConfig,
    InstanceFleetProvisioningSpecifications,
    InstanceGroupConfig,
    InstanceTypeConfig,
    JobFlowInstancesConfig,
    KerberosAttributes,
    ManagedScalingPolicy,
    MetricDimension,
    OnDemandCapacityReservationOptions,
    OnDemandProvisioningSpecification,
    PlacementGroupConfig,
    PlacementType,
    ScalingAction,
    ScalingConstraints,
    ScalingRule,
    ScalingTrigger,
    ScriptBootstrapActionConfig,
    SimpleScalingPolicyConfiguration,
    SpotProvisioningSpecification,
    SupportedProductConfig,
    Tag,
    VolumeSpecification,
)

g_application = Application(
    additional_info={"key1": "value1", "key2": "value2"},
    args=["arg1", "arg2"],
    name="name",
    version="version",
)


def test_Application():
    application = g_application
    assert application.to_request() == {
        "AdditionalInfo": {"key1": "value1", "key2": "value2"},
        "Args": ["arg1", "arg2"],
        "Name": "name",
        "Version": "version",
    }


g_script_bootstrap_action_config = ScriptBootstrapActionConfig(args=["arg1", "arg2"], path="path")


def test_ScriptBootstrapActionConfig():
    script_bootstrap_action_config = g_script_bootstrap_action_config
    assert script_bootstrap_action_config.to_request() == {"Args": ["arg1", "arg2"], "Path": "path"}


g_bootstrap_action_config = BootstrapActionConfig(
    name="name", script_bootstrap_action=g_script_bootstrap_action_config
)


def test_BootstrapActionConfig():
    bootstrap_action_config = g_bootstrap_action_config
    assert bootstrap_action_config.to_request() == {
        "Name": "name",
        "ScriptBootstrapAction": {"Args": ["arg1", "arg2"], "Path": "path"},
    }


g_configuration = Configuration(
    classification="classification",
    configurations=[Configuration(classification="classification1", properties={"key1": "value1"})],
    properties={"key": "value"},
)


def test_Configuration():
    configuration = g_configuration

    assert configuration.to_request() == {
        "Classification": "classification",
        "Configurations": [{"Classification": "classification1", "Properties": {"key1": "value1"}}],
        "Properties": {"key": "value"},
    }


def test_VolumeSpecification():
    volume_specification = VolumeSpecification(
        iops=1, size_in_gb=2, throughput=3, volume_type="four"
    )

    assert volume_specification.to_request() == {
        "Iops": 1,
        "SizeInGB": 2,
        "Throughput": 3,
        "VolumeType": "four",
    }


def test_EbsBlockDeviceConfig():
    ebs_block_device_config = EbsBlockDeviceConfig(
        volume_specification=VolumeSpecification(
            iops=1, size_in_gb=2, throughput=3, volume_type="four"
        ),
        volumes_per_instance=123,
    )
    assert ebs_block_device_config.to_request() == {
        "VolumeSpecification": {"Iops": 1, "SizeInGB": 2, "Throughput": 3, "VolumeType": "four"},
        "VolumesPerInstance": 123,
    }


g_ebs_configuration = EbsConfiguration(
    ebs_block_device_configs=[
        EbsBlockDeviceConfig(
            volume_specification=VolumeSpecification(
                iops=1, size_in_gb=2, throughput=3, volume_type="four"
            ),
            volumes_per_instance=123,
        )
    ],
    ebs_optimized=False,
)


def test_EbsConfiguration():
    ebs_configuration = g_ebs_configuration

    assert ebs_configuration.to_request() == {
        "EbsBlockDeviceConfigs": [
            {
                "VolumeSpecification": {
                    "Iops": 1,
                    "SizeInGB": 2,
                    "Throughput": 3,
                    "VolumeType": "four",
                },
                "VolumesPerInstance": 123,
            }
        ],
        "EbsOptimized": False,
    }


g_instance_type_config = InstanceTypeConfig(
    bid_price="one",
    bid_price_as_percentage_of_on_demand_price=12,
    configurations=[g_configuration],
    custom_ami_id="someId",
    ebs_configuration=g_ebs_configuration,
    instance_type="someInstance",
    weighted_capacity=9,
)


def test_InstanceTypeConfig():
    instance_type_config = g_instance_type_config

    assert instance_type_config.to_request() == {
        "BidPrice": "one",
        "BidPriceAsPercentageOfOnDemandPrice": 12,
        "Configurations": [
            {
                "Classification": "classification",
                "Configurations": [
                    {"Classification": "classification1", "Properties": {"key1": "value1"}}
                ],
                "Properties": {"key": "value"},
            }
        ],
        "CustomAmiId": "someId",
        "EbsConfiguration": {
            "EbsBlockDeviceConfigs": [
                {
                    "VolumeSpecification": {
                        "Iops": 1,
                        "SizeInGB": 2,
                        "Throughput": 3,
                        "VolumeType": "four",
                    },
                    "VolumesPerInstance": 123,
                }
            ],
            "EbsOptimized": False,
        },
        "InstanceType": "someInstance",
        "WeightedCapacity": 9,
    }


def test_OnDemandCapacityReservationOptions():
    on_demand_capacity_reservation_options = OnDemandCapacityReservationOptions(
        capacity_reservation_preference="open",
        capacity_reservation_resource_group_arn="groupArn",
        usage_strategy="lowest-price",
    )
    assert on_demand_capacity_reservation_options.to_request() == {
        "CapacityReservationPreference": "open",
        "CapacityReservationResourceGroupArn": "groupArn",
        "UsageStrategy": "lowest-price",
    }


g_on_demand_provisioning_specification = OnDemandProvisioningSpecification(
    allocation_strategy="lowest-price",
    capacity_reservation_options=OnDemandCapacityReservationOptions(
        capacity_reservation_preference="open",
        capacity_reservation_resource_group_arn="groupArn",
        usage_strategy="lowest-price",
    ),
)


def test_OnDemandProvisioningSpecification():
    on_demand_provisioning_specification = g_on_demand_provisioning_specification
    assert on_demand_provisioning_specification.to_request() == {
        "AllocationStrategy": "lowest-price",
        "CapacityReservationOptions": {
            "CapacityReservationPreference": "open",
            "CapacityReservationResourceGroupArn": "groupArn",
            "UsageStrategy": "lowest-price",
        },
    }


g_spot_provisioning_specification = SpotProvisioningSpecification(
    allocation_strategy="capacity-optimized",
    block_duration_minutes=13,
    timeout_action="TERMINATE_CLUSTER",
    timeout_duration_minutes=88,
)


def test_SpotProvisioningSpecification():
    spot_provisioning_specification = g_spot_provisioning_specification

    assert spot_provisioning_specification.to_request() == {
        "AllocationStrategy": "capacity-optimized",
        "BlockDurationMinutes": 13,
        "TimeoutAction": "TERMINATE_CLUSTER",
        "TimeoutDurationMinutes": 88,
    }


g_instance_fleet_provisioning_specifications = InstanceFleetProvisioningSpecifications(
    on_demand_specification=g_on_demand_provisioning_specification,
    spot_specification=g_spot_provisioning_specification,
)


def test_InstanceFleetProvisioningSpecifications():
    instance_fleet_provisioning_specifications = g_instance_fleet_provisioning_specifications
    assert instance_fleet_provisioning_specifications.to_request() == {
        "OnDemandSpecification": {
            "AllocationStrategy": "lowest-price",
            "CapacityReservationOptions": {
                "CapacityReservationPreference": "open",
                "CapacityReservationResourceGroupArn": "groupArn",
                "UsageStrategy": "lowest-price",
            },
        },
        "SpotSpecification": {
            "AllocationStrategy": "capacity-optimized",
            "BlockDurationMinutes": 13,
            "TimeoutAction": "TERMINATE_CLUSTER",
            "TimeoutDurationMinutes": 88,
        },
    }


g_instance_fleet_config = InstanceFleetConfig(
    instance_fleet_type="fleetType",
    instance_type_configs=[g_instance_type_config],
    launch_specifications=g_instance_fleet_provisioning_specifications,
    name="someName",
    target_on_demand_capacity=3,
    target_spot_capacity=4,
)


def test_InstanceFleetConfig():
    instance_fleet_config = g_instance_fleet_config

    assert instance_fleet_config.to_request() == {
        "InstanceFleetType": "fleetType",
        "InstanceTypeConfigs": [
            {
                "BidPrice": "one",
                "BidPriceAsPercentageOfOnDemandPrice": 12,
                "Configurations": [
                    {
                        "Classification": "classification",
                        "Configurations": [
                            {"Classification": "classification1", "Properties": {"key1": "value1"}}
                        ],
                        "Properties": {"key": "value"},
                    }
                ],
                "CustomAmiId": "someId",
                "EbsConfiguration": {
                    "EbsBlockDeviceConfigs": [
                        {
                            "VolumeSpecification": {
                                "Iops": 1,
                                "SizeInGB": 2,
                                "Throughput": 3,
                                "VolumeType": "four",
                            },
                            "VolumesPerInstance": 123,
                        }
                    ],
                    "EbsOptimized": False,
                },
                "InstanceType": "someInstance",
                "WeightedCapacity": 9,
            }
        ],
        "LaunchSpecifications": {
            "OnDemandSpecification": {
                "AllocationStrategy": "lowest-price",
                "CapacityReservationOptions": {
                    "CapacityReservationPreference": "open",
                    "CapacityReservationResourceGroupArn": "groupArn",
                    "UsageStrategy": "lowest-price",
                },
            },
            "SpotSpecification": {
                "AllocationStrategy": "capacity-optimized",
                "BlockDurationMinutes": 13,
                "TimeoutAction": "TERMINATE_CLUSTER",
                "TimeoutDurationMinutes": 88,
            },
        },
        "Name": "someName",
        "TargetOnDemandCapacity": 3,
        "TargetSpotCapacity": 4,
    }


def test_ScalingConstraints():
    scaling_constraints = ScalingConstraints(max_capacity=99, min_capacity=1)

    assert scaling_constraints.to_request() == {"MaxCapacity": 99, "MinCapacity": 1}


def test_SimpleScalingPolicyConfiguration():
    simple_scaling_policy_configurations = SimpleScalingPolicyConfiguration(
        adjustment_type="CHANGE_IN_CAPACITY", cool_down=9, scaling_adjustment=2
    )
    assert simple_scaling_policy_configurations.to_request() == {
        "AdjustmentType": "CHANGE_IN_CAPACITY",
        "CoolDown": 9,
        "ScalingAdjustment": 2,
    }


def test_ScalingAction():
    scaling_action = ScalingAction(
        market="SPOT",
        simple_scaling_policy_configuration=SimpleScalingPolicyConfiguration(
            adjustment_type="CHANGE_IN_CAPACITY", cool_down=9, scaling_adjustment=2
        ),
    )
    assert scaling_action.to_request() == {
        "Market": "SPOT",
        "SimpleScalingPolicyConfiguration": {
            "AdjustmentType": "CHANGE_IN_CAPACITY",
            "CoolDown": 9,
            "ScalingAdjustment": 2,
        },
    }


g_metric_dimension = MetricDimension(key="key1", value="val1")


def test_MetricDimension():
    metric_dimension = g_metric_dimension
    assert metric_dimension.to_request() == {"Key": "key1", "Value": "val1"}


g_cloud_watch_alarm_definition = CloudWatchAlarmDefinition(
    comparison_operator="LESS_THAN",
    dimensions=[MetricDimension(key="key1", value="val1")],
    evaluation_periods=1,
    metric_name="MetricName",
    namespace="Namespace",
    period=300,
    statistic="AVERAGE",
    threshold=2.2,
    unit="SECONDS",
)


def test_CloudWatchAlarmDefinition():
    cloud_watch_alarm_definition = g_cloud_watch_alarm_definition
    assert cloud_watch_alarm_definition.to_request() == {
        "ComparisonOperator": "LESS_THAN",
        "Dimensions": [{"Key": "key1", "Value": "val1"}],
        "EvaluationPeriods": 1,
        "MetricName": "MetricName",
        "Namespace": "Namespace",
        "Period": 300,
        "Statistic": "AVERAGE",
        "Threshold": 2.2,
        "Unit": "SECONDS",
    }


g_scaling_trigger = ScalingTrigger(cloud_watch_alarm_definition=g_cloud_watch_alarm_definition)


def test_ScalingTrigger():
    scaling_trigger = g_scaling_trigger
    assert scaling_trigger.to_request() == {
        "CloudWatchAlarmDefinition": {
            "ComparisonOperator": "LESS_THAN",
            "Dimensions": [{"Key": "key1", "Value": "val1"}],
            "EvaluationPeriods": 1,
            "MetricName": "MetricName",
            "Namespace": "Namespace",
            "Period": 300,
            "Statistic": "AVERAGE",
            "Threshold": 2.2,
            "Unit": "SECONDS",
        }
    }


g_scaling_rule = ScalingRule(
    action=ScalingAction(
        market="SPOT",
        simple_scaling_policy_configuration=SimpleScalingPolicyConfiguration(
            adjustment_type="CHANGE_IN_CAPACITY", cool_down=9, scaling_adjustment=2
        ),
    ),
    description="Description",
    name="Name",
    trigger=ScalingTrigger(
        cloud_watch_alarm_definition=CloudWatchAlarmDefinition(
            comparison_operator="LESS_THAN",
            dimensions=[MetricDimension(key="key1", value="val1")],
            evaluation_periods=1,
            metric_name="MetricName",
            namespace="Namespace",
            period=300,
            statistic="AVERAGE",
            threshold=2.2,
            unit="SECONDS",
        )
    ),
)


def test_ScalingRule():
    scaling_rule = g_scaling_rule

    assert scaling_rule.to_request() == {
        "Action": {
            "Market": "SPOT",
            "SimpleScalingPolicyConfiguration": {
                "AdjustmentType": "CHANGE_IN_CAPACITY",
                "CoolDown": 9,
                "ScalingAdjustment": 2,
            },
        },
        "Description": "Description",
        "Name": "Name",
        "Trigger": {
            "CloudWatchAlarmDefinition": {
                "ComparisonOperator": "LESS_THAN",
                "Dimensions": [{"Key": "key1", "Value": "val1"}],
                "EvaluationPeriods": 1,
                "MetricName": "MetricName",
                "Namespace": "Namespace",
                "Period": 300,
                "Statistic": "AVERAGE",
                "Threshold": 2.2,
                "Unit": "SECONDS",
            }
        },
    }


g_auto_scaling_policy = AutoScalingPolicy(
    constraints=ScalingConstraints(max_capacity=99, min_capacity=1),
    rules=[
        ScalingRule(
            action=ScalingAction(
                market="SPOT",
                simple_scaling_policy_configuration=SimpleScalingPolicyConfiguration(
                    adjustment_type="CHANGE_IN_CAPACITY", cool_down=9, scaling_adjustment=2
                ),
            ),
            description="Description",
            name="Name",
            trigger=ScalingTrigger(
                cloud_watch_alarm_definition=CloudWatchAlarmDefinition(
                    comparison_operator="LESS_THAN",
                    dimensions=[MetricDimension(key="key1", value="val1")],
                    evaluation_periods=1,
                    metric_name="MetricName",
                    namespace="Namespace",
                    period=300,
                    statistic="AVERAGE",
                    threshold=2.2,
                    unit="SECONDS",
                )
            ),
        )
    ],
)


def test_AutoScalingPolicy():
    auto_scaling_policy = g_auto_scaling_policy
    assert auto_scaling_policy.to_request() == {
        "Constraints": {"MaxCapacity": 99, "MinCapacity": 1},
        "Rules": [
            {
                "Action": {
                    "Market": "SPOT",
                    "SimpleScalingPolicyConfiguration": {
                        "AdjustmentType": "CHANGE_IN_CAPACITY",
                        "CoolDown": 9,
                        "ScalingAdjustment": 2,
                    },
                },
                "Description": "Description",
                "Name": "Name",
                "Trigger": {
                    "CloudWatchAlarmDefinition": {
                        "ComparisonOperator": "LESS_THAN",
                        "Dimensions": [{"Key": "key1", "Value": "val1"}],
                        "EvaluationPeriods": 1,
                        "MetricName": "MetricName",
                        "Namespace": "Namespace",
                        "Period": 300,
                        "Statistic": "AVERAGE",
                        "Threshold": 2.2,
                        "Unit": "SECONDS",
                    }
                },
            }
        ],
    }


g_instance_group_config = InstanceGroupConfig(
    auto_scaling_policy=g_auto_scaling_policy,
    bid_price="1234",
    configurations=[g_configuration],
    custom_ami_id="CustomAmiId",
    ebs_configuration=g_ebs_configuration,
    instance_count=1,
    instance_role="MASTER",
    instance_type="InstanceType",
    market="SPOT",
    name="Name",
)


def test_InstanceGroupConfig():
    instance_group_config = g_instance_group_config

    assert instance_group_config.to_request() == {
        "AutoScalingPolicy": {
            "Constraints": {"MaxCapacity": 99, "MinCapacity": 1},
            "Rules": [
                {
                    "Action": {
                        "Market": "SPOT",
                        "SimpleScalingPolicyConfiguration": {
                            "AdjustmentType": "CHANGE_IN_CAPACITY",
                            "CoolDown": 9,
                            "ScalingAdjustment": 2,
                        },
                    },
                    "Description": "Description",
                    "Name": "Name",
                    "Trigger": {
                        "CloudWatchAlarmDefinition": {
                            "ComparisonOperator": "LESS_THAN",
                            "Dimensions": [{"Key": "key1", "Value": "val1"}],
                            "EvaluationPeriods": 1,
                            "MetricName": "MetricName",
                            "Namespace": "Namespace",
                            "Period": 300,
                            "Statistic": "AVERAGE",
                            "Threshold": 2.2,
                            "Unit": "SECONDS",
                        }
                    },
                }
            ],
        },
        "BidPrice": "1234",
        "Configurations": [
            {
                "Classification": "classification",
                "Configurations": [
                    {"Classification": "classification1", "Properties": {"key1": "value1"}}
                ],
                "Properties": {"key": "value"},
            }
        ],
        "CustomAmiId": "CustomAmiId",
        "EbsConfiguration": {
            "EbsBlockDeviceConfigs": [
                {
                    "VolumeSpecification": {
                        "Iops": 1,
                        "SizeInGB": 2,
                        "Throughput": 3,
                        "VolumeType": "four",
                    },
                    "VolumesPerInstance": 123,
                }
            ],
            "EbsOptimized": False,
        },
        "InstanceCount": 1,
        "InstanceRole": "MASTER",
        "InstanceType": "InstanceType",
        "Market": "SPOT",
        "Name": "Name",
    }


g_placement_type = PlacementType(availability_zone="az1", availability_zones=["az2", "az3"])


def test_PlacementType():
    placement_type = g_placement_type
    assert placement_type.to_request() == {
        "AvailabilityZone": "az1",
        "AvailabilityZones": ["az2", "az3"],
    }


g_kerberos_attributes = KerberosAttributes(
    ad_domain_join_password="adPassword",
    ad_domain_join_user="adUser",
    cross_realm_trust_principal_password="crtpPassword",
    kdc_admin_password="kdcPassword",
    realm="EC2.INTERNAL",
)


def test_KerberosAttributes():
    kerberos_attributes = g_kerberos_attributes
    assert kerberos_attributes.to_request() == {
        "ADDomainJoinPassword": "adPassword",
        "ADDomainJoinUser": "adUser",
        "CrossRealmTrustPrincipalPassword": "crtpPassword",
        "KdcAdminPassword": "kdcPassword",
        "Realm": "EC2.INTERNAL",
    }


g_compute_limits = ComputeLimits(
    maximum_capacity_units=1,
    maximum_core_capacity_units=2,
    maximum_on_demand_capacity_units=3,
    minimum_capacity_units=4,
    unit_type="VCPU",
)


def test_ComputeLimits():
    compute_limits = g_compute_limits
    assert compute_limits.to_request() == {
        "MaximumCapacityUnits": 1,
        "MaximumCoreCapacityUnits": 2,
        "MaximumOnDemandCapacityUnits": 3,
        "MinimumCapacityUnits": 4,
        "UnitType": "VCPU",
    }


g_managed_scaling_policy = ManagedScalingPolicy(compute_limits=g_compute_limits)


def test_ManagedScalingPolicy():
    managed_scaling_policy = g_managed_scaling_policy
    assert managed_scaling_policy.to_request() == {
        "ComputeLimits": {
            "MaximumCapacityUnits": 1,
            "MaximumCoreCapacityUnits": 2,
            "MaximumOnDemandCapacityUnits": 3,
            "MinimumCapacityUnits": 4,
            "UnitType": "VCPU",
        }
    }


g_supported_product_config = SupportedProductConfig(args=["arg1", "arg2"], name="productName")


def test_SupportedProductConfig():
    supported_product_config = g_supported_product_config
    assert supported_product_config.to_request() == {
        "Args": ["arg1", "arg2"],
        "Name": "productName",
    }


g_placement_group_config = PlacementGroupConfig(instance_role="MASTER", placement_strategy="SPREAD")


def test_PlacementGroupConfig():
    placement_group_config = g_placement_group_config
    assert placement_group_config.to_request() == {
        "InstanceRole": "MASTER",
        "PlacementStrategy": "SPREAD",
    }


def test_Tag():
    tag = Tag(key="key", value="value")
    assert tag.to_request() == {"Key": "key", "Value": "value"}


g_job_flow_instances_config = JobFlowInstancesConfig(
    additional_master_security_groups=["grp1", "grp2"],
    additional_slave_security_groups=["grp3", "grp4"],
    ec2_key_name="keyName",
    ec2_subnet_id="id1",
    ec2_subnet_ids=["id2", "id3"],
    emr_managed_master_security_group="grp5",
    emr_managed_slave_security_group="grp6",
    hadoop_version="v1",
    instance_count=1,
    instance_fleets=[g_instance_fleet_config],
    instance_groups=[g_instance_group_config],
    master_instance_type="masterInstance",
    placement=g_placement_type,
    service_access_security_group="serviceAccessSecurityGroup",
    slave_instance_type="slaveInstanceType",
)


def test_JobFlowInstancesConfig():
    job_flow_instances_config = g_job_flow_instances_config

    assert job_flow_instances_config.to_request() == {
        "AdditionalMasterSecurityGroups": ["grp1", "grp2"],
        "AdditionalSlaveSecurityGroups": ["grp3", "grp4"],
        "Ec2KeyName": "keyName",
        "Ec2SubnetId": "id1",
        "Ec2SubnetIds": ["id2", "id3"],
        "EmrManagedMasterSecurityGroup": "grp5",
        "EmrManagedSlaveSecurityGroup": "grp6",
        "HadoopVersion": "v1",
        "InstanceCount": 1,
        "InstanceFleets": [
            {
                "InstanceFleetType": "fleetType",
                "InstanceTypeConfigs": [
                    {
                        "BidPrice": "one",
                        "BidPriceAsPercentageOfOnDemandPrice": 12,
                        "Configurations": [
                            {
                                "Classification": "classification",
                                "Configurations": [
                                    {
                                        "Classification": "classification1",
                                        "Properties": {"key1": "value1"},
                                    }
                                ],
                                "Properties": {"key": "value"},
                            }
                        ],
                        "CustomAmiId": "someId",
                        "EbsConfiguration": {
                            "EbsBlockDeviceConfigs": [
                                {
                                    "VolumeSpecification": {
                                        "Iops": 1,
                                        "SizeInGB": 2,
                                        "Throughput": 3,
                                        "VolumeType": "four",
                                    },
                                    "VolumesPerInstance": 123,
                                }
                            ],
                            "EbsOptimized": False,
                        },
                        "InstanceType": "someInstance",
                        "WeightedCapacity": 9,
                    }
                ],
                "LaunchSpecifications": {
                    "OnDemandSpecification": {
                        "AllocationStrategy": "lowest-price",
                        "CapacityReservationOptions": {
                            "CapacityReservationPreference": "open",
                            "CapacityReservationResourceGroupArn": "groupArn",
                            "UsageStrategy": "lowest-price",
                        },
                    },
                    "SpotSpecification": {
                        "AllocationStrategy": "capacity-optimized",
                        "BlockDurationMinutes": 13,
                        "TimeoutAction": "TERMINATE_CLUSTER",
                        "TimeoutDurationMinutes": 88,
                    },
                },
                "Name": "someName",
                "TargetOnDemandCapacity": 3,
                "TargetSpotCapacity": 4,
            }
        ],
        "InstanceGroups": [
            {
                "AutoScalingPolicy": {
                    "Constraints": {"MaxCapacity": 99, "MinCapacity": 1},
                    "Rules": [
                        {
                            "Action": {
                                "Market": "SPOT",
                                "SimpleScalingPolicyConfiguration": {
                                    "AdjustmentType": "CHANGE_IN_CAPACITY",
                                    "CoolDown": 9,
                                    "ScalingAdjustment": 2,
                                },
                            },
                            "Description": "Description",
                            "Name": "Name",
                            "Trigger": {
                                "CloudWatchAlarmDefinition": {
                                    "ComparisonOperator": "LESS_THAN",
                                    "Dimensions": [{"Key": "key1", "Value": "val1"}],
                                    "EvaluationPeriods": 1,
                                    "MetricName": "MetricName",
                                    "Namespace": "Namespace",
                                    "Period": 300,
                                    "Statistic": "AVERAGE",
                                    "Threshold": 2.2,
                                    "Unit": "SECONDS",
                                }
                            },
                        }
                    ],
                },
                "BidPrice": "1234",
                "Configurations": [
                    {
                        "Classification": "classification",
                        "Configurations": [
                            {"Classification": "classification1", "Properties": {"key1": "value1"}}
                        ],
                        "Properties": {"key": "value"},
                    }
                ],
                "CustomAmiId": "CustomAmiId",
                "EbsConfiguration": {
                    "EbsBlockDeviceConfigs": [
                        {
                            "VolumeSpecification": {
                                "Iops": 1,
                                "SizeInGB": 2,
                                "Throughput": 3,
                                "VolumeType": "four",
                            },
                            "VolumesPerInstance": 123,
                        }
                    ],
                    "EbsOptimized": False,
                },
                "InstanceCount": 1,
                "InstanceRole": "MASTER",
                "InstanceType": "InstanceType",
                "Market": "SPOT",
                "Name": "Name",
            }
        ],
        "MasterInstanceType": "masterInstance",
        "Placement": {"AvailabilityZone": "az1", "AvailabilityZones": ["az2", "az3"]},
        "ServiceAccessSecurityGroup": "serviceAccessSecurityGroup",
        "SlaveInstanceType": "slaveInstanceType",
    }


def test_ClusterConfig():
    cluster_config = ClusterConfig(
        additional_info="additionalInfo",
        ami_version="amiVersion",
        applications=[g_application],
        auto_scaling_role="autoScalingRole",
        bootstrap_actions=[g_bootstrap_action_config],
        configurations=[g_configuration],
        custom_ami_id="customAmiId",
        ebs_root_volume_size=1,
        instances=g_job_flow_instances_config,
        job_flow_role="jobFLowRole",
        kerberos_attributes=g_kerberos_attributes,
        log_encryption_kms_key_id="logEncryptionKmsKeyId",
        log_uri="logUri",
        managed_scaling_policy=g_managed_scaling_policy,
        new_supported_products=[g_supported_product_config],
        os_release_label="osReleaseLabel",
        placement_group_configs=[g_placement_group_config],
        release_label="releaseLabel",
        repo_upgrade_on_boot="repoUpgradeOnBoot",
        scale_down_behavior="scaleDownBehavior",
        security_configuration="securityConfiguration",
        service_role="serviceRole",
        step_concurrency_level=9,
        supported_products=["product1", "product2"],
        tags=[Tag(key="key", value="value")],
        visible_to_all_users=False,
    )
    assert cluster_config.to_request() == {
        "AdditionalInfo": "additionalInfo",
        "AmiVersion": "amiVersion",
        "Applications": [
            {
                "AdditionalInfo": {"key1": "value1", "key2": "value2"},
                "Args": ["arg1", "arg2"],
                "Name": "name",
                "Version": "version",
            }
        ],
        "AutoScalingRole": "autoScalingRole",
        "BootstrapActions": [
            {"Name": "name", "ScriptBootstrapAction": {"Args": ["arg1", "arg2"], "Path": "path"}}
        ],
        "Configurations": [
            {
                "Classification": "classification",
                "Configurations": [
                    {"Classification": "classification1", "Properties": {"key1": "value1"}}
                ],
                "Properties": {"key": "value"},
            }
        ],
        "CustomAmiId": "customAmiId",
        "EbsRootVolumeSize": 1,
        "Instances": {
            "AdditionalMasterSecurityGroups": ["grp1", "grp2"],
            "AdditionalSlaveSecurityGroups": ["grp3", "grp4"],
            "Ec2KeyName": "keyName",
            "Ec2SubnetId": "id1",
            "Ec2SubnetIds": ["id2", "id3"],
            "EmrManagedMasterSecurityGroup": "grp5",
            "EmrManagedSlaveSecurityGroup": "grp6",
            "HadoopVersion": "v1",
            "InstanceCount": 1,
            "InstanceFleets": [
                {
                    "InstanceFleetType": "fleetType",
                    "InstanceTypeConfigs": [
                        {
                            "BidPrice": "one",
                            "BidPriceAsPercentageOfOnDemandPrice": 12,
                            "Configurations": [
                                {
                                    "Classification": "classification",
                                    "Configurations": [
                                        {
                                            "Classification": "classification1",
                                            "Properties": {"key1": "value1"},
                                        }
                                    ],
                                    "Properties": {"key": "value"},
                                }
                            ],
                            "CustomAmiId": "someId",
                            "EbsConfiguration": {
                                "EbsBlockDeviceConfigs": [
                                    {
                                        "VolumeSpecification": {
                                            "Iops": 1,
                                            "SizeInGB": 2,
                                            "Throughput": 3,
                                            "VolumeType": "four",
                                        },
                                        "VolumesPerInstance": 123,
                                    }
                                ],
                                "EbsOptimized": False,
                            },
                            "InstanceType": "someInstance",
                            "WeightedCapacity": 9,
                        }
                    ],
                    "LaunchSpecifications": {
                        "OnDemandSpecification": {
                            "AllocationStrategy": "lowest-price",
                            "CapacityReservationOptions": {
                                "CapacityReservationPreference": "open",
                                "CapacityReservationResourceGroupArn": "groupArn",
                                "UsageStrategy": "lowest-price",
                            },
                        },
                        "SpotSpecification": {
                            "AllocationStrategy": "capacity-optimized",
                            "BlockDurationMinutes": 13,
                            "TimeoutAction": "TERMINATE_CLUSTER",
                            "TimeoutDurationMinutes": 88,
                        },
                    },
                    "Name": "someName",
                    "TargetOnDemandCapacity": 3,
                    "TargetSpotCapacity": 4,
                }
            ],
            "InstanceGroups": [
                {
                    "AutoScalingPolicy": {
                        "Constraints": {"MaxCapacity": 99, "MinCapacity": 1},
                        "Rules": [
                            {
                                "Action": {
                                    "Market": "SPOT",
                                    "SimpleScalingPolicyConfiguration": {
                                        "AdjustmentType": "CHANGE_IN_CAPACITY",
                                        "CoolDown": 9,
                                        "ScalingAdjustment": 2,
                                    },
                                },
                                "Description": "Description",
                                "Name": "Name",
                                "Trigger": {
                                    "CloudWatchAlarmDefinition": {
                                        "ComparisonOperator": "LESS_THAN",
                                        "Dimensions": [{"Key": "key1", "Value": "val1"}],
                                        "EvaluationPeriods": 1,
                                        "MetricName": "MetricName",
                                        "Namespace": "Namespace",
                                        "Period": 300,
                                        "Statistic": "AVERAGE",
                                        "Threshold": 2.2,
                                        "Unit": "SECONDS",
                                    }
                                },
                            }
                        ],
                    },
                    "BidPrice": "1234",
                    "Configurations": [
                        {
                            "Classification": "classification",
                            "Configurations": [
                                {
                                    "Classification": "classification1",
                                    "Properties": {"key1": "value1"},
                                }
                            ],
                            "Properties": {"key": "value"},
                        }
                    ],
                    "CustomAmiId": "CustomAmiId",
                    "EbsConfiguration": {
                        "EbsBlockDeviceConfigs": [
                            {
                                "VolumeSpecification": {
                                    "Iops": 1,
                                    "SizeInGB": 2,
                                    "Throughput": 3,
                                    "VolumeType": "four",
                                },
                                "VolumesPerInstance": 123,
                            }
                        ],
                        "EbsOptimized": False,
                    },
                    "InstanceCount": 1,
                    "InstanceRole": "MASTER",
                    "InstanceType": "InstanceType",
                    "Market": "SPOT",
                    "Name": "Name",
                }
            ],
            "MasterInstanceType": "masterInstance",
            "Placement": {"AvailabilityZone": "az1", "AvailabilityZones": ["az2", "az3"]},
            "ServiceAccessSecurityGroup": "serviceAccessSecurityGroup",
            "SlaveInstanceType": "slaveInstanceType",
        },
        "JobFlowRole": "jobFLowRole",
        "KerberosAttributes": {
            "ADDomainJoinPassword": "adPassword",
            "ADDomainJoinUser": "adUser",
            "CrossRealmTrustPrincipalPassword": "crtpPassword",
            "KdcAdminPassword": "kdcPassword",
            "Realm": "EC2.INTERNAL",
        },
        "LogEncryptionKmsKeyId": "logEncryptionKmsKeyId",
        "LogUri": "logUri",
        "ManagedScalingPolicy": {
            "ComputeLimits": {
                "MaximumCapacityUnits": 1,
                "MaximumCoreCapacityUnits": 2,
                "MaximumOnDemandCapacityUnits": 3,
                "MinimumCapacityUnits": 4,
                "UnitType": "VCPU",
            }
        },
        "NewSupportedProducts": [{"Args": ["arg1", "arg2"], "Name": "productName"}],
        "OSReleaseLabel": "osReleaseLabel",
        "PlacementGroupConfigs": [{"InstanceRole": "MASTER", "PlacementStrategy": "SPREAD"}],
        "ReleaseLabel": "releaseLabel",
        "RepoUpgradeOnBoot": "repoUpgradeOnBoot",
        "ScaleDownBehavior": "scaleDownBehavior",
        "SecurityConfiguration": "securityConfiguration",
        "ServiceRole": "serviceRole",
        "StepConcurrencyLevel": 9,
        "SupportedProducts": ["product1", "product2"],
        "Tags": [{"Key": "key", "Value": "value"}],
        "VisibleToAllUsers": False,
    }
