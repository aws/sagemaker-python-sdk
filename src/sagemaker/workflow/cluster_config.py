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
"""Models to generate ClusterConfig."""
from __future__ import absolute_import

from typing import Any, Dict, List, Mapping, Sequence
from sagemaker.workflow.entities import RequestType


class Application:
    """https://docs.aws.amazon.com/emr/latest/APIReference/API_Application.html"""

    def __init__(
        self,
        additional_info: Dict[str, str] = None,
        args: List[str] = None,
        name: str = None,
        version: str = None,
    ):
        """Initiates Application model.

        Attributes:
            additional_info: This option is for advanced users only. This is meta information
                about third-party applications that third-party vendors use for testing purposes.
            args: Arguments for Amazon EMR to pass to the application.
            name: The name of the application.
            version: The version of the application.
        """

        self.additional_info = additional_info
        self.args = args
        self.name = name
        self.version = version

    def to_request(self) -> RequestType:
        """Convert Application object to request dict."""
        request = {}

        if self.additional_info is not None:
            request["AdditionalInfo"] = self.additional_info
        if self.args is not None:
            request["Args"] = self.args
        if self.name is not None:
            request["Name"] = self.name
        if self.version is not None:
            request["Version"] = self.version

        return request


class ScriptBootstrapActionConfig:
    """Configuration of the script to run during a bootstrap action.

    https://docs.aws.amazon.com/emr/latest/APIReference/API_ScriptBootstrapActionConfig.html
    """

    def __init__(self, args: List[str], path: str):
        """Initiates ScriptBootstrapActionConfig model.

        Attributes:
            args: A list of command line arguments to pass to the bootstrap action script.
            path: Location in Amazon S3 of the script to run during a bootstrap action.
        """
        self.args = args
        self.path = path

    def build_args(self) -> RequestType:
        """Get the request structure for list of Args."""
        arg_list = []
        for arg in self.args:
            arg_list.append(arg)
        return arg_list

    def to_request(self) -> RequestType:
        """Convert ScriptBootstrapActionConfig object to request dict."""
        request = {}

        if self.args is not None:
            request["Args"] = self.build_args()
        if self.path is not None:
            request["Path"] = self.path

        return request


class BootstrapActionConfig:
    """Configuration of a bootstrap action.

    https://docs.aws.amazon.com/emr/latest/APIReference/API_BootstrapActionConfig.html
    """

    def __init__(
        self,
        name: str,  # required
        script_bootstrap_action: ScriptBootstrapActionConfig,  # required
    ):
        """Initiates BootstrapActionConfig model.

        Attributes:
            name (required): The name of the bootstrap action.
            script_bootstrap_action (required): The script run by the bootstrap action.
        """
        self.name = name
        self.script_bootstrap_action = script_bootstrap_action

    def to_request(self) -> RequestType:
        """Convert BootstrapActionConfig object to request dict."""
        # both name and script_bootstrap_action are required attribute,
        # hence omitting the 'Nope' checking
        request = {
            "Name": self.name,
            "ScriptBootstrapAction": self.script_bootstrap_action.to_request(),
        }

        return request


class Configuration:
    """An optional configuration specification to be used when provisioning cluster instances.

    This can include configurations for applications and software bundled with Amazon EMR.
    https://docs.aws.amazon.com/emr/latest/APIReference/API_Configuration.html
    """

    def __init__(
        self,
        classification: str = None,
        configurations: Sequence[Any] = None,
        properties: Mapping[str, str] = None,
    ):
        """Initiates Configuration model.

        Attributes:
            classification: The classification within a configuration.
            configurations: A list of additional configurations to apply within a configuration
                object.
            properties: A set of properties specified within a configuration classification.
        """
        self.classification = classification
        self.configurations = configurations
        self.properties = properties

    def to_request(self) -> RequestType:
        """Convert Configuration object to request dict."""
        request = {}

        if self.classification is not None:
            request["Classification"] = self.classification
        if self.configurations is not None:
            request["Configurations"] = self.build_configurations()
        if self.properties is not None:
            request["Properties"] = self.properties

        return request

    def build_configurations(self) -> RequestType:
        """Get the request structure for list of Configuration."""
        configuration_list = []
        for configuration in self.configurations:
            configuration_list.append(configuration.to_request())
        return configuration_list


class VolumeSpecification:
    """EBS volume specifications.

    Such as volume type, IOPS, size (GiB) and throughput (MiB/s) that are
    requested for the EBS volume attached to an EC2 instance in the cluster.
    https://docs.aws.amazon.com/emr/latest/APIReference/API_VolumeSpecification.html
    """

    def __init__(self, size_in_gb: int, volume_type: str, iops: int = None, throughput: int = None):
        """Initiates VolumeSpecification model.

        Attributes:
            size_in_gb (required):  The volume size, in gibibytes (GiB). This can be a number from
                1 - 1024.
            volume_type (required): The volume type. Volume types supported are gp3, gp2, io1, st1,
                sc1, and standard.
            iops: The number of I/O operations per second (IOPS) that the volume supports.
            throughput: The throughput, in mebibyte per second (MiB/s).
        """
        self.iops = iops
        self.size_in_gb = size_in_gb
        self.throughput = throughput
        self.volume_type = volume_type

    def to_request(self) -> RequestType:
        """Convert VolumeSpecification object to request dict."""
        request = {}

        if self.iops is not None:
            request["Iops"] = self.iops
        request["SizeInGB"] = self.size_in_gb
        if self.throughput is not None:
            request["Throughput"] = self.throughput
        request["VolumeType"] = self.volume_type

        return request


class EbsBlockDeviceConfig:
    """Configuration of requested EBS block device associated with the instance group.

    https://docs.aws.amazon.com/emr/latest/APIReference/API_EbsBlockDeviceConfig.html
    """

    def __init__(self, volume_specification: VolumeSpecification, volumes_per_instance: int = None):
        """Initiates EbsBlockDeviceConfig model.

        Attributes:
            volume_specification (required): EBS volume specifications such as volume type, IOPS,
                size (GiB) and throughput (MiB/s) that are requested for the EBS volume attached to
                an EC2 instance in the cluster.
            volumes_per_instance: Number of EBS volumes with a specific volume configuration that
                are associated with every instance in the instance group
        """
        self.volume_specification = volume_specification
        self.volumes_per_instance = volumes_per_instance

    def to_request(self) -> RequestType:
        """Convert EbsBlockDeviceConfig object to request dict."""
        request = {"VolumeSpecification": self.volume_specification.to_request()}

        if self.volumes_per_instance is not None:
            request["VolumesPerInstance"] = self.volumes_per_instance

        return request


class EbsConfiguration:
    """The Amazon EBS configuration of a cluster instance.

    https://docs.aws.amazon.com/emr/latest/APIReference/API_EbsConfiguration.html
    """

    def __init__(
        self,
        ebs_block_device_configs: Sequence[EbsBlockDeviceConfig] = None,
        ebs_optimized: bool = None,
    ):
        """Initiates EbsConfiguration model.

        Attributes:
            ebs_block_device_configs: An array of Amazon EBS volume specifications attached to a
                cluster instance.
            ebs_optimized: Indicates whether an Amazon EBS volume is EBS-optimized.
        """
        self.ebs_block_device_configs = ebs_block_device_configs
        self.ebs_optimized = ebs_optimized

    def build_ebs_block_device_configs(self) -> RequestType:
        """Get the request structure for list of EbsBlockDeviceConfig."""
        ebs_block_device_config_list = []
        for ebs_block_device_config in self.ebs_block_device_configs:
            ebs_block_device_config_list.append(ebs_block_device_config.to_request())
        return ebs_block_device_config_list

    def to_request(self) -> RequestType:
        """Convert EbsConfiguration object to request dict."""
        request = {}

        if self.ebs_block_device_configs is not None:
            request["EbsBlockDeviceConfigs"] = self.build_ebs_block_device_configs()
        if self.ebs_optimized is not None:
            request["EbsOptimized"] = self.ebs_optimized

        return request


class InstanceTypeConfig:
    """An instance type configuration for each instance type in an instance fleet.

    This determines the EC2 instances Amazon EMR attempts to provision to fulfill On-Demand and
    Spot target capacities.
    https://docs.aws.amazon.com/emr/latest/APIReference/API_InstanceTypeConfig.html
    """

    def __init__(
        self,
        instance_type: str,
        bid_price: str = None,
        bid_price_as_percentage_of_on_demand_price: float = None,
        configurations: Sequence[Configuration] = None,
        custom_ami_id: str = None,
        ebs_configuration: EbsConfiguration = None,
        weighted_capacity: int = None,
    ):
        """Initiates InstanceTypeConfig model.

        Attributes:
            instance_type (required): An EC2 instance type, such as m3.xlarge.
            bid_price: The bid price for each EC2 Spot Instance type as defined by InstanceType.
                Expressed in USD.
            bid_price_as_percentage_of_on_demand_price: The bid price, as a percentage of On-Demand
                price, for each EC2 Spot Instance as defined by InstanceType. Expressed as a number
                (for example, 20 specifies 20%).
            configurations: A configuration classification that applies when provisioning cluster
                instances, which can include configurations for applications and software that run
                on the cluster.
            custom_ami_id: The custom AMI ID to use for the instance type.
            ebs_configuration:  The configuration of Amazon Elastic Block Store (Amazon EBS)
                attached to each instance as defined by InstanceType.
            weighted_capacity:  The number of units that a provisioned instance of this type
                provides toward fulfilling the target capacities defined in InstanceFleetConfig.
        """
        self.bid_price = bid_price
        self.bid_price_as_percentage_of_on_demand_price = bid_price_as_percentage_of_on_demand_price
        self.configurations = configurations
        self.custom_ami_id = custom_ami_id
        self.ebs_configuration = ebs_configuration
        self.instance_type = instance_type
        self.weighted_capacity = weighted_capacity

    def build_configurations(self) -> RequestType:
        """Get the request structure for list of Configuration."""
        configuration_list = []
        for configuration in self.configurations:
            configuration_list.append(configuration.to_request())
        return configuration_list

    def to_request(self) -> RequestType:
        """Convert InstanceTypeConfig object to request dict."""
        request = {}
        if self.bid_price is not None:
            request["BidPrice"] = self.bid_price
        if self.bid_price_as_percentage_of_on_demand_price is not None:
            request[
                "BidPriceAsPercentageOfOnDemandPrice"
            ] = self.bid_price_as_percentage_of_on_demand_price
        if self.configurations is not None:
            request["Configurations"] = self.build_configurations()
        if self.custom_ami_id is not None:
            request["CustomAmiId"] = self.custom_ami_id
        if self.ebs_configuration is not None:
            request["EbsConfiguration"] = self.ebs_configuration.to_request()
        request["InstanceType"] = self.instance_type
        if self.weighted_capacity is not None:
            request["WeightedCapacity"] = self.weighted_capacity

        return request


class OnDemandCapacityReservationOptions:
    """Describes the strategy for using unused Capacity Reservations.

    https://docs.aws.amazon.com/emr/latest/APIReference/API_OnDemandCapacityReservationOptions.html
    """

    def __init__(
        self,
        capacity_reservation_preference: str = None,
        capacity_reservation_resource_group_arn: str = None,
        usage_strategy: str = None,
    ):
        """Initiates OnDemandCapacityReservationOptions model.

        Attributes:
            capacity_reservation_preference: Indicates the instance's Capacity Reservation
                preferences.
            capacity_reservation_resource_group_arn: The ARN of the Capacity Reservation resource
                group in which to run the instance.
            usage_strategy: Indicates whether to use unused Capacity Reservations for fulfilling
                On-Demand capacity.
        """
        self.capacity_reservation_preference = capacity_reservation_preference
        self.capacity_reservation_resource_group_arn = capacity_reservation_resource_group_arn
        self.usage_strategy = usage_strategy

    def to_request(self) -> RequestType:
        """Convert OnDemandCapacityReservationOptions object to request dict."""
        request = {}

        if self.capacity_reservation_preference is not None:
            request["CapacityReservationPreference"] = self.capacity_reservation_preference
        if self.capacity_reservation_resource_group_arn is not None:
            request[
                "CapacityReservationResourceGroupArn"
            ] = self.capacity_reservation_resource_group_arn
        if self.usage_strategy is not None:
            request["UsageStrategy"] = self.usage_strategy

        return request


class OnDemandProvisioningSpecification:
    """The launch specification for On-Demand Instances in the instance fleet.

    https://docs.aws.amazon.com/emr/latest/APIReference/API_OnDemandProvisioningSpecification.html
    """

    def __init__(
        self,
        allocation_strategy: str,
        capacity_reservation_options: OnDemandCapacityReservationOptions = None,
    ):
        """Initiates OnDemandProvisioningSpecification model.

        Attributes:
            allocation_strategy (required): Specifies the strategy to use in launching On-Demand
                instance fleets.
            capacity_reservation_options: The launch specification for On-Demand instances in the
                instance fleet, which determines the allocation strategy.
        """
        self.allocation_strategy = allocation_strategy
        self.capacity_reservation_options = capacity_reservation_options

    def to_request(self) -> RequestType:
        """Convert OnDemandProvisioningSpecification object to request dict."""
        request = {"AllocationStrategy": self.allocation_strategy}

        if self.capacity_reservation_options is not None:
            request["CapacityReservationOptions"] = self.capacity_reservation_options.to_request()

        return request


class SpotProvisioningSpecification:
    """The launch specification for Spot Instances in the instance fleet.

    This determines the defined duration, provisioning timeout behavior, and allocation strategy.
    https://docs.aws.amazon.com/emr/latest/APIReference/API_SpotProvisioningSpecification.html
    """

    def __init__(
        self,
        timeout_action: str,
        timeout_duration_minutes: int,
        allocation_strategy: str = None,
        block_duration_minutes: int = None,
    ):
        """Initiates SpotProvisioningSpecification model.

        Attributes:
            timeout_action (required):  The action to take when TargetSpotCapacity has not been
                fulfilled when the TimeoutDurationMinutes has expired.
            timeout_duration_minutes (required): The spot provisioning timeout period in minutes.
            allocation_strategy: Specifies the strategy to use in launching Spot Instance fleets.
            block_duration_minutes: The defined duration for Spot Instances (also known as Spot
                blocks) in minutes.
        """
        self.timeout_action = timeout_action
        self.timeout_duration_minutes = timeout_duration_minutes
        self.allocation_strategy = allocation_strategy
        self.block_duration_minutes = block_duration_minutes

    def to_request(self) -> RequestType:
        """Convert SpotProvisioningSpecification object to request dict."""
        request = {}
        if self.allocation_strategy is not None:
            request["AllocationStrategy"] = self.allocation_strategy
        if self.block_duration_minutes is not None:
            request["BlockDurationMinutes"] = self.block_duration_minutes

        request["TimeoutAction"] = self.timeout_action
        request["TimeoutDurationMinutes"] = self.timeout_duration_minutes

        return request


class InstanceFleetProvisioningSpecifications:
    """The launch specification for Spot Instances in the fleet.

    This determines the defined duration, provisioning timeout behavior, and allocation strategy.
    https://docs.aws.amazon.com/emr/latest/APIReference/
    API_InstanceFleetProvisioningSpecifications.html
    """

    def __init__(
        self,
        on_demand_specification: OnDemandProvisioningSpecification = None,
        spot_specification: SpotProvisioningSpecification = None,
    ):
        """Initiates InstanceFleetProvisioningSpecifications model.

        Attributes:
            on_demand_specification: The launch specification for On-Demand Instances in the
                instance fleet, which determines the allocation strategy.
            spot_specification: The launch specification for Spot Instances in the fleet, which
                determines the defined duration, provisioning timeout behavior, and allocation
                strategy.
        """
        self.on_demand_specification = on_demand_specification
        self.spot_specification = spot_specification

    def to_request(self) -> RequestType:
        """Convert InstanceFleetProvisioningSpecifications object to request dict."""
        request = {}

        if self.on_demand_specification is not None:
            request["OnDemandSpecification"] = self.on_demand_specification.to_request()
        if self.spot_specification is not None:
            request["SpotSpecification"] = self.spot_specification.to_request()

        return request


class InstanceFleetConfig:
    """The configuration that defines an instance fleet.

    https://docs.aws.amazon.com/emr/latest/APIReference/API_InstanceFleetConfig.html
    """

    def __init__(
        self,
        instance_fleet_type: str,
        instance_type_configs: Sequence[InstanceTypeConfig] = None,
        launch_specifications: InstanceFleetProvisioningSpecifications = None,
        name: str = None,
        target_on_demand_capacity: int = None,
        target_spot_capacity: int = None,
    ):
        """Initiates InstanceFleetConfig model.

        Attributes:
            instance_fleet_type (required): The node type that the instance fleet hosts. Valid
                values are MASTER, CORE, and TASK.
            instance_type_configs: The instance type configurations that define the EC2 instances
                in the instance fleet.
            launch_specifications: The launch specification for the instance fleet.
            name: The friendly name of the instance fleet.
            target_on_demand_capacity: The target capacity of On-Demand units for the instance
                fleet, which determines how many On-Demand Instances to provision.
            target_spot_capacity: The target capacity of Spot units for the instance fleet, which
                determines how many Spot Instances to provision.
        """
        self.instance_fleet_type = instance_fleet_type
        self.instance_type_configs = instance_type_configs
        self.launch_specifications = launch_specifications
        self.name = name
        self.target_on_demand_capacity = target_on_demand_capacity
        self.target_spot_capacity = target_spot_capacity

    def build_instance_type_configs(self) -> RequestType:
        """Get the request structure for list of InstanceTypeConfig."""
        instance_type_config_list = []
        for instance_type_config in self.instance_type_configs:
            instance_type_config_list.append(instance_type_config.to_request())
        return instance_type_config_list

    def to_request(self) -> RequestType:
        """Convert InstanceFleetConfig object to request dict."""
        request = {"InstanceFleetType": self.instance_fleet_type}
        if self.instance_type_configs is not None:
            request["InstanceTypeConfigs"] = self.build_instance_type_configs()
        if self.launch_specifications is not None:
            request["LaunchSpecifications"] = self.launch_specifications.to_request()
        if self.name is not None:
            request["Name"] = self.name
        if self.target_on_demand_capacity is not None:
            request["TargetOnDemandCapacity"] = self.target_on_demand_capacity
        if self.target_spot_capacity is not None:
            request["TargetSpotCapacity"] = self.target_spot_capacity

        return request


class ScalingConstraints:
    """The upper and lower EC2 instance limits for an automatic scaling policy.

    https://docs.aws.amazon.com/emr/latest/APIReference/API_ScalingConstraints.html
    """

    def __init__(self, max_capacity: int, min_capacity: int):
        """Initiates ScalingConstraints model.

        Attributes:
            max_capacity (required): The upper boundary of EC2 instances in an instance group
                beyond which scaling activities are not allowed to grow.
            min_capacity (required): The lower boundary of EC2 instances in an instance group
                below which scaling activities are not allowed to shrink.
        """
        self.max_capacity = max_capacity
        self.min_capacity = min_capacity

    def to_request(self) -> RequestType:
        """Convert ScalingConstraints object to request dict."""
        request = {"MaxCapacity": self.max_capacity, "MinCapacity": self.min_capacity}
        return request


class SimpleScalingPolicyConfiguration:
    """An automatic scaling configuration.

    This describes how the policy adds or removes instances, the cooldown period, and the number of
    EC2 instances that will be added each time the CloudWatch metric alarm condition is satisfied.
    https://docs.aws.amazon.com/emr/latest/APIReference/API_SimpleScalingPolicyConfiguration.html
    """

    def __init__(self, scaling_adjustment: int, adjustment_type: str = None, cool_down: int = None):
        """Initiates SimpleScalingPolicyConfiguration model.

        Attributes:
            scaling_adjustment (required): The amount by which to scale in or scale out, based on
                the specified AdjustmentType.
            adjustment_type: The way in which EC2 instances are added (if ScalingAdjustment is a
                positive number) or terminated (if ScalingAdjustment is a negative number) each
                time the scaling activity is triggered.
            cool_down: The amount of time, in seconds, after a scaling activity completes before
                any further trigger-related scaling activities can start. The default value is 0.
        """
        self.adjustment_type = adjustment_type
        self.cool_down = cool_down
        self.scaling_adjustment = scaling_adjustment

    def to_request(self) -> RequestType:
        """Convert SimpleScalingPolicyConfiguration object to request dict."""

        request = {}
        if self.adjustment_type is not None:
            request["AdjustmentType"] = self.adjustment_type
        if self.cool_down is not None:
            request["CoolDown"] = self.cool_down
        request["ScalingAdjustment"] = self.scaling_adjustment

        return request


class ScalingAction:
    """The type of adjustment the automatic scaling activity makes when triggered.

    https://docs.aws.amazon.com/emr/latest/APIReference/API_ScalingAction.html
    """

    def __init__(
        self,
        simple_scaling_policy_configuration: SimpleScalingPolicyConfiguration,
        market: str = None,
    ):
        """Initiates ScalingAction model.

        Attributes:
            simple_scaling_policy_configuration (required): The type of adjustment the automatic
                scaling activity makes when triggered, and the periodicity of the adjustment.
            market: Not available for instance groups. Instance groups use the market type
                specified for the group.
        """
        self.market = market
        self.simple_scaling_policy_configuration = simple_scaling_policy_configuration

    def to_request(self) -> RequestType:
        """Convert ScalingAction object to request dict."""
        request = {}
        if self.market is not None:
            request["Market"] = self.market
        request[
            "SimpleScalingPolicyConfiguration"
        ] = self.simple_scaling_policy_configuration.to_request()

        return request


class MetricDimension:
    """A CloudWatch dimension, which is specified using a Key-Value pair.

    https://docs.aws.amazon.com/emr/latest/APIReference/API_MetricDimension.html
    """

    def __init__(self, key: str = None, value: str = None):
        """Initiates MetricDimension model.

        Attributes:
            key: The dimension name
            value: The dimension value
        """
        self.key = key
        self.value = value

    def to_request(self) -> RequestType:
        """Convert MetricDimension object to request dict."""
        request = {}

        if self.key is not None:
            request["Key"] = self.key
        if self.value is not None:
            request["Value"] = self.value

        return request


class CloudWatchAlarmDefinition:
    """The definition of a CloudWatch metric alarm.

    This determines when an automatic scaling activity is triggered.
    https://docs.aws.amazon.com/emr/latest/APIReference/API_CloudWatchAlarmDefinition.html
    """

    def __init__(
        self,
        comparison_operator: str,
        metric_name: str,
        period: int,
        threshold: float,
        dimensions: Sequence[MetricDimension] = None,
        evaluation_periods: int = None,
        namespace: str = None,
        statistic: str = None,
        unit: str = None,
    ):
        """Initiates CloudWatchAlarmDefinition model.

        Attributes:
            comparison_operator (required): Determines how the metric specified by MetricName is
                compared to the value specified by Threshold.
            metric_name (required): The name of the CloudWatch metric that is watched to determine
                an alarm condition.
            period (required): The period, in seconds, over which the statistic is applied.
            threshold (required): The value against which the specified statistic is compared.
            dimensions: A CloudWatch metric dimension.
            evaluation_periods: The number of periods, in five-minute increments, during which the
                alarm condition must exist before the alarm triggers automatic scaling activity.
            namespace:  The namespace for the CloudWatch metric. The default is
                        AWS/ElasticMapReduce.
            statistic:  The statistic to apply to the metric associated with the alarm. The default
                        is AVERAGE.
            unit: The unit of measure associated with the CloudWatch metric being watched.
        """
        self.comparison_operator = comparison_operator
        self.dimensions = dimensions
        self.evaluation_periods = evaluation_periods
        self.metric_name = metric_name
        self.namespace = namespace
        self.period = period
        self.statistic = statistic
        self.threshold = threshold
        self.unit = unit

    def build_dimensions(self) -> RequestType:
        """Get the request structure for list of MetricDimension."""
        dimension_list = []
        for dimension in self.dimensions:
            dimension_list.append(dimension.to_request())
        return dimension_list

    def to_request(self) -> RequestType:
        """Convert CloudWatchAlarmDefinition object to request dict."""
        request = {"ComparisonOperator": self.comparison_operator}
        if self.dimensions is not None:
            request["Dimensions"] = self.build_dimensions()
        if self.evaluation_periods is not None:
            request["EvaluationPeriods"] = self.evaluation_periods
        request["MetricName"] = self.metric_name
        if self.namespace is not None:
            request["Namespace"] = self.namespace
        request["Period"] = self.period
        if self.statistic is not None:
            request["Statistic"] = self.statistic
        request["Threshold"] = self.threshold
        if self.unit is not None:
            request["Unit"] = self.unit

        return request


class ScalingTrigger:
    """The conditions that trigger an automatic scaling activity.

    https://docs.aws.amazon.com/emr/latest/APIReference/API_ScalingTrigger.html
    """

    def __init__(self, cloud_watch_alarm_definition: CloudWatchAlarmDefinition):
        """Initiates ScalingTrigger model.

        Attributes:
            cloud_watch_alarm_definition (required): The definition of a CloudWatch metric alarm.
        """
        self.cloud_watch_alarm_definition = cloud_watch_alarm_definition

    def to_request(self) -> RequestType:
        """Convert ScalingTrigger object to request dict."""
        request = {"CloudWatchAlarmDefinition": self.cloud_watch_alarm_definition.to_request()}
        return request


class ScalingRule:
    """A scale-in or scale-out rule that defines scaling activity.

    This including the CloudWatch metric alarm that triggers activity, how EC2 instances are added
    or removed, and the periodicity of adjustments.
    https://docs.aws.amazon.com/emr/latest/APIReference/API_ScalingRule.html
    """

    def __init__(
        self, action: ScalingAction, name: str, trigger: ScalingTrigger, description: str = None
    ):
        """Initiates ScalingRule model.

        Attributes:
            action (required): The conditions that trigger an automatic scaling activity.
            name (required): The name used to identify an automatic scaling rule.
            trigger (required): The CloudWatch alarm definition that determines when
                automatic scaling activity is triggered.
            description: A friendly, more verbose description of the automatic scaling rule.
        """
        self.action = action
        self.description = description
        self.name = name
        self.trigger = trigger

    def to_request(self) -> RequestType:
        """Convert ScalingRule object to request dict."""
        request = {"Action": self.action.to_request()}
        if self.description is not None:
            request["Description"] = self.description
        request["Name"] = self.name
        request["Trigger"] = self.trigger.to_request()

        return request


class AutoScalingPolicy:
    """An automatic scaling policy for a core instance group or task instance group.

    https://docs.aws.amazon.com/emr/latest/APIReference/API_AutoScalingPolicy.html
    """

    def __init__(self, constraints: ScalingConstraints, rules: Sequence[ScalingRule]):
        """Initiates AutoScalingPolicy model.

        Attributes:
            constraints (required): The upper and lower EC2 instance limits for an automatic
                scaling policy.
            rules (required): The scale-in and scale-out rules that comprise the automatic
                scaling policy.
        """
        self.constraints = constraints
        self.rules = rules

    def build_rules(self) -> RequestType:
        """Get the request structure for list of ScalingRule."""
        rule_list = []
        for rule in self.rules:
            rule_list.append(rule.to_request())
        return rule_list

    def to_request(self) -> RequestType:
        """Convert AutoScalingPolicy object to request dict."""
        request = {"Constraints": self.constraints.to_request(), "Rules": self.build_rules()}
        return request


class InstanceGroupConfig:
    """Configuration defining a new instance group.

    https://docs.aws.amazon.com/emr/latest/APIReference/API_InstanceGroupConfig.html
    """

    def __init__(
        self,
        instance_count: int,
        instance_role: str,
        instance_type: str,
        auto_scaling_policy: AutoScalingPolicy = None,
        bid_price: str = None,
        configurations: Sequence[Configuration] = None,
        custom_ami_id: str = None,
        ebs_configuration: EbsConfiguration = None,
        market: str = None,
        name: str = None,
    ):
        """Initiates InstanceGroupConfig model.

        Attributes:
            instance_count (required): Target number of instances for the instance group.
            instance_role (required): The role of the instance group in the cluster.
            instance_type (required): The EC2 instance type for all instances in the instance group.
            auto_scaling_policy: An automatic scaling policy for a core instance group or task
                instance group in an Amazon EMR cluster.
            bid_price: If specified, indicates that the instance group uses Spot Instances.
            configurations: Amazon EMR releases 4.x or later. The list of configurations supplied
                            for an EMR cluster instance group.
            custom_ami_id: The custom AMI ID to use for the provisioned instance group.
            ebs_configuration: EBS configurations that will be attached to each EC2 instance in the
                instance group.
            market: Market type of the EC2 instances used to create a cluster node.
            name: Friendly name given to the instance group.
        """
        self.auto_scaling_policy = auto_scaling_policy
        self.bid_price = bid_price
        self.configurations = configurations
        self.custom_ami_id = custom_ami_id
        self.ebs_configuration = ebs_configuration
        self.instance_count = instance_count
        self.instance_role = instance_role
        self.instance_type = instance_type
        self.market = market
        self.name = name

    def build_configurations(self) -> RequestType:
        """Get the request structure for list of Configuration."""
        configuration_list = []
        for configuration in self.configurations:
            configuration_list.append(configuration.to_request())
        return configuration_list

    def to_request(self) -> RequestType:
        """Convert InstanceGroupConfig object to request dict."""
        request = {}

        if self.auto_scaling_policy is not None:
            request["AutoScalingPolicy"] = self.auto_scaling_policy.to_request()
        if self.bid_price is not None:
            request["BidPrice"] = self.bid_price
        if self.configurations is not None:
            request["Configurations"] = self.build_configurations()
        if self.custom_ami_id is not None:
            request["CustomAmiId"] = self.custom_ami_id
        if self.ebs_configuration is not None:
            request["EbsConfiguration"] = self.ebs_configuration.to_request()
        request["InstanceCount"] = self.instance_count
        request["InstanceRole"] = self.instance_role
        request["InstanceType"] = self.instance_type
        if self.market is not None:
            request["Market"] = self.market
        if self.name is not None:
            request["Name"] = self.name

        return request


class PlacementType:
    """The Amazon EC2 Availability Zone configuration of the cluster (job flow).

    https://docs.aws.amazon.com/emr/latest/APIReference/API_PlacementType.html
    """

    def __init__(self, availability_zone: str = None, availability_zones: Sequence[str] = None):
        """Initiates PlacementType model.

        Arguments:
            availability_zone: The Amazon EC2 Availability Zone for the cluster.
            availability_zones: When multiple Availability Zones are specified, Amazon EMR
                evaluates them and launches instances in the optimal Availability Zone.
        """
        self.availability_zone = availability_zone
        self.availability_zones = availability_zones

    def to_request(self) -> RequestType:
        """Convert PlacementType object to request dict."""
        request = {}

        if self.availability_zone is not None:
            request["AvailabilityZone"] = self.availability_zone
            request["AvailabilityZones"] = self.availability_zones

        return request


class JobFlowInstancesConfig:
    """A description of the Amazon EC2 instance on which the cluster (job flow) runs.

    A valid JobFlowInstancesConfig must contain either InstanceGroups or InstanceFleets.
    They cannot be used together.
    https://docs.aws.amazon.com/emr/latest/APIReference/API_JobFlowInstancesConfig.html
    """

    def __init__(
        self,
        additional_master_security_groups: Sequence[str] = None,
        additional_slave_security_groups: Sequence[str] = None,
        ec2_key_name: str = None,
        ec2_subnet_id: str = None,
        ec2_subnet_ids: Sequence[str] = None,
        emr_managed_master_security_group: str = None,
        emr_managed_slave_security_group: str = None,
        hadoop_version: str = None,
        instance_count: int = None,
        instance_fleets: Sequence[InstanceFleetConfig] = None,
        instance_groups: Sequence[InstanceGroupConfig] = None,
        master_instance_type: str = None,
        placement: PlacementType = None,
        service_access_security_group: str = None,
        slave_instance_type: str = None,
    ):
        """Initiates JobFlowInstancesConfig model.

        Attributes:
            additional_master_security_groups: A list of additional Amazon EC2 security group IDs
                for the master node.
            additional_slave_security_groups: A list of additional Amazon EC2 security group IDs
                for the core and task nodes.
            ec2_key_name: The name of the EC2 key pair that can be used to connect to the master
                node using SSH as the user called "hadoop."
            ec2_subnet_id:  Applies to clusters that use the uniform instance group configuration.
            ec2_subnet_ids:  Applies to clusters that use the instance fleet configuration.
            emr_managed_master_security_group:  The identifier of the Amazon EC2 security group for
                the master node. If you specify EmrManagedMasterSecurityGroup, you must also
                specify EmrManagedSlaveSecurityGroup.
            emr_managed_slave_security_group: The identifier of the Amazon EC2 security group for
                the core and task nodes. If you specify EmrManagedSlaveSecurityGroup, you must
                also specify EmrManagedMasterSecurityGroup.
            hadoop_version: Applies only to Amazon EMR release versions earlier than 4.0.
                The Hadoop version for the cluster.
            instance_count: The number of EC2 instances in the cluster.
            instance_fleets: Describes the EC2 instances and instance configurations for clusters
                that use the instance fleet configuration. The instance fleet configuration is
                available only in Amazon EMR versions 4.8.0 and later, excluding 5.0.x versions.
            instance_groups: Configuration for the instance groups in a cluster.
            master_instance_type: The EC2 instance type of the master node.
            placement: The Availability Zone in which the cluster runs.
            service_access_security_group: The identifier of the Amazon EC2 security group for the
                Amazon EMR service to access clusters in VPC private subnets.
            slave_instance_type: The EC2 instance type of the core and task nodes.
        """
        self.additional_master_security_groups = additional_master_security_groups
        self.additional_slave_security_groups = additional_slave_security_groups
        self.ec2_key_name = ec2_key_name
        self.ec2_subnet_id = ec2_subnet_id
        self.ec2_subnet_ids = ec2_subnet_ids
        self.emr_managed_master_security_group = emr_managed_master_security_group
        self.emr_managed_slave_security_group = emr_managed_slave_security_group
        self.hadoop_version = hadoop_version
        self.instance_count = instance_count
        self.instance_fleets = instance_fleets
        self.instance_groups = instance_groups
        self.master_instance_type = master_instance_type
        self.placement = placement
        self.service_access_security_group = service_access_security_group
        self.slave_instance_type = slave_instance_type

    def build_instance_fleets(self) -> RequestType:
        """Get the request structure for list of InstanceFleetConfig."""
        instance_fleet_list = []
        for instance_fleet in self.instance_fleets:
            instance_fleet_list.append(instance_fleet.to_request())
        return instance_fleet_list

    def build_instance_groups(self) -> RequestType:
        """Get the request structure for list of InstanceGroupConfig."""
        instance_group_list = []
        for instance_group in self.instance_groups:
            instance_group_list.append(instance_group.to_request())
        return instance_group_list

    def to_request(self) -> RequestType:
        """Convert JobFlowInstancesConfig object to request dict."""
        request = {}
        if self.additional_master_security_groups is not None:
            request["AdditionalMasterSecurityGroups"] = self.additional_master_security_groups
        if self.additional_slave_security_groups is not None:
            request["AdditionalSlaveSecurityGroups"] = self.additional_slave_security_groups
        if self.ec2_key_name is not None:
            request["Ec2KeyName"] = self.ec2_key_name
        if self.ec2_subnet_id is not None:
            request["Ec2SubnetId"] = self.ec2_subnet_id
        if self.ec2_subnet_ids is not None:
            request["Ec2SubnetIds"] = self.ec2_subnet_ids
        if self.emr_managed_master_security_group is not None:
            request["EmrManagedMasterSecurityGroup"] = self.emr_managed_master_security_group
        if self.emr_managed_slave_security_group is not None:
            request["EmrManagedSlaveSecurityGroup"] = self.emr_managed_slave_security_group
        if self.hadoop_version is not None:
            request["HadoopVersion"] = self.hadoop_version
        if self.instance_count is not None:
            request["InstanceCount"] = self.instance_count
        if self.instance_fleets is not None:
            request["InstanceFleets"] = self.build_instance_fleets()
        if self.instance_groups is not None:
            request["InstanceGroups"] = self.build_instance_groups()
        if self.master_instance_type is not None:
            request["MasterInstanceType"] = self.master_instance_type
        if self.placement is not None:
            request["Placement"] = self.placement.to_request()
        if self.service_access_security_group is not None:
            request["ServiceAccessSecurityGroup"] = self.service_access_security_group
        if self.slave_instance_type is not None:
            request["SlaveInstanceType"] = self.slave_instance_type

        return request


class KerberosAttributes:
    """Attributes for Kerberos configuration.

    It is used only when Kerberos authentication is enabled using a security configuration.
    https://docs.aws.amazon.com/emr/latest/APIReference/API_KerberosAttributes.html
    """

    def __init__(
        self,
        kdc_admin_password: str,
        realm: str,
        ad_domain_join_password: str = None,
        ad_domain_join_user: str = None,
        cross_realm_trust_principal_password: str = None,
    ):
        """Initiates KerberosAttributes model.

        Attributes:
            kdc_admin_password (required):  The password used within the cluster for the kadmin
                service on the cluster-dedicated KDC, which maintains Kerberos principals, password
                policies, and keytabs for the cluster.
            realm (required): The name of the Kerberos realm to which all nodes in a cluster
                belong. For example, EC2.INTERNAL.
            ad_domain_join_password: The Active Directory password for ADDomainJoinUser.
            ad_domain_join_user: Required only when establishing a cross-realm trust with an
                Active Directory domain. A user with sufficient privileges to join resources to
                the domain.
            cross_realm_trust_principal_password: Required only when establishing a cross-realm
                trust with a KDC in a different realm. The cross-realm principal password, which
                must be identical across realms.
        """
        self.ad_domain_join_password = ad_domain_join_password
        self.ad_domain_join_user = ad_domain_join_user
        self.cross_realm_trust_principal_password = cross_realm_trust_principal_password
        self.kdc_admin_password = kdc_admin_password
        self.realm = realm

    def to_request(self) -> RequestType:
        """Convert KerberosAttributes object to request dict."""
        request = {}
        if self.ad_domain_join_password is not None:
            request["ADDomainJoinPassword"] = self.ad_domain_join_password
        if self.ad_domain_join_user is not None:
            request["ADDomainJoinUser"] = self.ad_domain_join_user
        if self.cross_realm_trust_principal_password is not None:
            request["CrossRealmTrustPrincipalPassword"] = self.cross_realm_trust_principal_password
        request["KdcAdminPassword"] = self.kdc_admin_password
        request["Realm"] = self.realm

        return request


class ComputeLimits:
    """The EC2 unit limits for a managed scaling policy.

    https://docs.aws.amazon.com/emr/latest/APIReference/API_ComputeLimits.html
    """

    def __init__(
        self,
        maximum_capacity_units: int,
        minimum_capacity_units: int,
        unit_type: str,
        maximum_core_capacity_units: int = None,
        maximum_on_demand_capacity_units: int = None,
    ):
        """Initiates ComputeLimits model.

        Attributes:
            maximum_capacity_units (required): The upper boundary of EC2 units.
            minimum_capacity_units (required): The lower boundary of EC2 units.
            unit_type (required): The unit type used for specifying a managed scaling policy.
            maximum_core_capacity_units: The upper boundary of EC2 units for core node
                type in a cluster.
            maximum_on_demand_capacity_units: The upper boundary of On-Demand EC2 units.
        """
        self.maximum_capacity_units = maximum_capacity_units
        self.maximum_core_capacity_units = maximum_core_capacity_units
        self.maximum_on_demand_capacity_units = maximum_on_demand_capacity_units
        self.minimum_capacity_units = minimum_capacity_units
        self.unit_type = unit_type

    def to_request(self) -> RequestType:
        """Convert ComputeLimits object to request dict."""
        request = {"MaximumCapacityUnits": self.maximum_capacity_units}
        if self.maximum_core_capacity_units is not None:
            request["MaximumCoreCapacityUnits"] = self.maximum_core_capacity_units
        if self.maximum_on_demand_capacity_units is not None:
            request["MaximumOnDemandCapacityUnits"] = self.maximum_on_demand_capacity_units
        request["MinimumCapacityUnits"] = self.minimum_capacity_units
        request["UnitType"] = self.unit_type

        return request


class ManagedScalingPolicy:
    """Managed scaling policy for an Amazon EMR cluster.

    https://docs.aws.amazon.com/emr/latest/APIReference/API_ManagedScalingPolicy.html
    """

    def __init__(self, compute_limits: ComputeLimits = None):
        """Initiates ManagedScalingPolicy model.

        Attributes:
            compute_limits: The EC2 unit limits for a managed scaling policy.
        """
        self.compute_limits = compute_limits

    def to_request(self) -> RequestType:
        """Convert ManagedScalingPolicy object to request dict."""
        request = {}
        if self.compute_limits is not None:
            request["ComputeLimits"] = self.compute_limits.to_request()

        return request


class SupportedProductConfig:
    """The list of supported product configurations that allow user-supplied arguments.

    https://docs.aws.amazon.com/emr/latest/APIReference/API_SupportedProductConfig.html
    """

    def __init__(self, args: Sequence[str] = None, name: str = None):
        """Initiates SupportedProductConfig model.

        Attributes:
            args: The list of user-supplied arguments.
            name: The name of the product configuration.
        """
        self.args = args
        self.name = name

    def to_request(self) -> RequestType:
        """Convert SupportedProductConfig object to request dict."""
        request = {}
        if self.args is not None:
            request["Args"] = self.args
        if self.name is not None:
            request["Name"] = self.name

        return request


class PlacementGroupConfig:
    """Placement group configuration for an Amazon EMR cluster.

    https://docs.aws.amazon.com/emr/latest/APIReference/API_PlacementGroupConfig.html
    """

    def __init__(self, instance_role: str, placement_strategy: str = None):
        """Initiates PlacementGroupConfig model.

        Arguments:
            instance_role (required): Role of the instance in the cluster.
            placement_strategy: EC2 Placement Group strategy associated with instance role.
        """
        self.instance_role = instance_role
        self.placement_strategy = placement_strategy

    def to_request(self) -> RequestType:
        """Convert PlacementGroupConfig object to request dict."""
        request = {"InstanceRole": self.instance_role}
        if self.placement_strategy is not None:
            request["PlacementStrategy"] = self.placement_strategy

        return request


class Tag:
    """A key-value pair containing user-defined metadata.

    This can be associated with an Amazon EMR resource.
    https://docs.aws.amazon.com/emr/latest/APIReference/API_Tag.html
    """

    def __init__(self, key: str = None, value: str = None):
        """Initiates Tag model.

        Attributes:
            key: A user-defined key, which is the minimum required information for a valid tag.
            value: A user-defined value, which is optional in a tag.
        """
        self.key = key
        self.value = value

    def to_request(self) -> RequestType:
        """Convert Tag object to request dict."""
        request = {}
        if self.key is not None:
            request["Key"] = self.key
        if self.value is not None:
            request["Value"] = self.value

        return request


class ClusterConfig:
    """Config for creating a new EMR cluster.

    This class is a subset of the arguments of RunJobFlowRequest
    https://docs.aws.amazon.com/emr/latest/APIReference/API_RunJobFlow.html
    The following attributes are removed in ClusterConfig:
        RunJobFlowRequest.Name
        RunJobFlowRequest.Steps
        RunJobFlowRequest.AutoTerminationPolicy
        RunJobFlowRequest.Instances.KeepJobFlowAliveWhenNoSteps
        RunJobFlowRequest.Instances.TerminationProtected
    """

    def __init__(
        self,
        instances: JobFlowInstancesConfig,
        additional_info: str = None,
        ami_version: str = None,
        applications: Sequence[Application] = None,
        auto_scaling_role: str = None,
        bootstrap_actions: Sequence[BootstrapActionConfig] = None,
        configurations: Sequence[Configuration] = None,
        custom_ami_id: str = None,
        ebs_root_volume_size: int = None,
        job_flow_role: str = None,
        kerberos_attributes: KerberosAttributes = None,
        log_encryption_kms_key_id: str = None,
        log_uri: str = None,
        managed_scaling_policy: ManagedScalingPolicy = None,
        new_supported_products: Sequence[SupportedProductConfig] = None,
        os_release_label: str = None,
        placement_group_configs: Sequence[PlacementGroupConfig] = None,
        release_label: str = None,
        repo_upgrade_on_boot: str = None,
        scale_down_behavior: str = None,
        security_configuration: str = None,
        service_role: str = None,
        step_concurrency_level: int = None,
        supported_products: Sequence[str] = None,
        tags: Sequence[Tag] = None,
        visible_to_all_users: bool = None,
    ):
        """Initiates ClusterConfig model.

        Attributes:
            instances (required): A specification of the number and type of Amazon EC2 instances.
            additional_info: A JSON string for selecting additional features.
            ami_version: Applies only to Amazon EMR AMI versions 3.x and 2.x. For Amazon EMR
                releases 4.0 and later, ReleaseLabel is used.
            applications: Applies to Amazon EMR releases 4.0 and later. A case-insensitive
                list of applications for Amazon EMR to install and configure when launching
                the cluster.
            auto_scaling_role: An IAM role for automatic scaling policies.
            bootstrap_actions: A list of bootstrap actions to run before Hadoop starts on the
                cluster nodes.
            configurations: For Amazon EMR releases 4.0 and later. The list of configurations
                supplied for the EMR cluster you are creating.
            custom_ami_id: Available only in Amazon EMR version 5.7.0 and later. The ID of a custom
                Amazon EBS-backed Linux AMI.
            ebs_root_volume_size: The size, in GiB, of the Amazon EBS root device volume of the
                Linux AMI that is used for each EC2 instance. Available in Amazon EMR version
                4.x and later.
            job_flow_role: Also called instance profile and EC2 role. An IAM role for an
                EMR cluster.
            kerberos_attributes: Attributes for Kerberos configuration when Kerberos authentication
                is enabled using a security configuration.
            log_encryption_kms_key_id: The AWS KMS key used for encrypting log files.
            log_uri: The location in Amazon S3 to write the log files of the job flow.
            managed_scaling_policy: The specified managed scaling policy for an Amazon EMR cluster.
            new_supported_products: A list of strings that indicates third-party software to use
                with the job flow that accepts a user argument list.
            os_release_label: Specifies a particular Amazon Linux release for all nodes in a
                cluster launch RunJobFlow request.
            placement_group_configs: The specified placement group configuration for an Amazon
                EMR cluster.
            release_label: The Amazon EMR release label, which determines the version of
                open-source application packages installed on the cluster.
            repo_upgrade_on_boot: Applies only when CustomAmiID is used.
            scale_down_behavior: Specifies the way that individual Amazon EC2 instances  terminate
                when an automatic scale-in activity occurs or an instance group is resized.
            security_configuration: The name of a security configuration to apply to the cluster.
            service_role: The IAM role that Amazon EMR assumes in order to access AWS resources
                on your behalf.
            step_concurrency_level: Specifies the number of steps that can be executed concurrently.
            supported_products: A list of strings that indicates third-party software to use.
            tags: A list of tags to associate with a cluster and propagate to Amazon
                EC2 instances.
            visible_to_all_users: Set this value to true so that IAM principals in the AWS
                account associated with the cluster can perform EMR actions on the cluster that
                their IAM policies allow.
        """

        self.additional_info = additional_info
        self.ami_version = ami_version
        self.applications = applications
        self.auto_scaling_role = auto_scaling_role
        self.bootstrap_actions = bootstrap_actions
        self.configurations = configurations
        self.custom_ami_id = custom_ami_id
        self.ebs_root_volume_size = ebs_root_volume_size
        self.instances = instances
        self.job_flow_role = job_flow_role
        self.kerberos_attributes = kerberos_attributes
        self.log_encryption_kms_key_id = log_encryption_kms_key_id
        self.log_uri = log_uri
        self.managed_scaling_policy = managed_scaling_policy
        self.new_supported_products = new_supported_products
        self.os_release_label = os_release_label
        self.placement_group_configs = placement_group_configs
        self.release_label = release_label
        self.repo_upgrade_on_boot = repo_upgrade_on_boot
        self.scale_down_behavior = scale_down_behavior
        self.security_configuration = security_configuration
        self.service_role = service_role
        self.step_concurrency_level = step_concurrency_level
        self.supported_products = supported_products
        self.tags = tags
        self.visible_to_all_users = visible_to_all_users

    def build_applications(self) -> RequestType:
        """Get the request structure for list of ApplicationTypeDef."""
        applications_list = []
        for application in self.applications:
            applications_list.append(application.to_request())
        return applications_list

    def build_bootstrap_actions(self) -> RequestType:
        """Get the request structure for list of ApplicationTypeDef."""
        bootstrap_action_list = []
        for bootstrap_action in self.bootstrap_actions:
            bootstrap_action_list.append(bootstrap_action.to_request())
        return bootstrap_action_list

    def build_configurations(self) -> RequestType:
        """Get the request structure for list of Configuration."""
        configuration_list = []
        for configuration in self.configurations:
            configuration_list.append(configuration.to_request())
        return configuration_list

    def build_new_supported_products(self) -> RequestType:
        """Get the request structure for list of SupportedProductConfig."""
        supported_product_config_list = []
        for supported_product_config in self.new_supported_products:
            supported_product_config_list.append(supported_product_config.to_request())
        return supported_product_config_list

    def build_placement_group_configs(self) -> RequestType:
        """Get the request structure for list of PlacementGroupConfig."""
        placement_group_config_list = []
        for placement_group_config in self.placement_group_configs:
            placement_group_config_list.append(placement_group_config.to_request())
        return placement_group_config_list

    def build_tags(self) -> RequestType:
        """Get the request structure for list of Tag."""
        tag_list = []
        for tag in self.tags:
            tag_list.append(tag.to_request())
        return tag_list

    def to_request(self) -> RequestType:
        """Convert ClusterConfig object to request dict."""
        cluster_config = {}
        if self.additional_info is not None:
            cluster_config["AdditionalInfo"] = self.additional_info
        if self.ami_version is not None:
            cluster_config["AmiVersion"] = self.ami_version
        if self.applications is not None:
            cluster_config["Applications"] = self.build_applications()
        if self.auto_scaling_role is not None:
            cluster_config["AutoScalingRole"] = self.auto_scaling_role
        if self.bootstrap_actions is not None:
            cluster_config["BootstrapActions"] = self.build_bootstrap_actions()
        if self.configurations is not None:
            cluster_config["Configurations"] = self.build_configurations()
        if self.custom_ami_id is not None:
            cluster_config["CustomAmiId"] = self.custom_ami_id
        if self.ebs_root_volume_size is not None:
            cluster_config["EbsRootVolumeSize"] = self.ebs_root_volume_size
        cluster_config["Instances"] = self.instances.to_request()
        if self.job_flow_role is not None:
            cluster_config["JobFlowRole"] = self.job_flow_role
        if self.kerberos_attributes is not None:
            cluster_config["KerberosAttributes"] = self.kerberos_attributes.to_request()
        if self.log_encryption_kms_key_id is not None:
            cluster_config["LogEncryptionKmsKeyId"] = self.log_encryption_kms_key_id
        if self.log_uri is not None:
            cluster_config["LogUri"] = self.log_uri
        if self.managed_scaling_policy is not None:
            cluster_config["ManagedScalingPolicy"] = self.managed_scaling_policy.to_request()
        if self.new_supported_products is not None:
            cluster_config["NewSupportedProducts"] = self.build_new_supported_products()
        if self.os_release_label is not None:
            cluster_config["OSReleaseLabel"] = self.os_release_label
        if self.placement_group_configs is not None:
            cluster_config["PlacementGroupConfigs"] = self.build_placement_group_configs()
        if self.release_label is not None:
            cluster_config["ReleaseLabel"] = self.release_label
        if self.repo_upgrade_on_boot is not None:
            cluster_config["RepoUpgradeOnBoot"] = self.repo_upgrade_on_boot
        if self.scale_down_behavior is not None:
            cluster_config["ScaleDownBehavior"] = self.scale_down_behavior
        if self.security_configuration is not None:
            cluster_config["SecurityConfiguration"] = self.security_configuration
        if self.service_role is not None:
            cluster_config["ServiceRole"] = self.service_role
        if self.step_concurrency_level is not None:
            cluster_config["StepConcurrencyLevel"] = self.step_concurrency_level
        if self.supported_products is not None:
            cluster_config["SupportedProducts"] = self.supported_products
        if self.tags is not None:
            cluster_config["Tags"] = self.build_tags()
        if self.visible_to_all_users is not None:
            cluster_config["VisibleToAllUsers"] = self.visible_to_all_users

        return cluster_config
