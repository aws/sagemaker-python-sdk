# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from __future__ import print_function, absolute_import

from abc import ABCMeta, abstractmethod
from collections import defaultdict
import datetime
import logging

from six import with_metaclass

from sagemaker.session import Session
from sagemaker.utils import DeferredError

try:
    import pandas as pd
except ImportError as e:
    logging.warning("pandas failed to import. Analytics features will be impaired or broken.")
    # Any subsequent attempt to use pandas will raise the ImportError
    pd = DeferredError(e)

METRICS_PERIOD_DEFAULT = 60  # seconds


class AnalyticsMetricsBase(with_metaclass(ABCMeta, object)):
    """Base class for tuning job or training job analytics classes.
    Understands common functionality like persistence and caching.
    """

    def export_csv(self, filename):
        """Persists the analytics dataframe to a file.

        Args:
            filename (str): The name of the file to save to.
        """
        self.dataframe().to_csv(filename)

    def dataframe(self, force_refresh=False):
        """A pandas dataframe with lots of interesting results about this object.
        Created by calling SageMaker List and Describe APIs and converting them into
        a convenient tabular summary.

        Args:
            force_refresh (bool): Set to True to fetch the latest data from SageMaker API.
        """
        if force_refresh:
            self.clear_cache()
        if self._dataframe is None:
            self._dataframe = self._fetch_dataframe()
        return self._dataframe

    @abstractmethod
    def _fetch_dataframe(self):
        """Sub-class must calculate the dataframe and return it.
        """

    def clear_cache(self):
        """Clear the object of all local caches of API methods, so
        that the next time any properties are accessed they will be refreshed from
        the service.
        """
        self._dataframe = None


class HyperparameterTuningJobAnalytics(AnalyticsMetricsBase):
    """Fetch results about a hyperparameter tuning job and make them accessible for analytics.
    """

    def __init__(self, hyperparameter_tuning_job_name, sagemaker_session=None):
        """Initialize a ``HyperparameterTuningJobAnalytics`` instance.

        Args:
            hyperparameter_tuning_job_name (str): name of the HyperparameterTuningJob to analyze.
            sagemaker_session (sagemaker.session.Session): Session object which manages interactions with
                Amazon SageMaker APIs and any other AWS services needed. If not specified, one is created
                using the default AWS configuration chain.
        """
        sagemaker_session = sagemaker_session or Session()
        self._sage_client = sagemaker_session.sagemaker_client
        self._tuning_job_name = hyperparameter_tuning_job_name
        self.clear_cache()

    @property
    def name(self):
        """Name of the HyperparameterTuningJob being analyzed
        """
        return self._tuning_job_name

    def __repr__(self):
        return "<sagemaker.HyperparameterTuningJobAnalytics for %s>" % self.name

    def clear_cache(self):
        """Clear the object of all local caches of API methods.
        """
        super(HyperparameterTuningJobAnalytics, self).clear_cache()
        self._tuning_job_describe_result = None
        self._training_job_summaries = None

    def _fetch_dataframe(self):
        """Return a pandas dataframe with all the training jobs, along with their
        hyperparameters, results, and metadata. This also includes a column to indicate
        if a training job was the best seen so far.
        """

        def reshape(training_summary):
            # Helper method to reshape a single training job summary into a dataframe record
            out = {}
            for k, v in training_summary["TunedHyperParameters"].items():
                # Something (bokeh?) gets confused with ints so convert to float
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    pass
                out[k] = v
            out["TrainingJobName"] = training_summary["TrainingJobName"]
            out["TrainingJobStatus"] = training_summary["TrainingJobStatus"]
            out["FinalObjectiveValue"] = training_summary.get(
                "FinalHyperParameterTuningJobObjectiveMetric", {}
            ).get("Value")

            start_time = training_summary.get("TrainingStartTime", None)
            end_time = training_summary.get("TrainingEndTime", None)
            out["TrainingStartTime"] = start_time
            out["TrainingEndTime"] = end_time
            if start_time and end_time:
                out["TrainingElapsedTimeSeconds"] = (end_time - start_time).total_seconds()
            return out

        # Run that helper over all the summaries.
        df = pd.DataFrame([reshape(tjs) for tjs in self.training_job_summaries()])
        return df

    @property
    def tuning_ranges(self):
        """A dictionary describing the ranges of all tuned hyperparameters.
        The keys are the names of the hyperparameter, and the values are the ranges.
        """
        out = {}
        for _, ranges in self.description()["HyperParameterTuningJobConfig"][
            "ParameterRanges"
        ].items():
            for param in ranges:
                out[param["Name"]] = param
        return out

    def description(self, force_refresh=False):
        """Call ``DescribeHyperParameterTuningJob`` for the hyperparameter tuning job.

        Args:
            force_refresh (bool): Set to True to fetch the latest data from SageMaker API.

        Returns:
            dict: The Amazon SageMaker response for ``DescribeHyperParameterTuningJob``.
        """
        if force_refresh:
            self.clear_cache()
        if not self._tuning_job_describe_result:
            self._tuning_job_describe_result = self._sage_client.describe_hyper_parameter_tuning_job(
                HyperParameterTuningJobName=self.name
            )
        return self._tuning_job_describe_result

    def training_job_summaries(self, force_refresh=False):
        """A (paginated) list of everything from ``ListTrainingJobsForTuningJob``.

        Args:
            force_refresh (bool): Set to True to fetch the latest data from SageMaker API.

        Returns:
            dict: The Amazon SageMaker response for ``ListTrainingJobsForTuningJob``.
        """
        if force_refresh:
            self.clear_cache()
        if self._training_job_summaries is not None:
            return self._training_job_summaries
        output = []
        next_args = {}
        for count in range(100):
            logging.debug("Calling list_training_jobs_for_hyper_parameter_tuning_job %d" % count)
            raw_result = self._sage_client.list_training_jobs_for_hyper_parameter_tuning_job(
                HyperParameterTuningJobName=self.name, MaxResults=100, **next_args
            )
            new_output = raw_result["TrainingJobSummaries"]
            output.extend(new_output)
            logging.debug(
                "Got %d more TrainingJobs. Total so far: %d" % (len(new_output), len(output))
            )
            if ("NextToken" in raw_result) and (len(new_output) > 0):
                next_args["NextToken"] = raw_result["NextToken"]
            else:
                break
        self._training_job_summaries = output
        return output


class TrainingJobAnalytics(AnalyticsMetricsBase):
    """Fetch training curve data from CloudWatch Metrics for a specific training job.
    """

    CLOUDWATCH_NAMESPACE = "/aws/sagemaker/TrainingJobs"

    def __init__(
        self,
        training_job_name,
        metric_names=None,
        sagemaker_session=None,
        start_time=None,
        end_time=None,
        period=None,
    ):
        """Initialize a ``TrainingJobAnalytics`` instance.

        Args:
            training_job_name (str): name of the TrainingJob to analyze.
            metric_names (list, optional): string names of all the metrics to collect for this training job.
                If not specified, then it will use all metric names configured for this job.
            sagemaker_session (sagemaker.session.Session): Session object which manages interactions with
                Amazon SageMaker APIs and any other AWS services needed. If not specified, one is specified
                using the default AWS configuration chain.
        """
        sagemaker_session = sagemaker_session or Session()
        self._sage_client = sagemaker_session.sagemaker_client
        self._cloudwatch = sagemaker_session.boto_session.client("cloudwatch")
        self._training_job_name = training_job_name
        self._start_time = start_time
        self._end_time = end_time
        self._period = period or METRICS_PERIOD_DEFAULT

        if metric_names:
            self._metric_names = metric_names
        else:
            self._metric_names = self._metric_names_for_training_job()
        self.clear_cache()

    @property
    def name(self):
        """Name of the TrainingJob being analyzed
        """
        return self._training_job_name

    def __repr__(self):
        return "<sagemaker.TrainingJobAnalytics for %s>" % self.name

    def clear_cache(self):
        """Clear the object of all local caches of API methods, so
        that the next time any properties are accessed they will be refreshed from
        the service.
        """
        super(TrainingJobAnalytics, self).clear_cache()
        self._data = defaultdict(list)
        self._time_interval = self._determine_timeinterval()

    def _determine_timeinterval(self):
        """Return a dictionary with two datetime objects, start_time and end_time,
        covering the interval of the training job
        """
        description = self._sage_client.describe_training_job(TrainingJobName=self.name)
        start_time = self._start_time or description[u"TrainingStartTime"]  # datetime object
        # Incrementing end time by 1 min since CloudWatch drops seconds before finding the logs.
        # This results in logs being searched in the time range in which the correct log line was not present.
        # Example - Log time - 2018-10-22 08:25:55
        #           Here calculated end time would also be 2018-10-22 08:25:55 (without 1 min addition)
        #           CW will consider end time as 2018-10-22 08:25 and will not be able to search the correct log.
        end_time = self._end_time or description.get(
            u"TrainingEndTime", datetime.datetime.utcnow()
        ) + datetime.timedelta(minutes=1)

        return {"start_time": start_time, "end_time": end_time}

    def _fetch_dataframe(self):
        for metric_name in self._metric_names:
            self._fetch_metric(metric_name)
        return pd.DataFrame(self._data)

    def _fetch_metric(self, metric_name):
        """Fetch all the values of a named metric, and add them to _data
        """
        request = {
            "Namespace": self.CLOUDWATCH_NAMESPACE,
            "MetricName": metric_name,
            "Dimensions": [{"Name": "TrainingJobName", "Value": self.name}],
            "StartTime": self._time_interval["start_time"],
            "EndTime": self._time_interval["end_time"],
            "Period": self._period,
            "Statistics": ["Average"],
        }
        raw_cwm_data = self._cloudwatch.get_metric_statistics(**request)["Datapoints"]
        if len(raw_cwm_data) == 0:
            logging.warning("Warning: No metrics called %s found" % metric_name)
            return

        # Process data: normalize to starting time, and sort.
        base_time = min(raw_cwm_data, key=lambda pt: pt["Timestamp"])["Timestamp"]
        all_xy = []
        for pt in raw_cwm_data:
            y = pt["Average"]
            x = (pt["Timestamp"] - base_time).total_seconds()
            all_xy.append([x, y])
        all_xy = sorted(all_xy, key=lambda x: x[0])

        # Store everything in _data to make a dataframe from
        for elapsed_seconds, value in all_xy:
            self._add_single_metric(elapsed_seconds, metric_name, value)

    def _add_single_metric(self, timestamp, metric_name, value):
        """Store a single metric in the _data dict which can be
        converted to a dataframe.
        """
        # note that this method is built this way to make it possible to
        # support live-refreshing charts in Bokeh at some point in the future.
        self._data["timestamp"].append(timestamp)
        self._data["metric_name"].append(metric_name)
        self._data["value"].append(value)

    def _metric_names_for_training_job(self):
        """Helper method to discover the metrics defined for a training job.
        """
        training_description = self._sage_client.describe_training_job(
            TrainingJobName=self._training_job_name
        )

        metric_definitions = training_description["AlgorithmSpecification"]["MetricDefinitions"]
        metric_names = [md["Name"] for md in metric_definitions]

        return metric_names
