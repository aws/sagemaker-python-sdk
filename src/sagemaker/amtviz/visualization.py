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
"""This module provides visualization capabilities for SageMaker hyperparameter tuning jobs.

It contains utilities to create interactive visualizations of hyperparameter tuning results
using Altair charts. The module enables users to analyze and understand the performance
of their hyperparameter optimization experiments through various visual representations
including:
- Progress of objective metrics over time
- Distribution of results
- Relationship between hyperparameters and objective values
- Training job metrics and instance utilization
- Comparative analysis across multiple tuning jobs

Main Features:
    - Visualize single or multiple hyperparameter tuning jobs
    - Display training job metrics from CloudWatch
    - Support for both completed and in-progress tuning jobs
    - Interactive filtering and highlighting of data points
    - CPU, memory, and GPU utilization visualization
    - Advanced visualization options for detailed analysis

Primary Classes and Functions:
    - visualize_tuning_job: Main function to create visualizations for tuning jobs
    - create_charts: Core chart creation functionality
    - get_job_analytics_data: Retrieves and processes tuning job data

Dependencies:
    - altair: For creating interactive visualizations
    - pandas: For data manipulation and analysis
    - boto3: For AWS service interaction
    - sagemaker: For accessing SageMaker resources
"""
from __future__ import absolute_import

from typing import Union, List, Optional, Tuple
import os
import warnings
import logging
import altair as alt
import pandas as pd
import numpy as np
import boto3
import sagemaker
from sagemaker.amtviz.job_metrics import get_cw_job_metrics

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", None)  # Don't truncate TrainingJobName


alt.data_transformers.disable_max_rows()
altair_renderer = os.getenv("ALTAIR_RENDERER", "default")
logger.info("Setting altair renderer to %s.", altair_renderer)
alt.renderers.enable(altair_renderer)


sm = boto3.client("sagemaker")


def _columnize(charts: List[alt.Chart], cols: int = 2) -> alt.VConcatChart:
    """Arrange charts in columns."""
    return alt.vconcat(*[alt.hconcat(*charts[i : i + cols]) for i in range(0, len(charts), cols)])


def visualize_tuning_job(
    tuning_jobs: Union[str, List[str], "sagemaker.tuner.HyperparameterTuner"],
    return_dfs: bool = False,
    job_metrics: Optional[List[str]] = None,
    trials_only: bool = False,
    advanced: bool = False,
) -> Union[alt.Chart, Tuple[alt.Chart, pd.DataFrame, pd.DataFrame]]:
    """Visualize SageMaker hyperparameter tuning jobs.

    Args:
        tuning_jobs: Single tuning job or list of tuning jobs (name or HyperparameterTuner object)
        return_dfs: Whether to return the underlying DataFrames
        job_metrics: List of additional job metrics to include
        trials_only: Whether to only show trials data
        advanced: Whether to show advanced visualizations

    Returns:
        If return_dfs is False, returns Altair chart
        If return_dfs is True, returns tuple of (chart, trials_df, full_df)
    """

    trials_df, tuned_parameters, objective_name, is_minimize = get_job_analytics_data(tuning_jobs)

    try:
        from IPython import get_ipython, display

        if get_ipython():
            # Running in a Jupyter Notebook
            display(trials_df.head(10))
        else:
            # Running in a non-Jupyter environment
            logger.info(trials_df.head(10).to_string())
    except ImportError:
        # Not running in a Jupyter Notebook
        logger.info(trials_df.head(10).to_string())

    full_df = _prepare_consolidated_df(trials_df) if not trials_only else pd.DataFrame()

    trials_df.columns = trials_df.columns.map(_clean_parameter_name)
    full_df.columns = full_df.columns.map(_clean_parameter_name)
    tuned_parameters = [_clean_parameter_name(tp) for tp in tuned_parameters]
    objective_name = _clean_parameter_name(objective_name)

    charts = create_charts(
        trials_df,
        tuned_parameters,
        full_df,
        objective_name,
        minimize_objective=is_minimize,
        job_metrics=job_metrics,
        advanced=advanced,
    )

    if return_dfs:
        return charts, trials_df, full_df
    return charts


def create_charts(
    trials_df: pd.DataFrame,
    tuning_parameters: List[str],
    full_df: pd.DataFrame,
    objective_name: str,
    minimize_objective: bool,
    job_metrics: Optional[List[str]] = None,
    highlight_trials: bool = True,
    color_trials: bool = False,
    advanced: bool = False,
) -> alt.Chart:
    """Create visualization charts for hyperparameter tuning results.

    Args:
        trials_df: DataFrame containing trials data
        tuning_parameters: List of hyperparameter names
        full_df: DataFrame with consolidated data
        objective_name: Name of the objective metric
        minimize_objective: Whether objective should be minimized
        job_metrics: Additional job metrics to include
        highlight_trials: Whether to highlight selected trials
        color_trials: Whether to color trials by job
        advanced: Whether to show advanced visualizations

    Returns:
        Altair chart visualization
    """

    if trials_df.empty:
        logger.info("No results available yet.")
        return pd.DataFrame()

    if job_metrics is None:
        job_metrics = []

    multiple_tuning_jobs = len(trials_df["TuningJobName"].unique()) > 1
    multiple_job_status = len(trials_df["TrainingJobStatus"].unique()) > 1

    # Rows, n>1
    # Detail Charts

    brush = alt.selection_interval(encodings=["x"], resolve="intersect", empty=True)

    job_highlight_selection = alt.selection_point(
        on="mouseover",
        nearest=False,
        empty=False,
        fields=["TrainingJobName", "TrainingStartTime"],
    )

    # create tooltip
    detail_tooltip = []
    for trp in [objective_name] + tuning_parameters:
        if trials_df[trp].dtype == np.float64:
            trp = alt.Tooltip(trp, format=".2e")
        detail_tooltip.append(trp)

    detail_tooltip.append(alt.Tooltip("TrainingStartTime:T", format="%H:%M:%S"))
    detail_tooltip.extend(["TrainingJobName", "TrainingJobStatus", "TrainingElapsedTimeSeconds"])

    # create stroke/stroke-width for tuning_jobs
    # and color for training jobs, if wanted
    # add coloring of the stroke to highlight correlated
    # data points
    jobs_props = {"shape": alt.Shape("TrainingJobStatus:N", legend=None)}

    if multiple_tuning_jobs:
        jobs_props["strokeWidth"] = alt.StrokeWidthValue(2.0)
        jobs_props["stroke"] = alt.Stroke("TuningJobName:N", legend=None)

    if color_trials:
        jobs_props["color"] = alt.Color("TrainingJobName:N")

    if highlight_trials:
        jobs_props["strokeWidth"] = alt.condition(
            job_highlight_selection,
            alt.StrokeWidthValue(2.0),
            alt.StrokeWidthValue(2.0),
        )
        jobs_props["stroke"] = alt.condition(
            job_highlight_selection,
            alt.StrokeValue("gold"),
            (
                alt.Stroke("TuningJobName:N", legend=None)
                if multiple_tuning_jobs
                else alt.StrokeValue("white")
            ),
        )

    opacity = alt.condition(brush, alt.value(1.0), alt.value(0.35))
    charts = []

    # Min and max of the objective. This is used in filtered
    # charts, so that the filtering does not make the axis
    # jump, which would make comparisons harder.
    objective_scale = alt.Scale(
        domain=(
            trials_df[objective_name].min(),
            trials_df[objective_name].max(),
        )
    )

    # If we have multiple tuning jobs, we also want to be able
    # to discriminate based on the individual tuning job, so
    # we just treat them as an additional tuning parameter
    tuning_job_param = ["TuningJobName"] if multiple_tuning_jobs else []
    tuning_parameters = tuning_parameters.copy() + tuning_job_param

    # If we use early stopping and at least some jobs were
    # stopped early, we want to be able to discriminate
    # those jobs.
    if multiple_job_status:
        tuning_parameters.append("TrainingJobStatus")

    def render_detail_charts():
        # To force a tuning job to sample a combination more than once, we
        # sometimes introduce a hyperparameter that has no effect.
        # It's values are random and without impact, so we omit it from analysis.
        ignored_parameters = {"dummy"}
        for tuning_parameter in tuning_parameters:
            if tuning_parameter in ignored_parameters:
                continue

            # Map dataframe's dtype to altair's types and
            # adjust scale if necessary
            scale_type = "linear"
            scale_log_base = 10

            few_values = len(trials_df[tuning_parameter].unique()) < 8
            parameter_type = "N"  # Nominal
            dtype = str(trials_df.dtypes[tuning_parameter])
            if "float" in dtype:
                parameter_type = "Q"  # Quantitative
                ratio = (trials_df[tuning_parameter].max() + 1e-10) / (
                    trials_df[tuning_parameter].min() + 1e-10
                )
                not_likely_discrete = (
                    len(trials_df[tuning_parameter].unique()) > trials_df[tuning_parameter].count()
                )  # edge case when both are equal
                if few_values and not_likely_discrete:
                    if ratio > 50:
                        scale_type = "log"
                    elif ratio > 10:
                        scale_type = "log"
                        scale_log_base = 2

            elif "int" in dtype or "object" in dtype:
                parameter_type = "O"  # Ordinal

            x_encoding = alt.X(
                f"{tuning_parameter}:{parameter_type}",
                scale=alt.Scale(
                    zero=False,
                    padding=1,
                    type=scale_type,
                    base=scale_log_base,
                ),
            )

            # Sync the coloring for categorical hyperparameters
            discrete = parameter_type in ["O", "N"] and few_values

            # Detail Chart
            charts.append(
                alt.Chart(trials_df)
                .add_params(brush)
                .add_params(job_highlight_selection)
                .mark_point(filled=True, size=50)
                .encode(
                    x=x_encoding,
                    y=alt.Y(
                        f"{objective_name}:Q",
                        scale=alt.Scale(zero=False, padding=1),
                        axis=alt.Axis(title=objective_name),
                    ),
                    opacity=opacity,
                    tooltip=detail_tooltip,
                    **jobs_props,
                )
            )

            if discrete:
                # Individually coloring the values only if we don't already
                # use the colors to show the different tuning jobs
                logger.info("%s, %s", parameter_type, tuning_parameter)
                if not multiple_tuning_jobs:
                    charts[-1] = charts[-1].encode(color=f"{tuning_parameter}:N")
                charts[-1] = (
                    (
                        charts[-1]
                        | alt.Chart(trials_df)
                        .transform_filter(brush)
                        .transform_density(
                            objective_name,
                            bandwidth=0.01,
                            groupby=[tuning_parameter],
                            # https://github.com/vega/altair/issues/3203#issuecomment-2141558911
                            # Specifying extent no longer necessary (>5.1.2).
                            extent=[
                                trials_df[objective_name].min(),
                                trials_df[objective_name].max(),
                            ],
                        )
                        .mark_area(opacity=0.5)
                        .encode(
                            x=alt.X(
                                "value:Q",
                                title=objective_name,
                                scale=objective_scale,
                            ),
                            y="density:Q",
                            color=alt.Color(
                                f"{tuning_parameter}:N",
                            ),
                            tooltip=tuning_parameter,
                        )
                    ).properties(title=tuning_parameter)
                    # .resolve_scale("independent")
                    # .resolve_legend(color="independent")
                )

            if advanced and parameter_type == "Q":
                # Adding tick marks to the detail charts with quantitative hyperparameters
                x_enc = x_encoding.copy()
                charts[-1].encoding.x.title = None
                charts[-1].encoding.x.axis = alt.Axis(labels=False)

                charts[-1] = charts[-1] & alt.Chart(trials_df).mark_tick(opacity=0.5).encode(
                    x=x_enc,
                    opacity=alt.condition(brush, alt.value(0.5), alt.value(0.1)),
                )

        return _columnize(charts)

    detail_charts = render_detail_charts()

    # First Row
    # Progress Over Time Chart

    def render_progress_chart():
        # Sorting trials by training start time, so that we can track the \
        # progress of the best objective so far over time
        trials_df_by_tst = trials_df.sort_values(["TuningJobName", "TrainingStartTime"])
        trials_df_by_tst["cum_objective"] = trials_df_by_tst.groupby(["TuningJobName"]).transform(
            lambda x: x.cummin() if minimize_objective else x.cummax()
        )[objective_name]

        progress_chart = (
            alt.Chart(trials_df_by_tst)
            .add_params(brush)
            .add_params(job_highlight_selection)
            .mark_point(filled=True, size=50)
            .encode(
                x=alt.X("TrainingStartTime:T", scale=alt.Scale(nice=True)),
                y=alt.Y(
                    f"{objective_name}:Q",
                    scale=alt.Scale(zero=False, padding=1),
                    axis=alt.Axis(title=objective_name),
                ),
                opacity=opacity,
                tooltip=detail_tooltip,
                **jobs_props,
            )
        )

        cum_obj_chart = (
            alt.Chart(trials_df_by_tst)
            .mark_line(
                interpolate="step-after",
                opacity=1.0,
                strokeDash=[3, 3],
                strokeWidth=2.0,
            )
            .encode(
                x=alt.X("TrainingStartTime:T", scale=alt.Scale(nice=True)),
                y=alt.Y("cum_objective:Q", scale=alt.Scale(zero=False, padding=1)),
                stroke=alt.Stroke("TuningJobName:N", legend=None),
            )
        )

        if advanced:
            return cum_obj_chart + progress_chart
        return progress_chart

    progress_chart = render_progress_chart()

    # First Row
    # KDE Training Objective
    result_hist_chart = (
        alt.Chart(trials_df)
        .transform_filter(brush)
        .transform_density(objective_name, bandwidth=0.01)
        .mark_area()
        .encode(
            x=alt.X("value:Q", scale=objective_scale, title=objective_name),
            y="density:Q",
        )
    )
    # Training Jobs
    training_jobs_chart = (
        alt.Chart(trials_df.sort_values(objective_name), title="Training Jobs")
        .mark_bar()
        .add_params(brush)
        .add_params(job_highlight_selection)
        .encode(
            y=alt.Y(f"{objective_name}:Q"),
            x=alt.X("TrainingJobName:N", sort=None),
            color=alt.Color("TrainingJobName:N"),
            opacity=opacity,
            **jobs_props,
        )
    )

    # Job Level Stats

    training_job_name_encodings = {
        "color": alt.condition(
            brush,
            alt.Color("TrainingJobName:N", legend=None),
            alt.value("grey"),
        ),
        "opacity": alt.condition(brush, alt.value(1.0), alt.value(0.3)),
        "strokeWidth": alt.condition(brush, alt.value(2.5), alt.value(0.8)),
    }

    duration_format = "%M:%S"
    metrics_tooltip = [
        "TrainingJobName:N",
        "value:Q",
        "label:N",
        alt.Tooltip("ts:T", format="%e:%H:%M"),
        alt.Tooltip("rel_ts:T", format="%e:%H:%M"),
    ]

    job_level_rows = alt.HConcatChart()

    # Use CW metrics
    if not full_df.empty:
        # Objective Progression

        objective_progression_chart = None
        # Suppress diagram if we only have one, final, value
        if (
            full_df.loc[full_df.label == objective_name]
            .groupby(["TuningJobName", "TrainingJobName"])[objective_name]
            .count()
            .max()
            > 1
        ):
            objective_progression_chart = (
                alt.Chart(full_df, title=f"Progression {objective_name}", width=400)
                .transform_filter(alt.FieldEqualPredicate(field="label", equal=objective_name))
                .mark_line(point=True)
                .encode(
                    x=alt.X("rel_ts:T", axis=alt.Axis(format=duration_format)),
                    y=alt.Y("value:Q", scale=alt.Scale(zero=False)),
                    **training_job_name_encodings,
                    tooltip=metrics_tooltip,
                )
                .interactive()
            )

            if multiple_job_status:
                objective_progression_chart = objective_progression_chart.encode(
                    strokeDash=alt.StrokeDash("TrainingJobStatus:N", legend=None)
                )

            # Secondary chart showing the same contents, but by absolute time.
            objective_progression_absolute_chart = objective_progression_chart.encode(
                x=alt.X("ts:T", scale=alt.Scale(nice=True))
            )

            objective_progression_chart = (
                objective_progression_chart | objective_progression_absolute_chart
            )

        ###

        job_metrics_charts = []
        for metric in job_metrics:
            metric_chart = (
                alt.Chart(full_df, title=metric, width=400)
                .transform_filter(alt.FieldEqualPredicate(field="label", equal=metric))
                .encode(
                    y=alt.Y("value:Q", scale=alt.Scale(zero=False)),
                    **training_job_name_encodings,
                    tooltip=metrics_tooltip,
                )
                .interactive()
            )

            if (
                full_df.loc[full_df.label == metric]
                .groupby(["TuningJobName", "TrainingJobName"])
                .count()
                .value.max()
                == 1
            ):
                # single value, render as a bar over the training jobs on the x-axis
                metric_chart = metric_chart.encode(
                    x=alt.X("TrainingJobName:N", sort=None)
                ).mark_bar(interpolate="linear", point=True)
            else:
                # multiple values, render the values over time on the x-axis
                metric_chart = metric_chart.encode(
                    x=alt.X("rel_ts:T", axis=alt.Axis(format=duration_format))
                ).mark_line(interpolate="linear", point=True)

            job_metrics_charts.append(metric_chart)

        job_metrics_chart = _columnize(job_metrics_charts, 3)

        # Job instance
        # 'MemoryUtilization', 'CPUUtilization'
        instance_metrics_chart = (
            alt.Chart(full_df, title="CPU and Memory")
            .transform_filter(
                alt.FieldOneOfPredicate(
                    field="label",
                    oneOf=[
                        "MemoryUtilization",
                        "CPUUtilization",
                    ],
                )
            )
            .mark_line()
            .encode(
                x=alt.X("rel_ts:T", axis=alt.Axis(format=duration_format)),
                y="value:Q",
                **training_job_name_encodings,
                strokeDash=alt.StrokeDash("label:N", legend=alt.Legend(orient="bottom")),
                tooltip=metrics_tooltip,
            )
            .interactive()
        )

        if "GPUUtilization" in full_df.label.values:
            instance_metrics_chart = (
                instance_metrics_chart
                | alt.Chart(full_df, title="GPU and GPU Memory")
                .transform_filter(
                    alt.FieldOneOfPredicate(
                        field="label",
                        oneOf=[
                            "GPUMemoryUtilization",
                            "GPUUtilization",
                        ],
                    )
                )
                .mark_line()
                .encode(
                    x=alt.X("rel_ts:T", axis=alt.Axis(format=duration_format)),
                    y=alt.Y("value:Q"),
                    **training_job_name_encodings,
                    strokeDash=alt.StrokeDash("label:N", legend=alt.Legend(orient="bottom")),
                    tooltip=metrics_tooltip,
                )
                .interactive()
            )

        job_level_rows = job_metrics_chart & instance_metrics_chart
        if objective_progression_chart:
            job_level_rows = objective_progression_chart & job_level_rows
        job_level_rows = job_level_rows.resolve_scale(strokeDash="independent").properties(
            title="Job / Instance Level Metrics"
        )

    overview_row = (progress_chart | result_hist_chart).properties(
        title="Hyper Parameter Tuning Job"
    )
    detail_rows = detail_charts.properties(title="Hyper Parameter Details")
    if job_level_rows:
        job_level_rows = training_jobs_chart & job_level_rows

    return overview_row & detail_rows & job_level_rows


def _clean_parameter_name(s):
    """Helper method to ensure proper parameter name characters for altair 5+"""
    return s.replace(":", "_").replace(".", "_")


def _prepare_training_job_metrics(jobs):
    """Fetches and combines CloudWatch metrics for multiple training jobs.

    Args:
        jobs (list): List of (job_name, start_time, end_time) tuples.

    Returns:
        pandas.DataFrame: Combined metrics DataFrame with 'TrainingJobName' column.
    """
    df = pd.DataFrame()
    for job_name, start_time, end_time in jobs:
        job_df = get_cw_job_metrics(
            job_name,
            start_time=pd.Timestamp(start_time) - pd.DateOffset(hours=8),
            end_time=pd.Timestamp(end_time) + pd.DateOffset(hours=8),
        )
        if job_df is None:
            logger.info("No CloudWatch metrics for %s. Skipping.", job_name)
            continue

        job_df["TrainingJobName"] = job_name
        df = pd.concat([df, job_df])
    return df


def _prepare_consolidated_df(trials_df):
    """Merges training job metrics with trials data into a consolidated DataFrame."""
    if trials_df.empty:
        return pd.DataFrame()

    logger.debug("Cache Hit/Miss: ", end="")
    jobs_df = _prepare_training_job_metrics(
        zip(
            trials_df.TrainingJobName.values,
            trials_df.TrainingStartTime.values,
            trials_df.TrainingEndTime.values,
        )
    )
    logger.info("")

    if jobs_df.empty:
        return pd.DataFrame()

    merged_df = pd.merge(jobs_df, trials_df, on="TrainingJobName")
    return merged_df


def _get_df(tuning_job_name, filter_out_stopped=False):
    """Retrieves hyperparameter tuning job results and returns preprocessed DataFrame.

    Returns a DataFrame containing tuning metrics and parameters for the specified job.
    """

    tuner = sagemaker.HyperparameterTuningJobAnalytics(tuning_job_name)

    df = tuner.dataframe()
    if df.empty:  # HPO job just started; no results yet
        return df

    df["TuningJobName"] = tuning_job_name

    # Filter out jobs without FinalObjectiveValue
    df = df[df["FinalObjectiveValue"] > -float("inf")]

    # Jobs early stopped by AMT are reported with their last
    # objective value, before they are stopped.
    # However this value may not be a good representation
    # of the eventual objective value we would have seen
    # if run without stopping. Therefore it may be confusing
    # to include those runs.
    # For now, if included, we use a different mark to
    # discriminate visually between a stopped and finished job

    if filter_out_stopped:
        df = df[df["TrainingJobStatus"] != "Stopped"]

    # Preprocessing values for [32], [64] etc.
    for tuning_range in tuner.tuning_ranges.values():
        parameter_name = tuning_range["Name"]
        if df.dtypes[parameter_name] == "O":
            try:
                # Remove decorations, like []
                df[parameter_name] = df[parameter_name].apply(
                    lambda v: v.replace("[", "").replace("]", "").replace('"', "")
                )

                # Is it an int? 3 would work, 3.4 would fail.
                try:
                    df[parameter_name] = df[parameter_name].astype(int)
                except ValueError:
                    # A float then?
                    df[parameter_name] = df[parameter_name].astype(float)

            except (ValueError, TypeError, AttributeError):
                # Catch exceptions that might occur during string manipulation or type conversion
                # - ValueError: Could not convert string to float/int
                # - TypeError: Object doesn't support the operation
                # - AttributeError: Object doesn't have replace method
                # Leaving the value untouched
                pass

    return df


def _get_tuning_job_names_with_parents(tuning_job_names):
    """Resolve dependent jobs, one level only"""

    all_tuning_job_names = []
    for tuning_job_name in tuning_job_names:
        tuning_job_result = sm.describe_hyper_parameter_tuning_job(
            HyperParameterTuningJobName=tuning_job_name
        )

        # find parent jobs and retrieve all tuner dataframes
        parent_jobs = []
        if "WarmStartConfig" in tuning_job_result:
            parent_jobs = [
                cfg["HyperParameterTuningJobName"]
                for cfg in tuning_job_result["WarmStartConfig"]["ParentHyperParameterTuningJobs"]
            ]
            if parent_jobs:
                logger.info("Tuning job %s's parents: %s", tuning_job_name, ", ".join(parent_jobs))
        all_tuning_job_names.extend([tuning_job_name, *parent_jobs])

    # return de-duplicated tuning job names
    return list(set(all_tuning_job_names))


def get_job_analytics_data(tuning_job_names):
    """Retrieves and processes analytics data from hyperparameter tuning jobs.

    Args:
        tuning_job_names (str or list): Single tuning job name or list of names/tuner objects.

    Returns:
        tuple: (DataFrame with training results, tuned params list, objective name, is_minimize).

    Raises:
        ValueError: If tuning jobs have different objectives or optimization directions.
    """
    if not isinstance(tuning_job_names, list):
        tuning_job_names = [tuning_job_names]

    # Ensure to create a list of tuning job names (strings)
    tuning_job_names = [
        (
            tuning_job.describe()["HyperParameterTuningJobName"]
            if isinstance(tuning_job, sagemaker.tuner.HyperparameterTuner)
            else tuning_job
        )
        for tuning_job in tuning_job_names
    ]

    # Maintain combined tuner dataframe from all tuning jobs
    df = pd.DataFrame()

    # maintain objective, direction of optimization and tuned parameters
    objective_name = None
    is_minimize = None
    tuned_parameters = None

    all_tuning_job_names = _get_tuning_job_names_with_parents(tuning_job_names)

    for tuning_job_name in all_tuning_job_names:
        tuning_job_result = sm.describe_hyper_parameter_tuning_job(
            HyperParameterTuningJobName=tuning_job_name
        )
        status = tuning_job_result["HyperParameterTuningJobStatus"]
        logger.info("Tuning job %-25s status: %s", tuning_job_name, status)

        df = pd.concat([df, _get_df(tuning_job_name)])

        # maintain objective and assure that all tuning jobs use the same
        job_is_minimize = (
            tuning_job_result["HyperParameterTuningJobConfig"]["HyperParameterTuningJobObjective"][
                "Type"
            ]
            != "Maximize"
        )
        job_objective_name = tuning_job_result["HyperParameterTuningJobConfig"][
            "HyperParameterTuningJobObjective"
        ]["MetricName"]
        job_tuned_parameters = [
            v["Name"]
            for v in sagemaker.HyperparameterTuningJobAnalytics(
                tuning_job_name
            ).tuning_ranges.values()
        ]

        if not objective_name:
            objective_name = job_objective_name
            is_minimize = job_is_minimize
            tuned_parameters = job_tuned_parameters
        else:
            if (
                objective_name != job_objective_name
                or is_minimize != job_is_minimize
                or set(tuned_parameters) != set(job_tuned_parameters)
            ):
                raise ValueError(
                    "All tuning jobs must use the same objective and optimization direction."
                )

    if not df.empty:
        # Cleanup wrongly encoded floats, e.g. containing quotes.
        for i, dtype in enumerate(df.dtypes):
            column_name = str(df.columns[i])
            if column_name in [
                "TrainingJobName",
                "TrainingJobStatus",
                "TuningJobName",
            ]:
                continue
            if dtype == "object":
                val = df[column_name].iloc[0]
                if isinstance(val, str) and val.startswith('"'):
                    try:
                        df[column_name] = df[column_name].apply(lambda x: int(x.replace('"', "")))
                    except (ValueError, TypeError, AttributeError):
                        # noqa: E722 nosec b110 if we fail, we just continue with what we had
                        pass  # Value is not an int, but a string

        df = df.sort_values("FinalObjectiveValue", ascending=is_minimize)
        df[objective_name] = df.pop("FinalObjectiveValue")

        # Fix potential issue with dates represented as objects, instead of a timestamp
        # This can in other cases lead to:
        # https://www.markhneedham.com/blog/2020/01/10/altair-typeerror-object-type-
        # date-not-json-serializable/
        # Seen this for TrainingEndTime, but will watch TrainingStartTime as well now.
        df["TrainingEndTime"] = pd.to_datetime(df["TrainingEndTime"])
        df["TrainingStartTime"] = pd.to_datetime(df["TrainingStartTime"])

        logger.info("")
        logger.info("Number of training jobs with valid objective: %d", len(df))
        logger.info("Lowest: %s Highest %s", min(df[objective_name]), max(df[objective_name]))

        tuned_parameters = [_clean_parameter_name(tp) for tp in tuned_parameters]

    return df, tuned_parameters, objective_name, is_minimize
