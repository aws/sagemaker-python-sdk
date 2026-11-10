API Reference
=============

DataPreparer
------------

.. autoclass:: sagemaker.ai_registry.data_preparer.preparer.DataPreparer
   :members: generate, evaluate, transform, stop
   :undoc-members:

DatasetJobConfig
----------------

.. autoclass:: sagemaker.ai_registry.data_preparer.configs.DatasetJobConfig
   :members: from_template, schema, print_schema, required_fields, to_job_config_document, dynamic_model_fields, dynamic_model_json_schema, resolved_schema_version, resolved_schema
   :undoc-members:

DataSet
-------

.. autoclass:: sagemaker.ai_registry.dataset.DataSet
   :members:
   :undoc-members:

JobCategory
-----------

.. autoclass:: sagemaker.ai_registry.data_preparer.constants.JobCategory
   :members:
   :undoc-members:

JobStatus
---------

.. autoclass:: sagemaker.ai_registry.data_preparer.constants.JobStatus
   :members:
   :undoc-members:

Auto-labeling options
---------------------

Typed per-task options for ``DataPreparer.generate``, passed as
``DatasetJobConfig(dataset_options=...)``.

.. automodule:: sagemaker.ai_registry.data_preparer.auto_labeling.shapes
   :members:
   :undoc-members:

LabelingSummary
---------------

.. autoclass:: sagemaker.ai_registry.data_preparer.auto_labeling.summary.LabelingSummary
   :members: from_dataset
   :undoc-members:

Exceptions
----------

.. automodule:: sagemaker.ai_registry.data_preparer.exceptions
   :members:
   :undoc-members:

Operators
---------

Client-side functions for an auto-labeling project. Import each name from
``sagemaker.ai_registry.data_preparer.auto_labeling``; where it is implemented is not part
of the import path. Each operator takes its inputs as parameters and returns a result
object; a refusal is raised as
:class:`~sagemaker.ai_registry.data_preparer.exceptions.DataPreparerOperatorError`.

Quick start: a labeling project in seven calls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The project-level functions follow the auto-labeling loop step for step and own the project
folder (its splits, work files and ``run-log.jsonl``). A project is a folder holding the
labeled seed (``seed.jsonl``) and the labeling spec (``labeling_spec.md``).

.. code-block:: python

    from sagemaker.ai_registry.data_preparer.auto_labeling import (
        inspect_labeling_data, prepare_labeling_seed, draft_labeling_spec, tune_labeling_spec, select_labeling_spec, label_corpus,
    )

    PROJ, S3, ROLE = "tickets", "s3://my-bucket/autolabel", "arn:aws:iam::123456789012:role/SDG"

    # What is in my data, and can it be labeled? fields, task, label set, taxonomy gate
    data = inspect_labeling_data(spec=f"{PROJ}/labeling_spec.md", seed=f"{PROJ}/seed.jsonl",
                        corpus="s3://my-bucket/tickets.jsonl", region="us-west-2")
    # Protect the seed: split dev / val / test once, freeze it, pick demos, build folds
    seed = prepare_labeling_seed(PROJ, data)
    # Optional: write the spec's rules from the dev split
    draft = draft_labeling_spec(PROJ, data)
    # Tune: one next() per spec version labels dev and val, reports the scores and the
    # failures, and says whether to refine; edit the spec between calls
    rounds = tune_labeling_spec(PROJ, data, spec=f"{PROJ}/labeling_spec.md", s3_prefix=S3, role_arn=ROLE)
    for findings in rounds:
        print(findings.summary())           # val accuracy, dev failures, REFINE / ACCEPT / STOP
    # Which spec ships: the gold gate on val with the do-no-harm floor; copies it to labeling_spec.md
    decision = select_labeling_spec(PROJ, rounds)
    # Label the corpus with it; the job certifies itself on the frozen test split
    result = label_corpus(PROJ, data, corpus="s3://my-bucket/tickets.jsonl", dataset_name="tickets-labeled",
                          s3_prefix=S3, role_arn=ROLE, target=0.85, jury_models="<id1>,<id2>")
    result.dataset_arn                      # the labeled DataSet; the labels were never downloaded

Each subsystem below lists its operators first, then the helpers a caller uses beside
them, then the exceptions it catches and the constants it compares against.

Results
~~~~~~~

.. autoclass:: sagemaker.ai_registry.data_preparer._operator_result.OperatorResult
   :members:

Inspect: what is in my data, and can this be labeled?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.inspect.inputs.inspect
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.inspect.inputs.inspect_labeling_data
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.inspect.task.detect
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.inspect.taxonomy.gate
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.inspect.taxonomy.parse_label_set

Seed: what am I measuring against, and is it still trustworthy?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.cold_start.draw_sample
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.split.freeze
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.split.prepare_labeling_seed
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.split.guard_read
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.split.rehash
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.split.relabel
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.folds.build_kfold
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.demos.select_demos
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.labels.check_seed_append
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.labels.merge_seed_labels
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.labels.build_overlay
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.review_export.flatten_export
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.split.find_splits_dir
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.split.load_state
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.split.state_path
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.split.test_ids
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.split.test_read_event
.. autoexception:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.split.SplitGuardError
.. autoexception:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.labels.SeedAppendRefused

Spec: what are the rules, and which version ships?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.spec.induce.induce_spec
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.spec.induce.draft_labeling_spec
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.spec.select.select
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.spec.select.select_labeling_spec
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.spec.adopt.adopt
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.spec.rubric.rubric_gaps
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.spec.adopt.check_pick_folder
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.spec.adopt.check_shippable
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.spec.adopt.no_adopted_spec_message
.. autodata:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.RUBRIC_TASKS
   :annotation:
.. autodata:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.PICK_FILES
   :annotation:
.. autoexception:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.spec.adopt.AdoptError

Job: run it and read it back
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.round.run_labeling_round
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.round.tune_labeling_spec
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.round.label_corpus
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.round.prepare_round
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.round.submit_round
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.round.read_round
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.register.resolve_dataset_ref
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.register.register_labeled_output
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.job._scratch.scratch_cleanup
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.dt_handoff.dt_view
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.run_log.labeling_run_record
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.round.round_summary
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.request.build_labeling_request
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.request.spec_score_scale
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.request.scratch_dataset_name
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.client.upload_text
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.client.submit_labeling_job
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.run_log.read_labeling_runs
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.run_log.latest_labeling_runs
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.run_log.labeling_run_invocation
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.run_log.labeling_run_for_output
.. autoexception:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.round.RoundRefused
.. autodata:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.MAX_SPEC_BYTES
   :annotation:
.. autoexception:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.request.LabelingRequestError
.. autoexception:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.client.CreateJobRefused
.. autoexception:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.client.OptionRefused
.. autoexception:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.client.LabelingBackendUnavailable
.. autoexception:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.client.JobWaitError

Score: how good is it, and can I prove it?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.score.scorers.score
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.score.scorers.parse_label_map
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.score.scorers.label_map_from_names
.. autodata:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.MAX_ROWS
   :annotation:

Review: which rows need a person?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.review.golden.route
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.review.golden.apply
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.review.golden.check_reviewed
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.review.decisions.apply_decisions
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.review.request.review_result
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.review.decisions.review_inputs
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.review.decisions.logged_decisions
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.review.decisions.merge_decisions
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.review.decisions.touched_test_rows
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.review.decisions.decision_events
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.review.decisions.dry_run_summary
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.review.decisions.default_label_set
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.review.request.review_rows
.. autoexception:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.review.golden.GoldenReviewError
.. autoexception:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.review.decisions.OverrideOutsideItem

Client reads and the row rule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

What the operators read on the client is announced, and the row-correctness rule is the job's.

.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators._rows.read_input_rows
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators._rows.read_jsonl
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators._rows.head_rows
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators._rows.report_client_read
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators._rows.print_client_reads
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators._rows.client_read_message
.. autofunction:: sagemaker.ai_registry.data_preparer.auto_labeling.operators._rowrule.job_tolerance
.. autodata:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.DEFAULT_CORRECT_TOLERANCE
   :annotation:
.. autodata:: sagemaker.ai_registry.data_preparer.auto_labeling.operators.GOLDEN_REVIEW_FIELD
   :annotation:
