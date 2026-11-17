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
"""The result every data-preparation operator returns, and the rules behind it.

An operator is a function: inputs are parameters, outputs are the returned object, and
a refusal is an exception. Three rules make the results composable:

1. **A result is a model, not a bag.** Every operator returns an
   :class:`OperatorResult` subclass with named, typed fields. ``to_json()`` gives
   the JSON-ready dict a command-line caller writes to disk, with unset optional
   fields left out, so the files the skill writes keep their shape.
2. **Events and warnings are returned, never written.** An operator that would append
   to the project's run log (a test-split read, a job launch, a decision) puts the
   entry in its result's ``events`` list, as the plain dict the project's writer
   accepts (``kind``, ``step``, ``iteration``, ``status``, ``arn`` for a job, and
   the kind's fields). The caller appends them. An operator that needs to *read* the
   run log takes the entries as a parameter. Likewise a note for the person running
   it (a deprecated input accepted this once, a default assumed) goes in
   ``warnings``; the caller decides where to show it.
3. **Refusals are exceptions.** A wrong input raises
   :class:`~sagemaker.ai_registry.data_preparer.exceptions.DataPreparerOperatorError`
   (or a subclass) whose message says what to pass instead. A command-line caller
   prints the message and exits non-zero.

What an operator does on the client, and the exceptions the labeling primitive makes:

* **Local files.** An operator writes none; the caller writes what the result carries
  (``texts``, ``copies``, a state document). The exceptions are the transfers a caller
  asks for by path: a DataSet or ``s3://`` reference resolved with ``download`` lands at
  the caller's ``out_path``, and a round's read-back downloads the job's artifacts under
  the caller's output folder. Uploads go to S3 only.
* **The project directory.** An operator takes its inputs as parameters, not by
  convention, with one exception: the frozen seed split's state
  (``splits/split_state.json``, with the split files beside it). The split guard finds
  it from the file being read or the caller's ``project_dir``, so a read of a split
  file is checked wherever it happens; the operators that guard a read (the labeling
  round, the golden review, the split operators themselves) inherit the convention.
  A second primitive that needs a project convention of its own names it here.
* **The project-level operators.** The labeling primitive's ``inspect_labeling_data``,
  ``prepare_labeling_seed``, ``draft_labeling_spec``, ``run_labeling_round`` / ``tune_labeling_spec``,
  ``select_labeling_spec`` and ``label_corpus`` are the loop's steps as one call each. They
  compose the step operators above and, by design, own the project folder: they write
  its splits, work files and ``run-log.jsonl``, which is what the skill's templates did
  around the same calls. They are the layer that writes; the step operators still do not.

Shared by every primitive from day one: an operator of the labeling, evaluation,
generation or transformation primitive returns this shape or there is no package.
Everything else an operator uses stays private to its primitive until a second one
imports it.
"""

from __future__ import absolute_import

from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field


class OperatorResult(BaseModel):
    """Base of every operator result.

    Subclasses declare their fields. ``events`` holds the run-log entries the
    operation produced, for the caller to append to the project's log; ``warnings``
    holds notes for the person running it, for the caller to show.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    events: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        """The JSON-ready dict of this result.

        Without ``events`` and ``warnings``, and without optional fields that were
        not set.
        """
        return self.model_dump(exclude={"events", "warnings"}, exclude_none=True, mode="json")
