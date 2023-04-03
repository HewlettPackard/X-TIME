###
# Copyright (2023) Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###

import typing as t

import pandas as pd

from xtime.contrib.tune_ext import Analysis
from xtime.io import IO


def describe(report_type: str, run: t.Optional[str] = None, file: t.Optional[str] = None) -> None:
    """Describe experiment with `REPORT_TYPE` report stored as MLflow `RUN` run, optionally save to `OUTPUT` file."""
    if report_type in ("summary", "best_trial") and run is None:
        print("When report type is `summary` or `best_trial`, MLflow run ID must be provided (--run=RUN_ID).")
        exit(1)

    summary: t.Optional[t.Union[t.Dict, pd.DataFrame]] = None
    if report_type == "summary":
        summary: t.Dict = Analysis.get_summary(run)
    elif report_type == "best_trial":
        summary: t.Dict = Analysis.get_best_trial(run)
    elif report_type == "final_trials":
        summary: pd.DataFrame = Analysis.get_final_trials()

    if file:
        IO.save_to_file(summary, file)
    else:
        IO.print(summary)
