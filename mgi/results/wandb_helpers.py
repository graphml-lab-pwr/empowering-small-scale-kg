from typing import Any

import pandas as pd
import wandb
from tqdm import tqdm
from wandb.apis.public import Run


def get_results(project: str) -> pd.DataFrame:
    api = wandb.Api()
    runs = api.runs(project)

    results = []
    for run in tqdm(runs, desc="Downloading results"):
        results.append(_acquire_run_data(run))
    df = pd.DataFrame(results)
    df = df[df.state == "finished"].reset_index(drop=True)
    return df


def _acquire_run_data(run: Run) -> dict[str, Any]:
    return {
        "name": run.name,
        "state": run.state,
        "summary": run.summary._json_dict,
        "config": run.config,
        "metrics": run.history(),  # type: ignore #  Call to untyped function
    }
