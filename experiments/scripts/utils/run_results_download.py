from pathlib import Path

import pandas as pd
import typer

from mgi.defaults import EXPERIMENTS_RESULTS_PATH
from mgi.results.wandb_helpers import get_results


def main(
    project: list[str] = typer.Option(...),
    output_path: Path = typer.Argument(EXPERIMENTS_RESULTS_PATH),
) -> None:
    print(locals())
    results = []
    for project_ in project:
        results.append(get_results(project_))
    results_df = pd.concat(results).reset_index()
    output_path.mkdir(exist_ok=True, parents=True)
    results_df.to_pickle(output_path / "results.pkl")


typer.run(main)
