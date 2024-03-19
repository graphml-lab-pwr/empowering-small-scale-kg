from pathlib import Path

import pandas as pd
import requests
import srsly
import typer
from mpire import WorkerPool
from rich import print

from mgi.defaults import WD50K_DATASET_PATH, WD50K_RAW_DATASET_PATH


def fetch_label(qid: str) -> tuple[str, str]:
    try:
        url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&format=json&ids={qid}"
        response = requests.get(url)
        data = response.json()

        if "entities" in data and qid in data["entities"]:
            entity = data["entities"][qid]
            if "labels" in entity and "en" in entity["labels"]:
                english_label = entity["labels"]["en"]["value"]
                return qid, english_label
    except Exception as e:
        print(response)
        print(e)

    return qid, qid


def map_qids_to_labels(qids: set[str], n_jobs: int) -> dict[str, str]:
    entity_labels = {}

    with WorkerPool(n_jobs=n_jobs, start_method="threading") as pool:
        results = pool.map(fetch_label, qids, progress_bar=True)

    for qid, label in results:
        entity_labels[qid] = label

    return entity_labels


def main(
    wd_path: Path = typer.Argument(WD50K_RAW_DATASET_PATH),
    output_path: Path = typer.Argument(WD50K_DATASET_PATH),
    n_jobs: int = typer.Option(800),
) -> None:
    train = pd.read_csv(wd_path / "train.txt", header=None)
    valid = pd.read_csv(wd_path / "valid.txt", header=None)
    test = pd.read_csv(wd_path / "test.txt", header=None)

    qids = (
        set(train[0]) | set(train[2]) | set(valid[0]) | set(valid[2]) | set(test[0]) | set(test[2])
    )
    mapping = map_qids_to_labels(qids, n_jobs)
    output_path.mkdir(parents=True, exist_ok=True)
    srsly.write_json(output_path / "qid_to_label.json", mapping)
    train.to_csv(output_path / "train.tsv", header=False, index=False, sep="\t")
    valid.to_csv(output_path / "valid.tsv", header=False, index=False, sep="\t")
    test.to_csv(output_path / "test.tsv", header=False, index=False, sep="\t")


typer.run(main)
