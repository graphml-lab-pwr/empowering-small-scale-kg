import csv
import gzip
from pathlib import Path

import typer
from rich import print
from tqdm import tqdm

from mgi.defaults import CONCEPTNET_DATASET_PATH, CONCEPTNET_EN_DATASET_PATH


def main(
    conceptnet_path: Path = typer.Argument(
        CONCEPTNET_DATASET_PATH / "original/conceptnet-assertions-5.7.0.csv.gz"
    ),
    output_path: Path = typer.Argument(CONCEPTNET_EN_DATASET_PATH),
) -> None:
    all_lines = 0
    english_lines = 0

    entities = set()
    relations = set()
    CONCEPTNET_EN_DATASET_PATH.mkdir(exist_ok=True)
    with gzip.open(conceptnet_path, "rt") as csvfile, open(
        output_path / "train.tsv", "w"
    ) as output_file:
        cn_reader = csv.reader(csvfile, delimiter="\t")
        cn_writer = csv.writer(output_file, delimiter="\t")
        for row in tqdm(cn_reader, total=34074917):
            all_lines += 1
            if row[2].startswith("/c/en/") and row[3].startswith("/c/en/"):
                english_lines += 1
                entities.add(row[2])
                entities.add(row[3])
                relations.add(row[1])
                cn_writer.writerow([row[2], row[1], row[3]])

    print("All lines: ", all_lines)
    print("English lines: ", english_lines)
    print("English lines percentage: ", english_lines / all_lines * 100)
    print("Entities: ", len(entities))
    print("Relations: ", len(relations))

    with open(CONCEPTNET_EN_DATASET_PATH / "valid.tsv", "w") as f:
        f.write("nan\tnan\tnan\n")
    with open(CONCEPTNET_EN_DATASET_PATH / "test.tsv", "w") as f:
        f.write("nan\tnan\tnan\n")


typer.run(main)
