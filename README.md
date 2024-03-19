# Empowering Small-Scale Knowledge Graphs

This repository contains the code used in the research paper titled 
*"Empowering Small-Scale Knowledge Graphs: A Strategy of
Leveraging General-Purpose Knowledge Graphs for Enriched Embeddings"* 
authored by Albert Sawczyn, Jakub Binkowski, Piotr Bielak, and Tomasz Kajdanowicz. 
The paper was accepted for the LREC-COLING 2024 conference.

You can find the paper here - TBA.

## Citation

```bibtex
TBA
```

## Getting Started

### Prerequisites

* Configured Python 3.10 environment.
* Installed Poetry.

### Installing

To install all dependencies, run:

```bash
poetry install 
```

### Downloading data

The repository uses DVC to manage the data. To download the data, run:

```bash
dvc pull
```

## Reproducing

The repository uses DVC to manage the experiments. 
* `dvc.yaml` contains the stages used before the training and after the training. Each stage is described in the comments.
* `experiments/configs/training/dvc.yaml` contains the training stages.  

To reproduce all stages run:

```bash
dvc repro
```

### Logging

The repository uses WandB to log the experiments. Before running the experiments, change the 
`wandb_entity` and `wandb_project` in 
`experiments/configs/training/{WN18RR.yaml,FB15k237.yaml,WD50K.yaml}` or set the environment 
variable `WANDB_MODE=offline`.

The logs can be found [here](https://wandb.ai/graph-ml-lab-wust/empowering-small-scale-kg).

Keep in mind that the metric values calculated during training can differ from those calculated
after training is completed. This is due to the fact that evaluation during training
is done without restricting entities to DKG.
