import os
from pathlib import Path

ROOT_PATH = Path(os.path.dirname(__file__)).parent.absolute()
DATA_PATH = ROOT_PATH / "data"

WN18RR_DATASET_PATH = DATA_PATH / "external/datasets/WN18RR/text"
FB15K237_DATASET_PATH = DATA_PATH / "external/datasets/FB15k-237"
CONCEPTNET_DATASET_PATH = DATA_PATH / "external/datasets/conceptnet"
CONCEPTNET_EN_DATASET_PATH = CONCEPTNET_DATASET_PATH / "en"
YAGO310_DATASET_PATH = DATA_PATH / "external/datasets/YAGO3-10"
WD50K_RAW_DATASET_PATH = DATA_PATH / "external/datasets/WD50K/wd50k/triples"
WD50K_DATASET_PATH = DATA_PATH / "external/datasets/WD50K_decoded/wd50k"
SAMPLED_DATASETS = DATA_PATH / "sampled_datasets"

CONFIGS_PATH = ROOT_PATH / "experiments/configs"
SAMPLING_CONFIGS_PATH = CONFIGS_PATH / "sampling"
TRAINING_CONFIGS_PATH = CONFIGS_PATH / "training"

FASTTEXT_MODEL_PATH = DATA_PATH / "external/models/fasttext/cc.en.300.bin"

EXPERIMENTS_RESULTS_PATH = DATA_PATH / "experiments_results"

PLOTS_PATH = DATA_PATH / "plots"
