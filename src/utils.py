import os

import nltk
from pathlib import Path

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')

DEFAULT_PATH_MODELS = Path(os.path.dirname(__file__)).joinpath('data')
DEFAULT_PATH_RAW_DATA = Path(os.path.dirname(__file__)).joinpath('data/raw_data')


def get_model_path(file: str) -> Path:
    return DEFAULT_PATH_MODELS.joinpath(file)


def get_data_path(file: str) -> Path:
    return DEFAULT_PATH_RAW_DATA.joinpath(file)
