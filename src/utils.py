import os
from enum import Enum

import nltk
import numpy as np
import pandas as pd
from nltk import word_tokenize
from pathlib import Path

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')

DEFAULT_PATH_MODELS = Path(os.path.dirname(__file__)).joinpath('data')
DEFAULT_PATH_RAW_DATA = Path(os.path.dirname(__file__)).joinpath('data/raw_data')


class ImageEntity(Enum):
    CAST = 'cast'
    MOVIE = 'movie'


def get_model_path(file: str) -> Path:
    return DEFAULT_PATH_MODELS.joinpath(file)


def get_data_path(file: str) -> Path:
    return DEFAULT_PATH_RAW_DATA.joinpath(file)


def remove_stop_words(_stop_words, sentence: str) -> str:
    word_tokens = word_tokenize(sentence)
    filtered_sentence = [w for w in word_tokens if not w.lower() in _stop_words]
    return ' '.join(filtered_sentence)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path.resolve(), index_col=0,
                       converters={'sentence_embedding': lambda s: np.fromstring(s[1:-1], sep=', '),
                                   'embedding': lambda s: np.fromstring(s[1:-1], sep=', ')})
