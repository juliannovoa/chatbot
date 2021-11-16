from pathlib import Path
import pandas as pd


class NameEntityRecognitionModel:
    EXPECTED_DATA_COLUMNS = {'lemma', 'next-lemma', 'next-next-lemma', 'next-next-pos', 'next-next-shape',
                             'next-next-word', 'next-pos', 'next-shape', 'next-word', 'pos', 'prev-iob',
                             'prev-lemma', 'prev-pos', 'prev-prev-iob', 'prev-prev-lemma', 'prev-prev-pos',
                             'prev-prev-shape', 'prev-prev-word', 'prev-shape', 'prev-word', 'sentence_idx',
                             'shape', 'word', 'tag'}

    @classmethod
    def load_dataset(cls, dataset: Path, size: int = -1) -> pd.DataFrame:
        if not dataset.is_file():
            raise FileNotFoundError("Dataset not found. Download dataset from "
                                    "https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus")

        df = pd.read_csv(dataset, encoding="ISO-8859-1")
        df = df.fillna(method='ffill')

        if size == -1:
            return df

        if size < 0 or size > len(df):
            raise ValueError('Invalid size value')

        return df[:size]

    @classmethod
    def check_data(cls, df: pd.DataFrame) -> bool:
        input_columns_set = set(df.columns.values)
        return input_columns_set == NameEntityRecognitionModel.EXPECTED_DATA_COLUMNS
