from pathlib import Path
import pandas as pd


class NameEntityRecognitionModel:

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
