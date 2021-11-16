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

    @classmethod
    def extract_features(cls, s: pd.Series) -> dict:
        print(s)
        word = s['word']
        postag = s['pos']
        features = {
            'word.lower()': word.lower(),  # the word in lowercase
            'word[-3:]': word[-3:],  # last three characters
            'word[-2:]': word[-2:],  # last two characters
            'word.isupper()': word.isupper(),  # true, if the word is in uppercase
            'word.istitle()': word.istitle(),
            # true, if the first character is in uppercase and remaining characters are in lowercase
            'word.isdigit()': word.isdigit(),  # true, if all characters are digits
            'postag': postag,  # POS tag
            'postag[:2]': postag[:2],  # IOB prefix
        }

        for p in ['prev-prev-', 'prev-', 'next-', 'next-next-']:
            word = s[f'{p}word']  # the neighbors
            if word.startswith('__START') or word.startswith('__END'):
                features.update({
                    f'{p}word:word.lower()': word.lower(),
                    f'{p}word:word.istitle()': False,
                    f'{p}word:word.isupper()': False,
                    f'{p}word:postag': word,
                    f'{p}word:postag[:2]': word,
                })
            else:
                postag = s[f'{p}pos']  # POS tag of the neighbors
                features.update({
                    f'{p}word:word.lower()': word.lower(),
                    f'{p}word:word.istitle()': word.istitle(),
                    f'{p}word:word.isupper()': word.isupper(),
                    f'{p}word:postag': postag,
                    f'{p}word:postag[:2]': postag[:2],
                })

        return features
