from collections import defaultdict

import numpy as np
from pathlib import Path
import pandas as pd
import sklearn_crfsuite
import pickle
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split
import logging
import nltk

from src import utils

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')


class NameEntityRecognitionModel:
    EXPECTED_DATA_COLUMNS = {'lemma', 'next-lemma', 'next-next-lemma', 'next-next-pos', 'next-next-shape',
                             'next-next-word', 'next-pos', 'next-shape', 'next-word', 'pos', 'prev-iob',
                             'prev-lemma', 'prev-pos', 'prev-prev-iob', 'prev-prev-lemma', 'prev-prev-pos',
                             'prev-prev-shape', 'prev-prev-word', 'prev-shape', 'prev-word', 'sentence_idx',
                             'shape', 'word', 'tag'}

    DATA_SET_PATH = utils.get_data_path('ner.csv')
    MODEL_PATH = utils.get_data_path('ner.model')

    @staticmethod
    def load_dataset(dataset: Path, size: int = -1) -> pd.DataFrame:
        if not dataset.is_file():
            raise FileNotFoundError("Dataset not found. Download dataset from "
                                    "https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus")

        df = pd.read_csv(dataset, encoding="ISO-8859-1", on_bad_lines='skip', index_col=0)
        df = df.fillna(method='ffill')

        if size == -1:
            return df

        if size < 0 or size > len(df):
            raise ValueError('Invalid size value')

        return df[:size]

    @staticmethod
    def extract_features(s: pd.Series) -> list:
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

    @staticmethod
    def train_model(dataset: pd.DataFrame) -> sklearn_crfsuite.estimator.CRF:
        x = []
        y = []
        sent_x = []
        sent_y = []
        sent_idx = 1
        for _, row in dataset.iterrows():
            if row['sentence_idx'] == sent_idx:
                sent_x.append(NameEntityRecognitionModel.extract_features(row))
                sent_y.append(row['tag'])
            else:
                x.append(sent_x)
                y.append(sent_y)
                sent_x = []
                sent_y = []
                sent_idx = row['sentence_idx']
                sent_x.append(NameEntityRecognitionModel.extract_features(row))
                sent_y.append(row['tag'])

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

        crf = sklearn_crfsuite.CRF(
            algorithm='l2sgd',  # l2sgd: Stochastic Gradient Descent with L2 regularization term
            max_iterations=1000,  # maximum number of iterations
        )

        crf.fit(x_train, y_train)

        classes = np.unique(y)
        classes = classes.tolist()
        new_classes = classes.copy()
        new_classes.pop()

        y_pred = crf.predict(x_test)

        logging.info("--- performance of the CRF model")
        logging.info(metrics.flat_classification_report(y_test, y_pred))

        return crf

    @staticmethod
    def get_pos_tag(sent: str) -> list:
        tokens = nltk.word_tokenize(sent)
        return nltk.tag.pos_tag(tokens)

    @staticmethod
    def process_input_text(sent: str) -> pd.DataFrame:
        preprocessed_text = NameEntityRecognitionModel.get_pos_tag(sent)
        words = ['__START2__', '__START1__']
        pos_tags = ['__START2__', '__START1__']
        for token, tag in preprocessed_text:
            words.append(token)
            pos_tags.append(tag)
        words.append('__END1__')
        words.append('__END2__')
        pos_tags.append('__END1__')
        pos_tags.append('__END2__')

        input_size = len(preprocessed_text)

        data = {'word': words[2:2 + input_size],
                'prev-prev-word': words[0:input_size],
                'prev-word': words[1:1 + input_size],
                'next-word': words[3:-1],
                'next-next-word': words[4:],
                'pos': pos_tags[2:2 + input_size],
                'prev-prev-pos': pos_tags[0:input_size],
                'prev-pos': pos_tags[1:1 + input_size],
                'next-pos': pos_tags[3:-1],
                'next-next-pos': pos_tags[4:]
                }
        return pd.DataFrame(data=data)

    @classmethod
    def check_data(cls, df: pd.DataFrame) -> bool:
        input_columns_set = set(df.columns.values)
        return input_columns_set == cls.EXPECTED_DATA_COLUMNS

    def __init__(self, dataset: Path = DATA_SET_PATH,
                 model_path: Path = MODEL_PATH) -> None:

        if not model_path.exists():
            logging.info('NER is not available. Start process to create it.')
            logging.info('Loading dataset.')
            df = NameEntityRecognitionModel.load_dataset(dataset.resolve())
            logging.info('Dataset loaded.')
            if NameEntityRecognitionModel.check_data(df):
                logging.info('Dataset structure is valid.')
                logging.info('Start training model.')
                model = NameEntityRecognitionModel.train_model(df)
                with open(model_path.resolve(), 'wb') as f:
                    pickle.dump(model, f)
                    f.close()
                logging.info('Model saved.')
            else:
                raise ValueError('Invalid dataset')
        with open(model_path.resolve(), 'rb') as f:
            self._ner_model = pickle.load(f)
            f.close()
        logging.info('Model loaded')

    def find_name_entities(self, sentence: str) -> defaultdict:
        data = NameEntityRecognitionModel.process_input_text(sentence)
        input_model = [[NameEntityRecognitionModel.extract_features(row) for _, row in data.iterrows()]]
        output_model = self._ner_model.predict(input_model)
        output_model = [item for sublist in output_model for item in sublist]
        entity = []
        entities = defaultdict(list)
        tag = ''

        for idx, val in enumerate(output_model):
            if val[:2] == 'B-':
                if len(entity) != 0:
                    entities[tag].append(' '.join(entity))
                    tag = ''
                    entity = []
                tag = val[2:]
                entity.append(data.iloc[idx]['word'])
            elif val[:2] == 'I-':
                entity.append(data.iloc[idx]['word'])
            elif len(entity) != 0:
                entities[tag].append(' '.join(entity))
                tag = ''
                entity = []

        if len(entity) != 0:
            entities[tag].append(' '.join(entity))
            tag = ''
            entity = []

        return entities
