import os
import pandas as pd

from pathlib import Path
from unittest import TestCase

from src.ner import NameEntityRecognitionModel


class TestNameEntityRecognitionModel(TestCase):
    MOCK_FILENAME = 'mock.csv'

    @classmethod
    def create_csv(cls, lines: int):
        with open(TestNameEntityRecognitionModel.MOCK_FILENAME, 'w') as f:
            f.write(','.join(NameEntityRecognitionModel.EXPECTED_DATA_COLUMNS))
            f.write('\n')
            for line in range(lines):
                f.write(f'{lines},thousand,of,demonstr,NNS,lowercase,demonstrators,IN,lowercase,Of,NNS,__START1__,'
                        f'__start1__,ABC,__START2__,__start2__,__START2__,wildcard,__START2__,wildcard,'
                        f'HI,1,capitalized,2005,O\n')

    @classmethod
    def delete_csv(cls):
        os.remove(TestNameEntityRecognitionModel.MOCK_FILENAME)

    def test_load_dataset_file_does_not_exists(self):
        with self.assertRaises(FileNotFoundError):
            file = Path('./no_file.file')
            NameEntityRecognitionModel.load_dataset(file)

    def test_load_dataset_file(self):
        lines = 3

        TestNameEntityRecognitionModel.create_csv(lines)
        file = Path(TestNameEntityRecognitionModel.MOCK_FILENAME)
        df = NameEntityRecognitionModel.load_dataset(file)
        self.assertEqual(len(df), lines)
        TestNameEntityRecognitionModel.delete_csv()

    def test_load_dataset_file_with_negative_size(self):
        lines = 3

        TestNameEntityRecognitionModel.create_csv(lines)
        file = Path(TestNameEntityRecognitionModel.MOCK_FILENAME)
        with self.assertRaises(ValueError):
            NameEntityRecognitionModel.load_dataset(file, -2)
        TestNameEntityRecognitionModel.delete_csv()

    def test_load_dataset_file_with_excessive_size(self):
        lines = 3

        TestNameEntityRecognitionModel.create_csv(lines)
        file = Path(TestNameEntityRecognitionModel.MOCK_FILENAME)
        with self.assertRaises(ValueError):
            NameEntityRecognitionModel.load_dataset(file, 4)
        TestNameEntityRecognitionModel.delete_csv()

    def test_load_dataset_file_with_size(self):
        lines = 3
        desired_lines = 2

        TestNameEntityRecognitionModel.create_csv(lines)
        file = Path(TestNameEntityRecognitionModel.MOCK_FILENAME)
        df = NameEntityRecognitionModel.load_dataset(file, desired_lines)
        self.assertEqual(len(df), desired_lines)
        os.remove(TestNameEntityRecognitionModel.MOCK_FILENAME)

    def test_check_data_correct(self):
        lines = 1

        TestNameEntityRecognitionModel.create_csv(lines)
        file = Path(TestNameEntityRecognitionModel.MOCK_FILENAME)
        df = NameEntityRecognitionModel.load_dataset(file)
        self.assertTrue(NameEntityRecognitionModel.check_data(df))
        os.remove(TestNameEntityRecognitionModel.MOCK_FILENAME)

    def test_check_data_less_columns(self):
        lines = 1

        TestNameEntityRecognitionModel.create_csv(lines)
        file = Path(TestNameEntityRecognitionModel.MOCK_FILENAME)
        df = NameEntityRecognitionModel.load_dataset(file)
        df.drop(df.columns[1], axis=1, inplace=True)
        self.assertFalse(NameEntityRecognitionModel.check_data(df))
        os.remove(TestNameEntityRecognitionModel.MOCK_FILENAME)

    def test_check_data_more_columns(self):
        lines = 1

        TestNameEntityRecognitionModel.create_csv(lines)
        file = Path(TestNameEntityRecognitionModel.MOCK_FILENAME)
        df = NameEntityRecognitionModel.load_dataset(file)
        self.assertFalse(NameEntityRecognitionModel.check_data(df.assign(new=[1])))
        os.remove(TestNameEntityRecognitionModel.MOCK_FILENAME)

    def test_extract_features(self):
        expected_features = {
            'word.lower()': '2005',
            'word[-3:]': '005',
            'word[-2:]': '05',
            'word.isupper()': False,
            'word.istitle()': False,
            'word.isdigit()': True,
            'postag': 'NNS',
            'postag[:2]': 'NN',
            'prev-prev-word:word.lower()': '__start2__',
            'prev-prev-word:word.istitle()': False,
            'prev-prev-word:word.isupper()': False,
            'prev-prev-word:postag': '__START2__',
            'prev-prev-word:postag[:2]': '__START2__',
            'prev-word:word.lower()': 'hi',
            'prev-word:word.istitle()': False,
            'prev-word:word.isupper()': True,
            'prev-word:postag': 'ABC',
            'prev-word:postag[:2]': 'AB',
            'next-word:word.lower()': 'of',
            'next-word:word.istitle()': True,
            'next-word:word.isupper()': False,
            'next-word:postag': 'IN',
            'next-word:postag[:2]': 'IN',
            'next-next-word:word.lower()': '__end2__',
            'next-next-word:word.istitle()': False,
            'next-next-word:word.isupper()': False,
            'next-next-word:postag': '__END2__',
            'next-next-word:postag[:2]': '__END2__',

        }
        s = pd.Series(data={'word': '2005',
                            'prev-prev-word': '__START2__',
                            'prev-word': 'HI',
                            'next-word': 'Of',
                            'next-next-word': '__END2__',
                            'pos': 'NNS',
                            'prev-prev-pos': '__START2__',
                            'prev-pos': 'ABC',
                            'next-pos': 'IN',
                            'next-next-pos': '__END2__'})

        features = NameEntityRecognitionModel.extract_features(s)
        self.assertDictEqual(features, expected_features)
