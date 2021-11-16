import os

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
                f.write('0,1,2\n')

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
