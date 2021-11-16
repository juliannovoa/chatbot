import os

from pathlib import Path
from unittest import TestCase

from src.ner import NameEntityRecognitionModel


class TestNameEntityRecognitionModel(TestCase):
    MOCK_FILENAME = 'mock.csv'

    @classmethod
    def create_csv(cls, lines: int):
        with open(TestNameEntityRecognitionModel.MOCK_FILENAME, 'w') as f:
            f.write('v1,v2,v3\n')
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
