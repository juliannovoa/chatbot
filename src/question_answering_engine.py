import re
from enum import Enum

from src.ner import NameEntityRecognitionModel


class QuestionType(Enum):
    WHO = 1
    UNK = -1


class QuestionSolver:

    def __init__(self):
        self._ner_model = NameEntityRecognitionModel()
        self._questions = {QuestionType.WHO: self.process_who_question,

                           }

    @classmethod
    def get_question_type(cls, question: str) -> QuestionType:
        if 'who ' in question:
            return QuestionType.WHO

        return QuestionType.UNK

    def answer_question(self, question: str) -> str:
        question_type = self.get_question_type(question)
        try:
            relation, entity = self._questions[question_type](question)
        except ValueError:
            return self.generate_excuse()
        # TODO: finish function
        return ''

    @staticmethod
    def process_who_question(question: str) -> tuple[str, str]:
        question_pattern = "(.*)who (is|was|were|has)? (.*(?!of)) (of )?(.*)"
        question = question.lower().rstrip('?')
        match = re.match(question_pattern, question)
        if match is None:
            raise ValueError('Invalid question')
        relation = match.group(3)
        entity = match.group(5)
        return relation, entity

    def generate_excuse(self) -> str:
        # TODO: implement function
        return ''
