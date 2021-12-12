from dataclasses import astuple, dataclass
from typing import Optional, MutableMapping

import pandas as pd

from src import utils, Fact


@dataclass
class CrowdQuestion:
    subject: str
    predicate: str
    object: str
    n_correct: int
    n_incorrect: int
    mistake_position: str
    right_label: str
    batch: str
    kappa: float = 1.0


class CrowdWorkers:
    MIN_APPROVAL_RATE = 50
    MIN_WORK_TIME = 10
    DATA_PATH = utils.get_data_path('ATAI_crowd_data.tsv')

    @classmethod
    def filter_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        filtered_data = df[df.LifetimeApprovalRate >= cls.MIN_APPROVAL_RATE]
        filtered_data = filtered_data[filtered_data.WorkTimeInSeconds >= cls.MIN_WORK_TIME]
        filtered_data = filtered_data[filtered_data.FixPosition != filtered_data.FixValue]

        return filtered_data

    def __init__(self):
        self.data = pd.read_csv(self.DATA_PATH.resolve(), sep='\t')
        self.data['LifetimeApprovalRate'] = self.data['LifetimeApprovalRate'].str.rstrip('%').astype('float')
        self.tasks = self.data['HITId'].unique()
        self.workers = self.data['WorkerId'].unique()
        self.filtered_data = self.filter_data(self.data)
        self.kappa_values = {}
        self.questions: MutableMapping[Fact, CrowdQuestion] = {}
        self._process_data()

    def _process_data(self) -> None:
        batches = self.data['HITTypeId'].unique()

        for batch in batches:
            df = self.filtered_data[self.filtered_data.HITTypeId == batch]
            indices = df.groupby(['HITId', 'Input1ID', 'Input2ID', 'Input3ID']).groups

            total_correct = 0
            total_incorrect = 0
            p_i_total = 0
            questions = 0
            for (idx, subj, pred, obj), _ in indices.items():
                n_correct = df[(df.HITId == idx) & (df.AnswerLabel == 'CORRECT')].shape[0]
                n_incorrect = df[(df.HITId == idx) & (df.AnswerLabel == 'INCORRECT')].shape[0]
                mistake_position = ''
                right_label = ''

                for _, row in df[df.HITId == idx].iterrows():
                    row.fillna('', inplace=True)
                    if row.FixPosition != '':
                        mistake_position = row.FixPosition
                    if row.FixValue != '':
                        right_label = row.FixValue

                n_answer = n_correct + n_incorrect
                p_i = 1 / (n_answer * (n_answer - 1)) * (n_correct * (n_correct - 1) + n_incorrect * (n_incorrect - 1))
                self.questions[Fact(subj, pred, obj)] = CrowdQuestion(subject=subj,
                                                                      predicate=pred,
                                                                      object=obj,
                                                                      n_correct=n_correct,
                                                                      n_incorrect=n_incorrect,
                                                                      mistake_position=mistake_position,
                                                                      right_label=right_label,
                                                                      batch=batch)
                total_correct += n_correct
                total_incorrect += n_incorrect
                p_i_total += p_i
                questions += 1

            total_answer = total_correct + total_incorrect
            p_correct = total_correct / total_answer
            p_incorrect = total_incorrect / total_answer
            p_i_mean = p_i_total / questions
            p_e = p_correct ** 2 + p_incorrect ** 2

            self.kappa_values[batch] = (p_i_mean - p_e) / (1 - p_e)

    def check_question(self, subject, predicate, obj) -> Optional[CrowdQuestion]:
        key = Fact(subject, predicate, obj)
        if key not in self.questions:
            return None
        question = self.questions[key]
        question.kappa = self.kappa_values[question.batch]
        return question

    def find_question_one_entity(self, question_predicate, question_entity) -> Optional[CrowdQuestion]:
        for fact, question in self.questions.items():
            subject, predicate, obj = astuple(fact)
            if question_predicate == predicate and (question_entity == obj or question_entity == subject):
                question.kappa = self.kappa_values[question.batch]
                return question
        return None

    def find_question_two_entities(self, question_predicate, question_entity1, question_entity2) -> Optional[
        CrowdQuestion]:
        for fact, question in self.questions.items():
            subject, predicate, obj = astuple(fact)
            if question_predicate == predicate \
                    and ((question_entity1 == obj and question_entity2 == subject)
                         or (question_entity1 == subject and question_entity2 == obj)):
                question.kappa = self.kappa_values[question.batch]
                return question
        return None
