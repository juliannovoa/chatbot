import pandas as pd

from src import utils


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
        self.questions = {}
        self.process_data()

    def process_data(self, ):
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
                self.questions[(subj, pred, obj)] = {'n_correct': n_correct,
                                                     'n_incorrect': n_incorrect,
                                                     'mistake_position': mistake_position,
                                                     'right_label': right_label,
                                                     'batch': batch}
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

    def check_question(self, s, p, o) -> dict:
        if (s, p, o) in self.questions:
            question = self.questions[(s, p, o)]
            question['kappa'] = self.kappa_values[question['batch']]
            return question

        return {}

    def find_subject(self, question_p, question_o) -> dict:
        for (s, p, o), v in self.questions.items():
            if question_p == p and question_o == o:
                v['subject'] = s
                return v

        return {}

    def find_object(self, question_p, question_s) -> dict:
        for (s, p, o), v in self.questions.items():
            if question_p == p and question_s == s:
                v['object'] = o
                return v

        return {}
