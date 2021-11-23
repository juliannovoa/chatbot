import logging
import pickle
import editdistance
import re
from enum import Enum
from pathlib import Path

from rdflib import Graph, Namespace, URIRef

from src import utils
from src.ner import NameEntityRecognitionModel


class QuestionType(Enum):
    WHO = 1
    UNK = -1


class InformationFinder:
    WD = Namespace('http://www.wikidata.org/entity/')
    WDT = Namespace('http://www.wikidata.org/prop/direct/')
    SCHEMA = Namespace('http://schema.org/')
    DDIS = Namespace('http://ddis.ch/atai/')
    RDFS = Namespace('http://www.w3.org/2000/01/rdf-schema#')

    RAW_GRAPH_PATH = utils.get_data_path('14_graph.nt')
    PROCESSED_GRAPH_PATH = utils.get_data_path('ner.model')

    @classmethod
    def node_is_instance(cls, name: str) -> bool:
        return name in cls.WD

    @classmethod
    def node_is_predicate(cls, name: str) -> bool:
        return name in cls.WDT or name in cls.SCHEMA or name in cls.DDIS

    def __init__(self, raw_graph: Path = RAW_GRAPH_PATH,
                 parsed_graph: Path = PROCESSED_GRAPH_PATH) -> None:
        if not parsed_graph.exists():
            logging.info('Graph not available. Start process to parse it.')
            g = Graph()
            g.parse(raw_graph, format='turtle')
            with open(parsed_graph.resolve(), 'wb') as f:
                pickle.dump(g, f)
                f.close()
                logging.info('Graph created.')

        logging.info('Loading graph.')
        with open(parsed_graph.resolve(), 'rb') as f:
            self._g = pickle.load(f)
            f.close()
        logging.info('Graph loaded.')

        self._nodes = {}
        self._predicates = {}

        logging.info('Retrieving nodes and predicates.')
        for node in self._g.all_nodes():
            if isinstance(node, URIRef):
                name = node.toPython()
                if self.node_is_instance(name) and name not in self._nodes:
                    if self._g.value(node, self.RDFS.label):
                        self._nodes[name] = self._g.value(node, self.RDFS.label).toPython()
                    else:
                        self._nodes[name] = re.sub(".*/", "", name)
                elif self.node_is_predicate(name) and name not in self._predicates:
                    if self._g.value(node, self.RDFS.label):
                        self._predicates[name] = self._g.value(node, self.RDFS.label).toPython()
                    else:
                        self._predicates[name] = re.sub(".*/", "", name)

        for _, p, _ in self._g:
            name = p.toPython()
            if self.node_is_predicate(name) and name not in self._predicates:
                self._predicates[name] = re.sub(".*/", "", name)

        logging.info('Nodes and predicates retrieved.')

    def get_closest_item(self, input_instance: str, threshold: int = 500, predicate: bool = False) -> str:
        match_node = []
        logging.debug(f"--- entity matching for \"{input_instance}\"\n")

        if predicate:
            data = self._predicates
        else:
            data = self._nodes

        for k, v in data.items():
            distance = editdistance.eval(v, input_instance)
            if distance < threshold:
                threshold = distance
                match_node = [k]
                logging.debug('New min distance')
                logging.debug(f"edit distance between {v} and {input_instance}: {distance}")
            elif distance == threshold:
                match_node.append(k)
                logging.debug(f"edit distance between {v} and {input_instance}: {distance}")

        logging.debug(f"Entity matched to \"{input_instance} is {match_node}\"\n")
        return match_node


class QuestionSolver:

    def __init__(self):
        self._ner_model = NameEntityRecognitionModel()
        self._questions = {QuestionType.WHO: self.process_who_question,

                           }

    @classmethod
    def get_question_type(cls, question: str) -> QuestionType:
        if 'who ' in question.lower():
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
