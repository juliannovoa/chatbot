import logging
import pickle
from collections import defaultdict

import editdistance
import re
from enum import Enum

import rdflib
from pathlib import Path

from rdflib import Graph, Namespace, URIRef

from src import utils
from src.ner import NameEntityRecognitionModel


class QuestionType(Enum):
    WHO_OF = '(.*)who (is|are|was|were) (the )?(((?!of).)*) of (the movie )?(.*)'
    UNK = -1


class InformationFinder:
    WD = Namespace('http://www.wikidata.org/entity/')
    WDT = Namespace('http://www.wikidata.org/prop/direct/')
    SCHEMA = Namespace('http://schema.org/')
    DDIS = Namespace('http://ddis.ch/atai/')
    RDFS = Namespace('http://www.w3.org/2000/01/rdf-schema#')

    RAW_GRAPH_PATH = utils.get_data_path('14_graph.nt')
    PROCESSED_GRAPH_PATH = utils.get_model_path('graph.g')

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

    def get_closest_item(self, input_instance: str, threshold: int = 20, predicate: bool = False) -> list:
        match_node = []
        logging.debug(f"--- entity matching for \"{input_instance}\"")

        if predicate:
            data = self._predicates
        else:
            data = self._nodes

        for k, v in data.items():
            distance = editdistance.eval(v.lower(), input_instance.lower())
            if distance < threshold:
                threshold = distance
                match_node = [k]
                logging.debug('New min distance')
                logging.debug(f"edit distance between {v} and {input_instance}: {distance}")
            elif distance == threshold:
                match_node.append(k)
                logging.debug(f"edit distance between {v} and {input_instance}: {distance}")

        logging.debug(f"Entities matched to \"{input_instance}\" is {match_node}")
        return match_node

    def query(self, query: str) -> rdflib.query.Result:
        return self._g.query(query)

    def get_predicate_description(self, key: str) -> str:
        logging.debug(f'Get predicate description of {key}({type(key)})')
        return self._predicates[key]

    def get_node_description(self, key: str) -> str:
        logging.debug(f'Get node description of {key}({type(key)})')
        return self._nodes[key]


class QuestionSolver:

    def __init__(self):
        self._ner_model = NameEntityRecognitionModel()
        self._information_finder = InformationFinder()
        self._questions = {QuestionType.WHO_OF: self.process_who_of_question,

                           }

    @classmethod
    def get_question_type(cls, question: str) -> QuestionType:
        question = question.lower().rstrip('?')
        if re.match(QuestionType.WHO_OF.value, question):
            return QuestionType.WHO_OF

        return QuestionType.UNK

    def answer_question(self, question: str) -> str:
        question_type = self.get_question_type(question)
        try:
            return self._questions[question_type](question)
        except ValueError:
            return self.generate_excuse()

    def process_who_of_question(self, question: str) -> str:
        question = question.lower().rstrip('?')
        match = re.match(QuestionType.WHO_OF.value, question)
        if match is None:
            raise ValueError('Invalid question')
        predicates = self._information_finder.get_closest_item(match.group(4), predicate=True)
        entities = self._information_finder.get_closest_item(match.group(7))
        logging.debug(f'Entities: {entities}')
        logging.debug(f'Predicates: {predicates}')
        query = '''
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX wd: <http://www.wikidata.org/entity/>
            SELECT DISTINCT ?x WHERE {{ 
                <{e}> <{p}> ?x .
                ?x wdt:P31 wd:Q5. 
            }}
        '''
        information = self.process_query(query, predicates, entities)

        if len(information) == 0:
            logging.debug('Information not found. Trying second degree search')
            query = '''
                PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                PREFIX wd: <http://www.wikidata.org/entity/>
                SELECT DISTINCT ?x ?y WHERE {{
                    <{e}> ?a ?y . 
                    ?y <{p}> ?x .
                    ?x wdt:P31 wd:Q5.  
                }}
            '''
            information = self.process_query(query, predicates, entities, True)
        if len(information) == 0:
            return self.generate_excuse()
        else:
            return self.process_response(information)

    def process_query(self, query: str, predicates: list, entities: list, early_stop=False) -> defaultdict[set]:
        information = defaultdict(set)
        logging.debug('Start query.')
        logging.debug(f'Entities: {entities}')
        logging.debug(f'Predicates: {predicates}')
        for predicate in predicates:
            for entity in entities:
                logging.debug(f'Query elements: {predicate}(P:{type(predicate)}) and {entity}(E:{type(entity)})')
                formatted_query = query.format(e=entity, p=predicate)
                for row in self._information_finder.query(formatted_query):
                    subj = row.x.toPython()
                    try:
                        obj = row.y.toPython()
                    except AttributeError:
                        obj = entity
                    information[(obj, predicate)].add(subj)
                if len(information) >= 1 and early_stop:
                    return information

        return information

    def process_response(self, information: defaultdict) -> str:
        output = []
        for (o, p), s in information.items():
            predicate = self._information_finder.get_predicate_description(p)
            obj = self._information_finder.get_node_description(o)
            description = self.get_film_description(o)
            if len(s) == 1:
                for it in s:
                    item = self._information_finder.get_node_description(it)
                    output.append(f'{item} is the {predicate} of {obj}{description}')
            else:
                output.append(f'The {predicate}s of {obj}{description} are:')
                for it in s:
                    item = self._information_finder.get_node_description(it)
                    output.append(f'   {item}')

        return '\n'.join(output)

    def get_film_description(self, film: str) -> str:
        query = f'''
                    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                    SELECT DISTINCT ?x WHERE {{
                        <{film}> wdt:P577 ?x .
                    }}
                '''
        logging.debug('Start description query.')

        for row in self._information_finder.query(query):
            date = row.x.toPython()
            if isinstance(date, int):
                return f'({date})'
            else:
                return f'({date.year})'

        query = f'''
                    PREFIX ns2: <http://schema.org/>
                    SELECT DISTINCT ?x WHERE {{
                        <{film}> ns2:description ?x . 
                    }}
                '''
        for row in self._information_finder.query(query):
            return f'({row.x.toPython()})'

        return ''

    def generate_excuse(self) -> str:
        # TODO: implement function
        return "I don't know"