import logging
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Tuple, List, Set

import editdistance
import numpy as np
import pandas as pd
import rdflib
from pathlib import Path
from rdflib import Graph, Namespace, URIRef
from sentence_transformers import SentenceTransformer, util

from src import utils
from src.crowdsourcing import CrowdWorkers


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path.resolve(), index_col=0,
                       converters={'sentence_embedding': lambda s: np.fromstring(s[1:-1], sep=', ')})


@dataclass
class Fact(object):
    subject: str
    predicate: str
    object: str


class KnowledgeGraph:
    WD = Namespace('http://www.wikidata.org/entity/')
    WDT = Namespace('http://www.wikidata.org/prop/direct/')
    SCHEMA = Namespace('http://schema.org/')
    DDIS = Namespace('http://ddis.ch/atai/')
    RDFS = Namespace('http://www.w3.org/2000/01/rdf-schema#')

    SHORT_PREFIX = {
        WDT: 'wdt:',
        SCHEMA: 'schema:',
        DDIS: 'ddis:',
        WD: 'wd:'
    }

    RAW_GRAPH_PATH = utils.get_data_path('14_graph.nt')
    PROCESSED_GRAPH_PATH = utils.get_model_path('graph.g')
    ENTITIES_PATH = utils.get_model_path('entities.csv')
    PREDICATES_PATH = utils.get_model_path('predicates.csv')

    SENTENCE_EMBEDDINGS_MODEL = 'all-MiniLM-L6-v2'

    @classmethod
    def element_is_entity(cls, element) -> bool:
        return isinstance(element, str) and element.startswith(cls.SHORT_PREFIX[cls.WD])

    @classmethod
    def element_is_predicate(cls, element: str) -> bool:
        return element.startswith((cls.SHORT_PREFIX[cls.WDT], cls.SHORT_PREFIX[cls.SCHEMA], cls.SHORT_PREFIX[cls.DDIS]))

    @classmethod
    def get_short_element_name(cls, node: str) -> str:
        prefix = ''
        if node in cls.WDT:
            prefix = 'wdt:'
        elif node in cls.SCHEMA:
            prefix = 'schema:'
        elif node in cls.DDIS:
            prefix = 'ddis:'
        elif node in cls.WD:
            prefix = 'wd:'
        return re.sub(".*/", prefix, node)

    @classmethod
    def extend_element_name(cls, name: str) -> str:
        if 'wdt:' in name:
            return re.sub('wdt:', cls.WDT, name)
        elif 'schema:' in name:
            return re.sub("schema:", cls.SCHEMA, name)
        elif 'ddis:' in name:
            return re.sub("ddis:", cls.DDIS, name)
        elif 'wd:' in name:
            return re.sub("wd:", cls.WD, name)
        else:
            return name

    def __init__(self, raw_graph: Path = RAW_GRAPH_PATH,
                 parsed_graph: Path = PROCESSED_GRAPH_PATH,
                 parsed_entities: Path = ENTITIES_PATH,
                 parsed_predicates: Path = PREDICATES_PATH) -> None:
        logging.info('Loading crowd workers.')
        self._crowd_workers = CrowdWorkers()
        logging.debug('Crowd workers loaded.')

        logging.info('Loading SentenceTransformer model.')
        self._sentence_embedding = SentenceTransformer(self.SENTENCE_EMBEDDINGS_MODEL)
        logging.debug('Sentence-embeddings model loaded.')

        if not parsed_graph.exists():
            logging.info('Knowledge graph not available. Start process to parse it.')
            g = Graph()
            g.parse(raw_graph, format='turtle')
            with open(parsed_graph.resolve(), 'wb') as f:
                logging.debug('Saving knowledge graph.')
                pickle.dump(g, f)
                del g
                logging.debug('Knowledge graph saved.')

        logging.info('Loading knowledge graph.')
        with open(parsed_graph.resolve(), 'rb') as f:
            self._kg = pickle.load(f)
        logging.debug(f'Knowledge graph loaded.')

        if not parsed_entities.exists() or not parsed_predicates.exists():
            entities, predicates = self._parse_entities_and_predicates
            logging.debug('Saving predicates.')
            predicates.to_csv(parsed_predicates.resolve())
            del predicates
            logging.debug('Saving entities.')
            entities.to_csv(parsed_entities.resolve())
            del entities
            logging.debug('Entities and predicates saved.')

        logging.debug('Loading predicates.')
        self._predicates = _read_csv(parsed_predicates)
        logging.debug('Loading entities.')
        self._entities = _read_csv(parsed_entities)
        logging.debug('Entities predicates loaded.')

    @property
    def _parse_entities_and_predicates(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logging.info('Computing entities and predicates.')
        entity_labels: List[str] = []
        entity_names: List[str] = []
        entity_name_set: Set[str] = set()
        predicate_labels: List[str] = []
        predicate_names: List[str] = []
        predicate_name_set: Set[str] = set()
        for node in self._kg.all_nodes():
            if not isinstance(node, URIRef):
                continue
            name = self.get_short_element_name(node.toPython())
            if self.element_is_entity(name) and name not in entity_name_set:
                if self._kg.value(node, self.RDFS.label):
                    label = self._kg.value(node, self.RDFS.label).toPython()
                else:
                    label = re.sub(".*/", "", name)
                entity_names.append(name)
                entity_name_set.add(name)
                entity_labels.append(label)
            elif self.element_is_predicate(name) and name not in predicate_name_set:
                if self._kg.value(node, self.RDFS.label):
                    label = self._kg.value(node, self.RDFS.label).toPython()
                else:
                    label = re.sub(".*/", "", name)
                predicate_names.append(name)
                predicate_name_set.add(name)
                predicate_labels.append(label)

        for _, predicate, _ in self._kg:
            name = self.get_short_element_name(predicate.toPython())
            if self.element_is_predicate(name) and name not in predicate_name_set:
                label = re.sub(".*/", "", name)
                predicate_names.append(name)
                predicate_name_set.add(name)
                predicate_labels.append(label)

        logging.debug('Computing predicate embeddings.')
        predicate_embeddings = [emb.tolist() for emb in
                                self._sentence_embedding.encode(predicate_labels, convert_to_numpy=False,
                                                                normalize_embeddings=True)]
        logging.debug('Create predicate dataframe.')
        predicates = pd.DataFrame({'name': predicate_names,
                                   'label': predicate_labels,
                                   'sentence_embedding': predicate_embeddings,
                                   'embedding': None}
                                  ).set_index('name')

        logging.debug('Computing entity embeddings.')
        entity_embeddings = [emb.tolist() for emb in
                             self._sentence_embedding.encode(entity_labels, convert_to_numpy=False,
                                                             normalize_embeddings=True)]
        logging.debug('Create entity dataframe.')
        entities = pd.DataFrame({'name': entity_names,
                                 'label': entity_labels,
                                 'sentence_embedding': entity_embeddings,
                                 'embedding': None}
                                ).set_index('name')

        logging.debug('Entities and predicates retrieved.')
        return entities, predicates

    def get_closest_node(self, query: str, embedding_threshold: float = 0.5,
                         edit_distance_threshold: int = 100, predicate: bool = False) -> List[str]:
        logging.debug(f'--- entity matching for "{query}"')
        if predicate:
            data = self._predicates
        else:
            data = self._entities
            edit_distance_threshold = 10
            embedding_threshold = 0.25
        query_embedding = self._sentence_embedding.encode(query, convert_to_tensor=True)
        matches = []
        for name, node in data.items():
            if editdistance.eval(query, node['description']) > edit_distance_threshold:
                continue
            similarity = util.pytorch_cos_sim(query_embedding, node['embedding'])
            if similarity > embedding_threshold:
                embedding_threshold = similarity
                matches = [name]
                logging.debug('New max similarity')
                logging.debug(f"edit distance between {node['description']} and {query}: {similarity}")
            elif similarity == embedding_threshold:
                matches.append(name)
                logging.debug(f"edit distance between {node['description']} and {query}: {similarity}")
        logging.debug(f'Entities matched to "{query}" are {matches}')
        return matches

    def query(self, query: str) -> rdflib.query.Result:
        return self._kg.query(query)

    def get_node_label(self, node_name: str, is_predicate: bool = False, short_name: bool = False) -> str:
        data = self._predicates if is_predicate else self._entities
        key = self.extend_element_name(node_name) if short_name else node_name
        logging.debug(f'Get node label of {key}({type(key)})')
        if not is_predicate and not self.element_is_entity(key):
            # key is a literal
            return key
        return data[key]['label']

    # def get_node_description(self, key: str) -> str:
    #     logging.debug(f'Get node description of {key}({type(key)})')
    #     if self.element_is_entity(key):
    #         return self._entities[key]['description']
    #     else:
    #         return key
    #
    # def get_node_description_by_short_name(self, key: str) -> str:
    #     logging.debug(f'Get node description of {key}({type(key)})')
    #     long_name = self.extend_element_name(key)
    #     if self.element_is_entity(long_name):
    #         return self._entities[long_name]['description']
    #     else:
    #         return long_name

    # TODO: mover.
    def find_crowd_question(self, predicates: List[str], entities: List[str]) -> Mapping[str, Any]:
        for predicate in predicates:
            short_predicate = self._predicates[predicate]['short_name']
            for entity in entities:
                short_entity = self._entities[entity]['short_name']
                if data := self._crowd_workers.find_question(short_predicate, short_entity):
                    return data
        return {}

    def _process_query_search(self, query: str, predicates: List[str], entities: List[str], early_stop:bool=False) -> List[Fact]:
        facts = []
        for predicate in predicates:
            for entity in entities:
                logging.debug(f'Query elements: {predicate}(P:{type(predicate)}) and {entity}(E:{type(entity)})')
                formatted_query = query.format(e=entity, p=predicate)
                for row in self._knowledge_graph.query(formatted_query):
                    obj = row.x.toPython()
                    try:
                        subj = row.y.toPython()
                    except AttributeError:
                        subj = entity
                    facts.append(Fact(subj, predicate, obj)
                if facts and early_stop:
                    return facts
        return facts