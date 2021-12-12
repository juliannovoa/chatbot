import logging
import pickle
import re
from typing import Any, Mapping, Tuple, List

import editdistance
import rdflib
from pathlib import Path
from rdflib import Graph, Namespace, URIRef
from sentence_transformers import SentenceTransformer, util

from src import utils
from src.crowdsourcing import CrowdWorkers


class KnowledgeGraph:
    WD = Namespace('http://www.wikidata.org/entity/')
    WDT = Namespace('http://www.wikidata.org/prop/direct/')
    SCHEMA = Namespace('http://schema.org/')
    DDIS = Namespace('http://ddis.ch/atai/')
    RDFS = Namespace('http://www.w3.org/2000/01/rdf-schema#')

    RAW_GRAPH_PATH = utils.get_data_path('14_graph.nt')
    PROCESSED_GRAPH_PATH = utils.get_model_path('graph.g')
    ENTITIES_PATH = utils.get_model_path('entities.pickle')
    PREDICATES_PATH = utils.get_model_path('predicates.pickle')

    SENTENCE_EMBEDDINGS_MODEL = 'all-MiniLM-L6-v2'

    @classmethod
    def element_is_entity(cls, element) -> bool:
        return isinstance(element, str) and cls.WD in element

    @classmethod
    def element_is_predicate(cls, element: str) -> bool:
        return cls.WDT in element or cls.SCHEMA in element or cls.DDIS in element

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

        # TODO: sacar de aquí.
        logging.info('Loading crowd workers.')
        # self._crowd_workers = CrowdWorkers()
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
            self._g = pickle.load(f)
        logging.debug('Knowledge graph loaded.')

        if not parsed_entities.exists() or not parsed_predicates.exists():
            entities, predicates = self._parse_entities_and_predicates()
            logging.debug('Saving entities and predicates.')
            with open(parsed_entities.resolve(), 'wb') as f:
                pickle.dump(entities, f)
                del entities
            with open(parsed_predicates.resolve(), 'wb') as f:
                pickle.dump(predicates, f)
                del predicates
            logging.debug('Entities and predicates saved.')

        logging.info('Loading entities and predicates.')
        with open(parsed_entities.resolve(), 'rb') as f:
            self._entities = pickle.load(f)
        with open(parsed_predicates.resolve(), 'rb') as f:
            self._predicates = pickle.load(f)
        logging.debug('Entities and predicates loaded.')

    def _parse_entities_and_predicates(self) -> Tuple[Mapping, Mapping]:
        logging.info('Retrieving entities and predicates.')
        entities = {}
        predicates = {}
        entity_labels = []
        entity_names = []
        predicate_labels = []
        predicate_names = []
        for node in self._g.all_nodes():
            if not isinstance(node, URIRef):
                continue
            name = node.toPython()
            if self.element_is_entity(name) and name not in entities:
                entity_names.append(name)
                if self._g.value(node, self.RDFS.label):
                    label = self._g.value(node, self.RDFS.label).toPython()
                else:
                    label = re.sub(".*/", "", name)
                # TODO: shortname se puede calcular a partir de name
                entities[name] = {'label': label,
                                  'short_name': self.get_short_element_name(node)}
                entity_labels.append(label)
            elif self.element_is_predicate(name) and name not in predicates:
                predicate_names.append(name)
                if self._g.value(node, self.RDFS.label):
                    label = self._g.value(node, self.RDFS.label).toPython()
                else:
                    label = re.sub(".*/", "", name)
                # TODO: shortname se puede calcular a partir de name
                predicates[name] = {'label': label,
                                    'short_name': self.get_short_element_name(node)}
                predicate_labels.append(label)

        for _, predicate, _ in self._g:
            name = predicate.toPython()
            if self.element_is_predicate(name) and name not in predicates:
                predicate_names.append(name)
                label = re.sub(".*/", "", name)
                predicates[name] = {'label': label,
                                    'short_name': self.get_short_element_name(predicate)}
                predicate_labels.append(label)

        logging.debug('Computing entity embeddings.')
        entity_embeddings = self._sentence_embedding.encode(entity_labels, convert_to_tensor=True)
        for name, embedding in zip(entity_names, entity_embeddings):
            entities[name]['embedding'] = embedding

        logging.debug('Computing predicate embeddings.')
        predicate_embeddings = self._sentence_embedding.encode(predicate_labels, convert_to_tensor=True)
        for name, embedding in zip(predicate_names, predicate_embeddings):
            predicates[name]['embedding'] = embedding

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
        return self._g.query(query)

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
    def get_crowd_information_object(self, predicates: List[str], entities: List[str]) -> Mapping[str, Any]:
        for predicate in predicates:
            short_predicate = self._predicates[predicate]['short_name']
            for entity in entities:
                short_entity = self._entities[entity]['short_name']
                if data := self._crowd_workers.find_question(short_predicate, short_entity):
                    return data
        return {}
