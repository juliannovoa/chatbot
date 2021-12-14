import logging
import pickle
import re
from collections import defaultdict
from typing import Tuple, List, Set, Optional, Mapping

import editdistance
import pandas as pd
from pathlib import Path
from rdflib import Graph, Namespace, URIRef
from sentence_transformers import SentenceTransformer

from src import utils, Fact
from src.crowdsourcing import CrowdWorkers, CrowdQuestion
from src.utils import ImageEntity, read_csv


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
    SENTENCE_TRANSFORMER_PATH = utils.get_model_path('all-MiniLM-L6-v2')

    SENTENCE_EMBEDDINGS_MODEL = 'all-MiniLM-L6-v2'

    @classmethod
    def element_is_entity(cls, element) -> bool:
        return isinstance(element, str) and (element.startswith(cls.SHORT_PREFIX[cls.WD]) or element in cls.WD)

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
        self._sentence_embedding = SentenceTransformer(self.SENTENCE_TRANSFORMER_PATH.resolve())
        logging.info('Sentence-embeddings model loaded.')

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
        logging.info(f'Knowledge graph loaded.')

        if not parsed_entities.exists() or not parsed_predicates.exists():
            entities, predicates = self._parse_entities_and_predicates()
            logging.debug('Saving predicates.')
            predicates.to_csv(parsed_predicates.resolve())
            del predicates
            logging.debug('Saving entities.')
            entities.to_csv(parsed_entities.resolve())
            del entities
            logging.debug('Entities and predicates saved.')

        logging.info('Loading predicates.')
        self._predicates = read_csv(parsed_predicates)
        logging.info('Loading entities.')
        self._entities = read_csv(parsed_entities)
        logging.info('Entities predicates loaded.')

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
                                   'sentence_embedding': predicate_embeddings}
                                  ).set_index('name')

        logging.debug('Computing entity embeddings.')
        entity_embeddings = [emb.tolist() for emb in
                             self._sentence_embedding.encode(entity_labels, convert_to_numpy=False,
                                                             normalize_embeddings=True)]
        logging.debug('Create entity dataframe.')
        entities = pd.DataFrame({'name': entity_names,
                                 'label': entity_labels,
                                 'sentence_embedding': entity_embeddings}
                                ).set_index('name')

        logging.debug('Entities and predicates retrieved.')
        return entities, predicates

    def find_closest_node(self, query: str, embedding_threshold: float = 0.35,
                          edit_distance_threshold: int = 100, predicate: bool = False) -> List[str]:
        logging.debug(f'--- entity matching for "{query}"')
        if predicate:
            df = self._predicates
        else:
            df = self._entities
            edit_distance_threshold = 10
            embedding_threshold = 0.25
        query_embedding = self._sentence_embedding.encode(query)
        # Filter candidates by edit distance.
        candidates = df[df.label.apply(lambda x: int(editdistance.eval(query, x)) <= edit_distance_threshold)]
        # Select candidates by cosine similarity (= dot product when vectors are normalized).
        sim = candidates['sentence_embedding'].apply(lambda x: query_embedding.dot(x))
        max_sim = sim.values.max()
        if max_sim < embedding_threshold:
            logging.debug('Similarity too low')
            return []
        matches = candidates[sim.values == max_sim].index.tolist()
        # for name, node in df.items():
        #     if editdistance.eval(query, node['description']) > edit_distance_threshold:
        #         continue
        #     similarity = util.pytorch_cos_sim(query_embedding, node['embedding'])
        #     if similarity > embedding_threshold:
        #         embedding_threshold = similarity
        #         matches = [name]
        #         logging.debug('New max similarity')
        #         logging.debug(f"edit distance between {node['description']} and {query}: {similarity}")
        #     elif similarity == embedding_threshold:
        #         matches.append(name)
        #         logging.debug(f"edit distance between {node['description']} and {query}: {similarity}")
        logging.debug(f'{"Predicates" if predicate else "Entities"} matched to "{query}" are {matches}')
        return matches

    def get_node_label(self, name: str, is_predicate: bool = False) -> str:
        data = self._predicates if is_predicate else self._entities
        if not is_predicate and not self.element_is_entity(name):
            # key is a literal
            return name
        return data.loc[name]['label']

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

    def find_crowd_question(self, predicates: List[str], entities: List[str]) -> Optional[CrowdQuestion]:
        for predicate in predicates:
            for entity in entities:
                if data := self._crowd_workers.find_question_one_entity(predicate, entity):
                    return data
        return None

    def find_crowd_question_two_entities(self, predicates: List[str], entities1: List[str], entities2: List[str]) -> \
            Optional[CrowdQuestion]:
        for predicate in predicates:
            for entity1 in entities1:
                for entity2 in entities2:
                    if data := self._crowd_workers.find_question_two_entities(predicate, entity1, entity2):
                        return data
        return None

    def find_facts(self, predicates: List[str], entities: List[str]) -> List[Fact]:
        logging.debug(f'Query: predicates={predicates} and entities={entities})')
        facts = []
        obj_query = '''
                        PREFIX ddis: <http://ddis.ch/atai/>
                        PREFIX wd: <http://www.wikidata.org/entity/>
                        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                        PREFIX schema: <http://schema.org/>
                        SELECT DISTINCT ?x WHERE {{
                            {e} {p} ?x .
                        }}
                    '''
        for predicate in predicates:
            for entity in entities:
                formatted_query = obj_query.format(e=entity, p=predicate)
                for row in self._kg.query(formatted_query):
                    obj = row.x.toPython()
                    if self.element_is_entity(obj):
                        obj = self.get_short_element_name(obj)
                    facts.append(Fact(entity, predicate, obj))

        subject_query = '''
                            PREFIX ddis: <http://ddis.ch/atai/>
                            PREFIX wd: <http://www.wikidata.org/entity/>
                            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                            PREFIX schema: <http://schema.org/>
                            SELECT DISTINCT ?x WHERE {{
                                ?x {p} {e} .
                            }}
                        '''
        for predicate in predicates:
            for entity in entities:
                formatted_query = subject_query.format(e=entity, p=predicate)
                for row in self._kg.query(formatted_query):
                    subject = row.x.toPython()
                    if self.element_is_entity(subject):
                        subject = self.get_short_element_name(subject)
                    facts.append(Fact(subject, predicate, entity))
        return facts

    def find_facts_two_entities(self, predicates: List[str], entities1: List[str], entities2: List[str]) -> List[Fact]:
        facts = []
        query = '''
                    PREFIX ddis: <http://ddis.ch/atai/>
                    PREFIX wd: <http://www.wikidata.org/entity/>
                    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                    PREFIX schema: <http://schema.org/>
                    ASK {{
                        {s} {p} {o} .
                    }}
                '''
        for predicate in predicates:
            for entity1 in entities1:
                for entity2 in entities2:
                    logging.debug(f'Query elements: {predicate} {entity1}) and {entity2})')
                    if self._kg.query(query.format(s=entity1, p=predicate, o=entity2)):
                        facts.append(Fact(entity1, predicate, entity2))
                    if self._kg.query(query.format(s=entity2, p=predicate, o=entity1)):
                        facts.append(Fact(entity2, predicate, entity1))
        return facts

    def get_film_description(self, film: str) -> str:
        logging.debug(f'get_film_description for {film}')
        query = f'''
                    PREFIX ddis: <http://ddis.ch/atai/>
                    PREFIX wd: <http://www.wikidata.org/entity/>
                    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                    PREFIX schema: <http://schema.org/>
                    SELECT DISTINCT ?x WHERE {{
                        {film} wdt:P577 ?x .
                    }}
                '''
        for row in self._kg.query(query):
            date = row.x.toPython()
            if isinstance(date, int):
                return f'({date})'
            else:
                return f'({date.year})'
        query = f'''
                    PREFIX ddis: <http://ddis.ch/atai/>
                    PREFIX wd: <http://www.wikidata.org/entity/>
                    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                    PREFIX schema: <http://schema.org/>
                    PREFIX ns2: <http://schema.org/>
                    SELECT DISTINCT ?x WHERE {{
                        {film} ns2:description ?x . 
                    }}
                '''
        for row in self._kg.query(query):
            return f'({row.x.toPython()})'
        return ''

    def find_imdb_ids(self, entities: List[str]) -> Mapping[ImageEntity, List[str]]:
        logging.debug(f'Looking for imdb ids for {entities}')
        query = '''
                    PREFIX ddis: <http://ddis.ch/atai/>
                    PREFIX wd: <http://www.wikidata.org/entity/>
                    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                    PREFIX schema: <http://schema.org/>
                    SELECT DISTINCT ?x WHERE {{
                        {e} wdt:P345 ?x
                    }}
                '''
        imdb_ids = defaultdict(list[str])
        for entity in entities:
            for row in self._kg.query(query.format(e=entity)):
                imdb_id = row.x.toPython()
                if imdb_id.startswith('tt'):
                    imdb_ids[ImageEntity.MOVIE].append(imdb_id)
                else:
                    imdb_ids[ImageEntity.CAST].append(imdb_id)
        return imdb_ids

    def find_films(self) -> List[str]:
        logging.debug(f'Looking for imdb ids for all films')
        query = '''
                    PREFIX ddis: <http://ddis.ch/atai/>
                    PREFIX wd: <http://www.wikidata.org/entity/>
                    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                    PREFIX schema: <http://schema.org/>
                    SELECT DISTINCT ?x WHERE {
                        ?x wdt:P31 wd:Q11424
                    }
                '''
        films = []

        for row in self._kg.query(query):
            films.append(self.get_short_element_name(row.x.toPython()))
        return films
