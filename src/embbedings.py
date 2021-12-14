import csv
import logging
from dataclasses import astuple
from enum import Enum
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from pathlib import Path

from src import utils, Fact
from src.knowledge_graph import KnowledgeGraph
from src.utils import read_csv


class DataFrameFields(Enum):
    ENTITY = 'entity'
    PREDICATE = 'predicate'
    EMBEDDING = 'embedding'
    IS_MOVIE = 'is_movie'


class Embeddings:
    ENTITY_EMBEDDINGS_PATH = utils.get_data_path('entity_embeds.npy')
    PREDICATE_EMBEDDINGS_PATH = utils.get_data_path('relation_embeds.npy')
    ENTITY_TO_ID_PATH = utils.get_data_path('entity_ids.del')
    PREDICATE_TO_ID_PATH = utils.get_data_path('relation_ids.del')
    PARSED_ENTITY_EMBEDDINGS_PATH = utils.get_model_path('entity_embeds.parsed')
    PARSED_PREDICATE_EMBEDDINGS_PATH = utils.get_model_path('predicate_embeds.parsed')

    def __init__(self, knowledge_graph: KnowledgeGraph,
                 parsed_entity_embeddings: Path = PARSED_ENTITY_EMBEDDINGS_PATH,
                 parsed_predicate_embeddings: Path = PARSED_PREDICATE_EMBEDDINGS_PATH):

        self._knowledge_graph = knowledge_graph

        if not parsed_entity_embeddings.exists() or not parsed_predicate_embeddings.exists():
            entities, predicates = self._parse_embeddings()
            logging.debug('Saving predicates embeddings.')
            predicates.to_csv(parsed_predicate_embeddings.resolve())
            del predicates
            logging.debug('Saving entities embeddings.')
            entities.to_csv(parsed_entity_embeddings.resolve())
            del entities
            logging.debug('Entities and predicates saved.')

        logging.info('Loading predicates embeddings.')
        self._predicate_embeddings = read_csv(parsed_predicate_embeddings)
        logging.info('Loading entities embeddings.')
        self._entity_embeddings = read_csv(parsed_entity_embeddings)
        logging.info('Entities predicates loaded embeddings.')

    def _parse_embeddings(self,
                          entity_path: Path = ENTITY_EMBEDDINGS_PATH,
                          predicate_path: Path = PREDICATE_EMBEDDINGS_PATH,
                          entity2id: Path = ENTITY_TO_ID_PATH,
                          predicate2id: Path = PREDICATE_TO_ID_PATH) -> Tuple[pd.DataFrame, pd.DataFrame]:

        entity_emb = np.load(entity_path.resolve())
        predicate_emb = np.load(predicate_path.resolve())

        logging.debug('Parsing embeddings of entities.')
        with open(entity2id.resolve(), 'r') as f:
            entities = {
                self._knowledge_graph.get_short_element_name(ent): entity_emb[int(idx)].tolist()
                for idx, ent in csv.reader(f, delimiter='\t')
            }
        df_entities = pd.DataFrame(data=entities.items(), columns=[DataFrameFields.ENTITY.value,
                                                                   DataFrameFields.EMBEDDING.value])
        df_entities.set_index(DataFrameFields.ENTITY.value, inplace=True)
        films = self._knowledge_graph.find_films()
        df_entities[DataFrameFields.IS_MOVIE.value] = df_entities.index.isin(films)

        logging.debug('Parsing embeddings of predicates.')
        with open(predicate2id.resolve(), 'r') as f:
            entities = {
                self._knowledge_graph.get_short_element_name(ent): predicate_emb[int(idx)].tolist()
                for idx, ent in csv.reader(f, delimiter='\t')
            }

        df_predicates = pd.DataFrame(data=entities.items(), columns=[DataFrameFields.PREDICATE.value,
                                                                     DataFrameFields.EMBEDDING.value])
        df_predicates.set_index(DataFrameFields.PREDICATE.value, inplace=True)

        return df_entities, df_predicates

    def check_triplet(self, fact: Fact, threshold: int = 100) -> Optional[str]:
        logging.debug(f'Checking triplet: {fact}')
        subject, predicate, obj = astuple(fact)
        if subject not in self._entity_embeddings.index \
                or obj not in self._entity_embeddings.index \
                or predicate not in self._predicate_embeddings.index:
            return None

        expected_embedding = \
            self._entity_embeddings.embedding.loc[subject] + self._predicate_embeddings.embedding.loc[predicate]
        ranking = self._entity_embeddings.embedding.apply(
            lambda row: np.linalg.norm(expected_embedding - row)).rank()
        if ranking.loc[obj] < threshold:
            return None
        output = [
            f'The {self._knowledge_graph.get_node_label(predicate, is_predicate=True)} of {self._knowledge_graph.get_node_label(subject)} could be:']
        for idx in ranking[ranking <= 3].index:
            output.append(f'\t{self._knowledge_graph.get_node_label(idx)}')

        return '\n'.join(output)

    def get_similar_film(self, film_entities: List[str], top_n: int = 8, n_return=2) -> List[str]:
        logging.debug('Get similar films with embeddings.')
        output = []
        for film_entity in film_entities:
            if not self._entity_embeddings[DataFrameFields.IS_MOVIE.value].loc[film_entity]:
                logging.debug(f'{film_entity} is not a film.')
                continue
            movie_embedding = self._entity_embeddings[DataFrameFields.EMBEDDING.value].loc[film_entity]
            movies = self._entity_embeddings[self._entity_embeddings[DataFrameFields.IS_MOVIE.value]]
            ranking = movies[DataFrameFields.EMBEDDING.value].apply(
                lambda row: np.linalg.norm(movie_embedding - row)).rank()

            for film in ranking[(ranking <= top_n) & (ranking > 0)].sample(n_return).index:
                output.append(f'\t{self._knowledge_graph.get_node_label(film)}')
        logging.debug(f'Recommendations: {output}')
        return output
