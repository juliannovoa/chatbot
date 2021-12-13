import csv
import logging
from dataclasses import astuple
from typing import Tuple

import numpy as np
import pandas as pd
from pathlib import Path

from src import utils, Fact
from src.knowledge_graph import KnowledgeGraph
from src.utils import read_csv


class Embeddings:
    ENTITY_EMBEDDINGS_PATH = utils.get_data_path('entity_embeds.npy')
    PREDICATE_EMBEDDINGS_PATH = utils.get_data_path('entity_embeds.npy')
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
                self._knowledge_graph.get_short_element_name(ent):
                    (entity_emb[int(idx)] / np.linalg.norm(entity_emb[int(idx)])).tolist()
                for idx, ent in csv.reader(f, delimiter='\t')
            }
        df_entities = pd.DataFrame(data=entities.items(), columns=['entity', 'embedding'])
        df_entities.set_index('entity', inplace=True)

        logging.debug('Parsing embeddings of predicates.')
        with open(predicate2id.resolve(), 'r') as f:
            entities = {
                self._knowledge_graph.get_short_element_name(ent):
                    (predicate_emb[int(idx)] / np.linalg.norm(predicate_emb[int(idx)])).tolist()
                for idx, ent in csv.reader(f, delimiter='\t')
            }

        df_predicates = pd.DataFrame(data=entities.items(), columns=['predicate', 'embedding'])
        df_predicates.set_index('predicate', inplace=True)

        return df_entities, df_predicates

    def check_triplet(self, fact: Fact, threshold: int = 100) -> str:
        subject, predicate, obj = astuple(fact)
        if subject not in self._entity_embeddings.index \
                or obj not in self._entity_embeddings.index \
                or predicate not in self._predicate_embeddings.index:
            return ''

        expected_object_embedding = \
            self._entity_embeddings.embedding.loc[subject] + self._predicate_embeddings.embedding.loc[predicate]
        ranking = self._entity_embeddings.embedding.apply(lambda row: expected_object_embedding.dot(row)).rank(ascending=False)
        if ranking.loc[obj] < threshold:
            return ''
        output = ['However, I think that my information could be wrong.',
                  f'The {self._knowledge_graph.get_node_label(predicate, is_predicate=True)} of {self._knowledge_graph.get_node_label(subject)} could be:']
        for idx in ranking[ranking <= 3].index:
            output.append(f'\t{self._knowledge_graph.get_node_label(idx)}')

        return '\n'.join(output)
