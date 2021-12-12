import logging
import re
from pathlib import Path
from typing import List

import pandas as pd
from nltk.corpus import stopwords

from src import utils
from src.knowledge_graph import KnowledgeGraph
from src.ner import NameNameEntityRecognitionModel


class Multimedia:
    MULTIMEDIA_PATH = utils.get_data_path('images.json')
    MULTIMEDIA_KEYWORDS = ("pictur", "imag", "poster", "frame")

    def __init__(self, knowledge_graph: KnowledgeGraph, ner: NameNameEntityRecognitionModel,
                 path: Path = MULTIMEDIA_PATH):
        self._knowledge_graph = knowledge_graph
        self._ner = ner
        self._read_data(path)
        self._stop_words = set(stopwords.words('english'))

    def _read_data(self, path: Path) -> None:
        with open(path.resolve()) as f:
            self._image_data = pd.read_json('src/data/raw_data/images.json')

    def process_question(self, question: str) -> str:
        imdb_ids = self._find_imdb_id(self._find_named_entities(question))
        if not imdb_ids:
            raise ValueError('No imdb id found')

        if "poster" in question:
            key = 'poster'
        elif "frame" in question:
            return self._process_frame(question)
        else:
            return self._process_picture(question)

        return self._find_images(imdb_ids=imdb_ids, img_type=key)

    def _find_images(self, imdb_ids: List[str], img_type: str) -> str:
        df = self._image_data[self._image_data.type == img_type]

        for imdb_id in imdb_ids:
            df = df[df.movie.apply(lambda row: imdb_id in row) | df.cast.apply(lambda row: imdb_id in row)]

        if df.empty:
            raise ValueError('Image not found')
        answer = ['I have found some pictures:']
        for img in df.img[:3]:
            answer.append(f'image:{re.sub(".jpg", "", img)}')
        return '\n'.join(answer)

    def _find_named_entities(self, query: str) -> List[str]:
        logging.debug(f'Looking for entities')
        ner_result = self._ner.find_name_entities(query)
        nodes = []
        for named_entities in ner_result.values():
            for named_entity in named_entities:
                nodes.extend(self._knowledge_graph.get_closest_node(named_entity, predicate=False))
        if nodes:
            return nodes
        return self._knowledge_graph.get_closest_node(utils.remove_stop_words(self._stop_words, query), predicate=False)

    def _find_imdb_id(self, nodes: List[str]) -> List[str]:
        logging.debug(f'Looking for imdb id')

        query = '''
                    PREFIX ddis: <http://ddis.ch/atai/> 
                    PREFIX wd: <http://www.wikidata.org/entity/> 
                    PREFIX wdt: <http://www.wikidata.org/prop/direct/> 
                    PREFIX schema: <http://schema.org/> 
                    SELECT DISTINCT ?x WHERE {{ 
                        {n} wdt:P345 ?x 
                    }}
                '''

        imdb_ids = []
        for node in nodes:
            logging.debug(f'Query imdb id of {node}')
            formatted_query = query.format(n=node)
            for row in self._knowledge_graph.query(formatted_query):
                imdb_ids.append(row.x.toPython())
        return imdb_ids
