import os
import pickle
from unittest import TestCase
from unittest.mock import patch

import rdflib
from pathlib import Path
from rdflib import Graph

from src.question_answering_engine import KnowledgeGraph


class TestInformationFinder(TestCase):
    MOCK_FILENAME = 'mock.nt'
    MOCK_GRAPH = 'mock.graph'
    ENTITY_1 = 'P125'
    LABEL_ENTITY_1 = 'Rotten Tomatoes ID'
    ENTITY_2 = 'Q30'
    LABEL_ENTITY_2 = 'Q30'
    PREDICATE = 'P17'
    LABEL_PREDICATE = 'P17'

    @classmethod
    def create_nt(cls):
        with open(cls.MOCK_FILENAME, 'w') as f:
            f.write(f'<{str(KnowledgeGraph.WD)}{cls.ENTITY_1}> ')
            f.write(f'<{str(KnowledgeGraph.RDFS)}label> \"{cls.LABEL_ENTITY_1}\"@en ;\n')
            f.write(f'    <{str(KnowledgeGraph.WDT)}{cls.PREDICATE}> ')
            f.write(f'<{str(KnowledgeGraph.WD)}{cls.ENTITY_2}> .')
        f.close()

    @classmethod
    def delete_nt(cls):
        os.remove(cls.MOCK_FILENAME)
        os.remove(cls.MOCK_GRAPH)

    @patch('sentence_transformers.SentenceTransformer.__init__')
    def test_init_existing_model(self, sent_transformer):
        sent_transformer.return_value = None
        graph = Graph()
        graph_path = Path('./mock.graph')
        with open(graph_path, 'wb') as f:
            pickle.dump(graph, f)
            f.close()
        t1 = os.path.getmtime(graph_path)
        information_finder = KnowledgeGraph(parsed_graph=graph_path)
        t2 = os.path.getmtime(graph_path)
        self.assertEqual(t1, t2)
        self.assertEqual(type(graph), type(information_finder._g))
        os.remove(graph_path)

    @patch('rdflib.Graph.parse')
    @patch('sentence_transformers.SentenceTransformer.__init__')
    def test_init_not_existing_model(self, sent_transformer, parse_graph):
        sent_transformer.return_value = None
        parse_graph.return_value = Graph()
        graph_path = Path('./mock.model')
        self.assertFalse(graph_path.exists())
        KnowledgeGraph(parsed_graph=graph_path)
        self.assertTrue(graph_path.exists())
        os.remove(graph_path)

    @patch('sentence_transformers.SentenceTransformer.__init__')
    @patch('sentence_transformers.SentenceTransformer.encode')
    def test_parse_predicates(self, sent_encoded, sent_transformer):
        sent_transformer.return_value = None
        self.create_nt()
        info_finder = KnowledgeGraph(raw_graph=Path(self.MOCK_FILENAME),
                                     parsed_graph=Path(self.MOCK_GRAPH))
        expected_predicates = {f'{str(KnowledgeGraph.WDT)}{self.PREDICATE}': {'description': self.LABEL_PREDICATE}}
        self.assertDictEqual(expected_predicates, info_finder._predicates)
        self.delete_nt()

    @patch('sentence_transformers.SentenceTransformer.__init__')
    @patch('sentence_transformers.SentenceTransformer.encode')
    def test_parse_instances(self, sent_encoded, sent_transformer):
        sent_transformer.return_value = None
        self.create_nt()
        info_finder = KnowledgeGraph(raw_graph=Path(self.MOCK_FILENAME),
                                     parsed_graph=Path(self.MOCK_GRAPH))
        expected_instances = {f'{str(KnowledgeGraph.WD)}{self.ENTITY_1}': {'description': self.LABEL_ENTITY_1},
                              f'{str(KnowledgeGraph.WD)}{self.ENTITY_2}': {'description': self.LABEL_ENTITY_2}}
        self.assertDictEqual(expected_instances, info_finder._entities)
        self.delete_nt()

    def test_node_is_instance(self):
        self.assertTrue(KnowledgeGraph.element_is_entity('http://www.wikidata.org/entity/Q'))
        self.assertFalse(KnowledgeGraph.element_is_entity('http://www.wikidata.org/prop/direct/Q'))

    def test_node_is_predicate(self):
        self.assertTrue(KnowledgeGraph.element_is_predicate('http://www.wikidata.org/prop/direct/Q'))
        self.assertTrue(KnowledgeGraph.element_is_predicate('http://ddis.ch/atai/Q'))
        self.assertTrue(KnowledgeGraph.element_is_predicate('http://schema.org/Q'))
        self.assertFalse(KnowledgeGraph.element_is_predicate('http://www.wikidata.org/entity/Q'))

    def test_get_closest_instance(self):
        self.create_nt()
        info_finder = KnowledgeGraph(raw_graph=Path(self.MOCK_FILENAME),
                                     parsed_graph=Path(self.MOCK_GRAPH))

        pred = info_finder.get_closest_node(self.LABEL_ENTITY_1)
        self.assertEqual(pred, [f'{str(KnowledgeGraph.WD)}{self.ENTITY_1}'])
        self.delete_nt()

    def test_get_closest_predicate(self):
        self.create_nt()
        info_finder = KnowledgeGraph(raw_graph=Path(self.MOCK_FILENAME),
                                     parsed_graph=Path(self.MOCK_GRAPH))

        pred = info_finder.get_closest_node(self.LABEL_PREDICATE, predicate=True)
        self.assertEqual(pred, [f'{str(KnowledgeGraph.WDT)}{self.PREDICATE}'])
        self.delete_nt()

    def test_query(self):
        self.create_nt()
        info_finder = KnowledgeGraph(raw_graph=Path(self.MOCK_FILENAME),
                                     parsed_graph=Path(self.MOCK_GRAPH))
        query = f'''
            SELECT ?x WHERE {{
                ?x <{str(KnowledgeGraph.WDT)}{self.PREDICATE}>  <{str(KnowledgeGraph.WD)}{self.ENTITY_2}> .
            }}
        '''
        response = info_finder.query(query)
        self.assertIsInstance(response, rdflib.query.Result)
        for row in response:
            self.assertEqual(row.x.toPython(), f'{str(KnowledgeGraph.WD)}{self.ENTITY_1}')
        self.delete_nt()
