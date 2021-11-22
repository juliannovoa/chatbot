import os
import pickle
from unittest.mock import patch

from pathlib import Path
from unittest import TestCase

from rdflib import Graph

from src.question_answering_engine import InformationFinder


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
            f.write(f'<{str(InformationFinder.WD)}{cls.ENTITY_1}> ')
            f.write(f'<{str(InformationFinder.RDFS)}label> \"{cls.LABEL_ENTITY_1}\"@en ;\n')
            f.write(f'    <{str(InformationFinder.WDT)}{cls.PREDICATE}> ')
            f.write(f'<{str(InformationFinder.WD)}{cls.ENTITY_2}> .')
        f.close()

    @classmethod
    def delete_nt(cls):
        os.remove(cls.MOCK_FILENAME)
        os.remove(cls.MOCK_GRAPH)

    def test_init_existing_model(self):
        graph = Graph()
        graph_path = Path('./mock.graph')
        with open(graph_path, 'wb') as f:
            pickle.dump(graph, f)
            f.close()
        t1 = os.path.getmtime(graph_path)
        information_finder = InformationFinder(parsed_graph=graph_path)
        t2 = os.path.getmtime(graph_path)
        self.assertEqual(t1, t2)
        self.assertEqual(type(graph), type(information_finder._g))
        os.remove(graph_path)

    @patch('rdflib.Graph.parse')
    def test_init_not_existing_model(self, parse_graph):
        parse_graph.return_value = Graph()
        graph_path = Path('./mock.model')
        self.assertFalse(graph_path.exists())
        InformationFinder(parsed_graph=graph_path)
        self.assertTrue(graph_path.exists())
        os.remove(graph_path)

    def test_parse_predicates(self):
        self.create_nt()
        info_finder = InformationFinder(raw_graph=Path(self.MOCK_FILENAME),
                                        parsed_graph=Path(self.MOCK_GRAPH))
        expected_predicates = {f'{str(InformationFinder.WDT)}{self.PREDICATE}': self.LABEL_PREDICATE}
        self.assertDictEqual(expected_predicates, info_finder._predicates)
        self.delete_nt()

    def test_parse_instances(self):
        self.create_nt()
        info_finder = InformationFinder(raw_graph=Path(self.MOCK_FILENAME),
                                        parsed_graph=Path(self.MOCK_GRAPH))
        expected_instances = {f'{str(InformationFinder.WD)}{self.ENTITY_1}': self.LABEL_ENTITY_1,
                              f'{str(InformationFinder.WD)}{self.ENTITY_2}': self.LABEL_ENTITY_2}
        self.assertDictEqual(expected_instances, info_finder._nodes)
        self.delete_nt()

    def test_node_is_instance(self):
        self.assertTrue(InformationFinder.node_is_instance('http://www.wikidata.org/entity/Q'))
        self.assertFalse(InformationFinder.node_is_instance('http://www.wikidata.org/prop/direct/Q'))

    def test_node_is_predicate(self):
        self.assertTrue(InformationFinder.node_is_predicate('http://www.wikidata.org/prop/direct/Q'))
        self.assertTrue(InformationFinder.node_is_predicate('http://ddis.ch/atai/Q'))
        self.assertTrue(InformationFinder.node_is_predicate('http://schema.org/Q'))
        self.assertFalse(InformationFinder.node_is_predicate('http://www.wikidata.org/entity/Q'))
