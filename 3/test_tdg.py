import os
import unittest
import json
import networkx as nx
from generate_tdg import main as generate_tdg

class TestTDGGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Directory for test Java project
        cls.test_dir = 'test_java_project'
        # Generate TDG
        generate_tdg(cls.test_dir)
        # Load generated TDG
        with open(os.path.join(cls.test_dir, 'tdg.json'), 'r') as file:
            cls.tdg_data = json.load(file)
            cls.tdg = nx.readwrite.json_graph.node_link_graph(cls.tdg_data)

    def test_nodes_exist(self):
        expected_nodes = {
            'Method:main(args)',
            'Param:args',
            'Field:message',
            'Method:getMessage()',
            'Method:setMessage(message)',
            'Param:message',
            'Method:add(a, b)',
            'Param:a',
            'Param:b'
        }
        actual_nodes = set(self.tdg.nodes)
        for node in expected_nodes:
            with self.subTest(node=node):
                self.assertIn(node, actual_nodes)

    def test_node_types(self):
        expected_types = {
            'Method:main(args)': 'method',
            'Param:args': 'parameter',
            'Field:message': 'field',
            'Method:getMessage()': 'method',
            'Method:setMessage(message)': 'method',
            'Param:message': 'parameter',
            'Method:add(a, b)': 'method',
            'Param:a': 'parameter',
            'Param:b': 'parameter'
        }
        for node, expected_type in expected_types.items():
            with self.subTest(node=node):
                self.assertEqual(self.tdg.nodes[node]['type'], expected_type)

    def test_edges_exist(self):
        expected_edges = [
            ('Param:args', 'Method:main(args)'),
            ('Param:message', 'Method:setMessage(message)')
        ]
        for edge in expected_edges:
            with self.subTest(edge=edge):
                self.assertIn(edge, self.tdg.edges)

if __name__ == '__main__':
    unittest.main()

