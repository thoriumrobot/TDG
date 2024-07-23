import unittest
import json
import networkx as nx
from create_tdg import parse_java_file, extract_types, build_tdg, save_graph

class TestTDGCreation(unittest.TestCase):

    def setUp(self):
        # Sample Java code to parse and test
        self.sample_java_code = """
        public class Example {
            private int field1;
            private String field2;

            public int method1(int param1) {
                int localVar = param1 + field1;
                return localVar;
            }

            public void method2(String param2) {
                System.out.println(param2 + field2);
            }
        }
        """
        self.expected_types = {
            'Example': {
                'methods': {
                    'method1': {
                        'return_type': 'int',
                        'parameters': {'param1': 'int'},
                        'local_vars': {'localVar': 'int'}
                    },
                    'method2': {
                        'return_type': 'void',
                        'parameters': {'param2': 'String'},
                        'local_vars': {}
                    }
                },
                'fields': {
                    'field1': 'int',
                    'field2': 'String'
                }
            }
        }

    def test_parse_java_file(self):
        tree = parse_java_file('Example.java')
        self.assertIsNotNone(tree)

    def test_extract_types(self):
        tree = parse_java_file('Example.java')
        types = extract_types(tree)
        self.assertDictEqual(types, self.expected_types)

    def test_build_tdg(self):
        types = self.expected_types
        graph = build_tdg(types)
        
        # Check nodes
        self.assertIn('Example', graph.nodes)
        self.assertIn('Example.field1', graph.nodes)
        self.assertIn('Example.field2', graph.nodes)
        self.assertIn('Example.method1', graph.nodes)
        self.assertIn('Example.method1.param1', graph.nodes)
        self.assertIn('Example.method1.localVar', graph.nodes)
        self.assertIn('Example.method2', graph.nodes)
        self.assertIn('Example.method2.param2', graph.nodes)

        # Check edges
        self.assertTrue(graph.has_edge('Example.method1', 'int'))
        self.assertTrue(graph.has_edge('Example.method1', 'Example.method1.param1'))
        self.assertTrue(graph.has_edge('Example.method1.param1', 'int'))
        self.assertTrue(graph.has_edge('Example.method1', 'Example.method1.localVar'))
        self.assertTrue(graph.has_edge('Example.method1.localVar', 'int'))
        self.assertTrue(graph.has_edge('Example.method2', 'void'))
        self.assertTrue(graph.has_edge('Example.method2', 'Example.method2.param2'))
        self.assertTrue(graph.has_edge('Example.method2.param2', 'String'))
        self.assertTrue(graph.has_edge('Example', 'Example.field1'))
        self.assertTrue(graph.has_edge('Example.field1', 'int'))
        self.assertTrue(graph.has_edge('Example', 'Example.field2'))
        self.assertTrue(graph.has_edge('Example.field2', 'String'))

    def test_save_graph(self):
        types = self.expected_types
        graph = build_tdg(types)
        save_graph(graph, 'test_tdg.json')

        with open('test_tdg.json', 'r', encoding='utf-8') as file:
            data = json.load(file)

        loaded_graph = nx.readwrite.json_graph.node_link_graph(data)
        self.assertTrue(nx.is_isomorphic(graph, loaded_graph))

if __name__ == '__main__':
    # Write the sample Java code to a temporary file for testing
    with open('Example.java', 'w', encoding='utf-8') as file:
        file.write("""
        public class Example {
            private int field1;
            private String field2;

            public int method1(int param1) {
                int localVar = param1 + field1;
                return localVar;
            }

            public void method2(String param2) {
                System.out.println(param2 + field2);
            }
        }
        """)
    
    unittest.main()

