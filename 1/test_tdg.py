import os
import unittest
import json
from create_tdg import TypeDependencyGraph, process_java_file, process_project_directory

class TestTypeDependencyGraph(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = os.path.join(os.path.dirname(__file__), 'sample_java')
    
    def test_tdg_creation(self):
        # Define the expected TDG output
        expected_tdg = {
            "DataUtil": {
                "depends_on": [],
                "type": "Class"
            },
            "DataUtil.formatData": {
                "depends_on": [
                    "DataUtil.formatData.return",
                    "DataUtil.formatData.data"
                ],
                "type": "Method"
            },
            "DataUtil.formatData.return": {
                "depends_on": [],
                "type": "String"
            },
            "DataUtil.formatData.data": {
                "depends_on": [],
                "type": "String"
            },
            "DataUtil.main": {
                "depends_on": [
                    "DataUtil.main.args",
                    "DataUtil.main.result"
                ],
                "type": "Method"
            },
            "DataUtil.main.args": {
                "depends_on": [],
                "type": "String[]"
            },
            "DataUtil.main.result": {
                "depends_on": [
                    "DataUtil.formatData"
                ],
                "type": "String"
            }
        }

        # Create the TDG
        tdg = TypeDependencyGraph()
        java_file_path = os.path.join(self.test_dir, 'DataUtil.java')
        process_java_file(java_file_path, tdg)
        actual_tdg = tdg.to_dict()

        # Assert that the actual TDG matches the expected TDG
        self.assertEqual(actual_tdg, expected_tdg)
    
    def test_project_tdg_creation(self):
        # Define the expected TDG output for the project
        expected_tdg = {
            "DataUtil": {
                "depends_on": [],
                "type": "Class"
            },
            "DataUtil.formatData": {
                "depends_on": [
                    "DataUtil.formatData.return",
                    "DataUtil.formatData.data"
                ],
                "type": "Method"
            },
            "DataUtil.formatData.return": {
                "depends_on": [],
                "type": "String"
            },
            "DataUtil.formatData.data": {
                "depends_on": [],
                "type": "String"
            },
            "DataUtil.main": {
                "depends_on": [
                    "DataUtil.main.args",
                    "DataUtil.main.result"
                ],
                "type": "Method"
            },
            "DataUtil.main.args": {
                "depends_on": [],
                "type": "String[]"
            },
            "DataUtil.main.result": {
                "depends_on": [
                    "DataUtil.formatData"
                ],
                "type": "String"
            }
        }

        # Process the entire project directory
        output_json = 'test_type_dependency_graph.json'
        process_project_directory(self.test_dir, output_json)

        # Load the generated TDG
        with open(output_json, 'r') as f:
            actual_tdg = json.load(f)
        
        # Assert that the actual TDG matches the expected TDG
        self.assertEqual(actual_tdg, expected_tdg)

        # Clean up the generated JSON file
        os.remove(output_json)

if __name__ == '__main__':
    unittest.main()

