import os
import sys
import json
import javalang
import networkx as nx
import logging
from collections import defaultdict
import traceback

class JavaTDG:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.classnames = set()

    def add_node(self, node_id, node_type, name):
        self.graph.add_node(node_id, attr={'type': node_type, 'name': name})
        logging.debug(f"Added node {node_id} with attributes {self.graph.nodes[node_id]['attr']}")

    def add_edge(self, from_node, to_node, edge_type):
        self.graph.add_edge(from_node, to_node, type=edge_type)

    def add_classname(self, classname):
        self.classnames.add(classname)

def process_file(file_path, output_dir):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        tree = javalang.parse.parse(content)
        file_name = os.path.basename(file_path)
        logging.info(f"Processing file {file_path}")

        for path, node in tree:
            if isinstance(node, javalang.tree.ClassDeclaration):
                tdg = JavaTDG()
                class_id = f"{file_name}.{node.name}"
                tdg.add_node(class_id, "class", node.name)
                tdg.add_classname(node.name)
                for method in node.methods:
                    method_id = f"{class_id}.{method.name}()"
                    tdg.add_node(method_id, "method", method.name)
                    tdg.add_edge(class_id, method_id, "contains")
                    for param in method.parameters:
                        param_id = f"{method_id}.{param.name}"
                        tdg.add_node(param_id, "parameter", param.name)
                        tdg.add_edge(method_id, param_id, "has_parameter")
                for field in node.fields:
                    for decl in field.declarators:
                        field_id = f"{class_id}.{decl.name}"
                        tdg.add_node(field_id, "field", decl.name)
                        tdg.add_edge(class_id, field_id, "has_field")
                for path, node in tree:
                    if isinstance(node, javalang.tree.Literal) and node.value == "null":
                        if node.position:
                            null_id = f"{file_name}.null_{node.position.line}_{node.position.column}"
                            tdg.add_node(null_id, "literal", "null")
                            parent = path[-2] if len(path) > 1 else None
                            if parent:
                                parent_id = None
                                if hasattr(parent, 'name'):
                                    parent_id = f"{file_name}.{parent.name}"
                                elif isinstance(parent, javalang.tree.MethodInvocation):
                                    parent_id = f"{file_name}.{parent.member}"
                                elif isinstance(parent, javalang.tree.Assignment):
                                    parent_id = f"{file_name}.assignment_{parent.position.line}_{parent.position.column}"
                                if parent_id:
                                    tdg.add_edge(parent_id, null_id, "contains")
                
                output_path = os.path.join(output_dir, f"{class_id}.json")
                save_tdg_to_json(tdg, output_path)
    except javalang.parser.JavaSyntaxError as e:
        logging.error(f"Syntax error in file {file_path}: {e}")
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        logging.error(traceback.format_exc())

def save_tdg_to_json(tdg, output_path):
    data = nx.node_link_data(tdg.graph)
    with open(output_path, 'w') as f:
        json.dump(data, f)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_tdg.py <ProjectDir> <OutputDir>")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO)
    project_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(project_dir):
        for file in files:
            if file.endswith('.java'):
                process_file(os.path.join(root, file), output_dir)
