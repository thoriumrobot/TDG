import os
import sys
import json
import javalang
import networkx as nx
import numpy as np
from tensorflow.keras.models import load_model
import logging

class JavaTDG:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.classnames = set()
        self.aliasgraph = nx.DiGraph()
        self.callgraph = defaultdict(list)

    def add_node(self, node_id, node_type, name):
        self.graph.add_node(node_id, type=node_type, name=name)

    def add_edge(self, from_node, to_node, edge_type):
        self.graph.add_edge(from_node, to_node, type=edge_type)

    def add_classname(self, classname):
        self.classnames.add(classname)

    def add_alias(self, from_node, to_node):
        self.aliasgraph.add_edge(from_node, to_node, type='alias')

def process_file(file_path, tdg):
    with open(file_path, 'r') as file:
        content = file.read()
    tree = javalang.parse.parse(content)
    file_name = os.path.basename(file_path)

    for path, node in tree:
        if isinstance(node, javalang.tree.ClassDeclaration):
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
        elif isinstance(node, javalang.tree.MethodDeclaration):
            method_id = f"{file_name}.{node.name}()"
            tdg.add_node(method_id, "method", node.name)
            for param in node.parameters:
                param_id = f"{method_id}.{param.name}"
                tdg.add_node(param_id, "parameter", param.name)
                tdg.add_edge(method_id, param_id, "has_parameter")
        elif isinstance(node, javalang.tree.FieldDeclaration):
            for decl in node.declarators:
                field_id = f"{file_name}.{decl.name}"
                tdg.add_node(field_id, "field", decl.name)
                tdg.add_edge(file_name, field_id, "has_field")
        elif isinstance(node, javalang.tree.VariableDeclarator):
            var_id = f"{file_name}.{node.name}"
            tdg.add_node(var_id, "variable", node.name)

def process_directory(directory_path):
    tdg = JavaTDG()
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.java'):
                process_file(os.path.join(root, file), tdg)
    return tdg

def extract_features(node):
    return [node['attr'].get('type', 0), node['attr'].get('name', 0)]

def preprocess_tdg(tdg):
    features = []
    node_ids = []
    for node in tdg.graph.nodes(data=True):
        node_id, attr = node
        feature_vector = extract_features(attr)
        features.append(feature_vector)
        node_ids.append(node_id)
    return np.array(features), node_ids

def annotate_file(file_path, annotations):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    for annotation in annotations:
        node_id, line_num, col_num = annotation
        lines[line_num - 1] = lines[line_num - 1][:col_num] + "@Nullable " + lines[line_num - 1][col_num:]
    
    with open(file_path, 'w') as file:
        file.writelines(lines)

def main(project_dir, model_path, output_dir):
    tdg = process_directory(project_dir)
    features, node_ids = preprocess_tdg(tdg)
    
    model = load_model(model_path)
    predictions = model.predict(features)
    
    annotations = []
    for node_id, prediction in zip(node_ids, predictions):
        if prediction > 0.5:  # Assuming a threshold of 0.5 for @Nullable annotation
            node_info = node_id.split('.')
            file_name = node_info[0]
            line_num = int(node_info[1])
            col_num = 0  # Adjust as necessary, this example assumes column 0

            annotations.append((node_id, line_num, col_num))
    
    for file_name in set([ann[0].split('.')[0] for ann in annotations]):
        file_path = os.path.join(project_dir, file_name)
        file_annotations = [ann for ann in annotations if ann[0].split('.')[0] == file_name]
        annotate_file(file_path, file_annotations)
    
    logging.info("Annotation complete.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python predict.py <ProjectDir> <ModelPath> <OutputDir>")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO)

    project_dir = sys.argv[1]
    model_path = sys.argv[2]
    output_dir = sys.argv[3]

    main(project_dir, model_path, output_dir)

