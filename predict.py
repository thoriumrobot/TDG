import os
import sys
import json
import javalang
import networkx as nx
import numpy as np
from tensorflow.keras.models import load_model
import logging
from collections import defaultdict
import traceback

class JavaTDG:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.classnames = set()

    def add_node(self, node_id, node_type, name, nullable=False):
        self.graph.add_node(node_id, attr={'type': node_type, 'name': name, 'nullable': nullable})
        logging.debug(f"Added node {node_id} with attributes {self.graph.nodes[node_id]['attr']}")

    def add_edge(self, from_node, to_node, edge_type):
        self.graph.add_edge(from_node, to_node, type=edge_type)

    def add_classname(self, classname):
        self.classnames.add(classname)

def get_parent_id(file_name, parent):
    if parent is None:
        return None
    if hasattr(parent, 'name'):
        return f"{file_name}.{parent.name}"
    if isinstance(parent, javalang.tree.MethodInvocation):
        return f"{file_name}.{parent.member}"
    if isinstance(parent, javalang.tree.Assignment):
        if parent.position:
            return f"{file_name}.assignment_{parent.position.line}_{parent.position.column}"
    return None

def process_file(file_path, tdg):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        tree = javalang.parse.parse(content)
        file_name = os.path.basename(file_path)
        logging.info(f"Processing file {file_path}")

        for path, node in tree:
            if isinstance(node, javalang.tree.ClassDeclaration):
                class_id = f"{file_name}.{node.name}"
                tdg.add_node(class_id, "class", node.name)
                tdg.add_classname(node.name)
                for method in node.methods:
                    method_id = f"{class_id}.{method.name}()"
                    tdg.add_node(method_id, "method", node.name)
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
            elif isinstance(node, javalang.tree.Literal) and node.value == "null":
                if node.position:
                    null_id = f"{file_name}.null_{node.position.line}_{node.position.column}"
                    tdg.add_node(null_id, "literal", "null")
                    parent = path[-2] if len(path) > 1 else None
                    parent_id = get_parent_id(file_name, parent)
                    if parent_id:
                        tdg.add_edge(parent_id, null_id, "contains")
    except javalang.parser.JavaSyntaxError as e:
        logging.error(f"Syntax error in file {file_path}: {e}")
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        logging.error(traceback.format_exc())

def process_directory(directory_path):
    tdg = JavaTDG()
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.java'):
                process_file(os.path.join(root, file), tdg)
    return tdg

def extract_features(attr):
    type_mapping = {'class': 0, 'method': 1, 'field': 2, 'parameter': 3, 'variable': 4, 'literal': 5}
    name_mapping = defaultdict(lambda: len(name_mapping))

    node_type = attr.get('type', '')
    node_name = attr.get('name', '')
    nullable = float(attr.get('nullable', 0))

    type_id = type_mapping.get(node_type, len(type_mapping))
    name_id = name_mapping[node_name]

    return [float(type_id), float(name_id), nullable]

def preprocess_tdg(tdg):
    features = []
    node_ids = []
    for node in tdg.graph.nodes(data=True):
        node_id, attr = node
        if 'attr' in attr:
            feature_vector = extract_features(attr['attr'])
            features.append(feature_vector)
            node_ids.append(node_id)
        else:
            logging.warning(f"Node {node_id} is missing 'attr' attribute")
    logging.info(f"Extracted {len(features)} features from TDG")
    return np.array(features), node_ids

def annotate_file(file_path, annotations):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    for annotation in annotations:
        node_id, line_num, col_num = annotation
        if 0 <= line_num - 1 < len(lines):
            lines[line_num - 1] = lines[line_num - 1][:col_num] + "@Nullable " + lines[line_num - 1][col_num:]
        else:
            logging.warning(f"Line number {line_num} is out of range in file {file_path}")
    
    with open(file_path, 'w') as file:
        file.writelines(lines)

def process_project(project_dir, model):
    tdg = process_directory(project_dir)
    features, node_ids = preprocess_tdg(tdg)

    if features.size == 0:
        logging.warning(f"No valid features extracted for project {project_dir}. Skipping annotation.")
        return
    
    predictions = model.predict(features)
    
    annotations = []
    for node_id, prediction in zip(node_ids, predictions):
        if prediction > 0.5:  # Assuming a threshold of 0.5 for @Nullable annotation
            node_info = node_id.split('.')
            file_name = node_info[0]
            try:
                line_num = int(node_info[1])
            except ValueError:
                logging.warning(f"Invalid line number in node_id {node_id} for project {project_dir}. Skipping annotation.")
                continue
            col_num = 0

            annotations.append((node_id, line_num, col_num))
    
    for file_name in set([ann[0].split('.')[0] for ann in annotations]):
        file_path = os.path.join(project_dir, file_name)
        file_annotations = [ann for ann in annotations if ann[0].split('.')[0] == file_name]
        annotate_file(file_path, file_annotations)
    
    logging.info(f"Annotation complete for project {project_dir}")

def main(project_dir, model_path, output_dir):
    model = load_model(model_path)
    
    for subdir in os.listdir(project_dir):
        subdir_path = os.path.join(project_dir, subdir)
        if os.path.isdir(subdir_path):
            output_subdir = os.path.join(output_dir, subdir)
            os.makedirs(output_subdir, exist_ok=True)
            process_project(subdir_path, model)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python predict.py <ProjectDir> <ModelPath> <OutputDir>")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO)

    project_dir = sys.argv[1]
    model_path = sys.argv[2]
    output_dir = sys.argv[3]

    main(project_dir, model_path, output_dir)
