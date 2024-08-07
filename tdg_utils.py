import json
import networkx as nx
from collections import defaultdict
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import random
import logging
import javalang
import traceback

class JavaTDG:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.classnames = set()

    def add_node(self, node_id, node_type, name, nullable=False, actual_type=None):
        self.graph.add_node(node_id, attr={'type': node_type, 'name': name, 'nullable': nullable, 'actual_type': actual_type})

    def add_edge(self, from_node, to_node, edge_type):
        self.graph.add_edge(from_node, to_node, type=edge_type)

    def add_classname(self, classname):
        self.classnames.add(classname)

def f1_score(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def extract_features(attr):
    type_mapping = {'class': 0, 'method': 1, 'field': 2, 'parameter': 3, 'variable': 4, 'literal': 5}
    name_mapping = defaultdict(lambda: len(name_mapping))
    type_name_mapping = defaultdict(lambda: len(type_name_mapping))

    node_type = attr.get('type', '')
    node_name = attr.get('name', '')
    actual_type = attr.get('actual_type', '')
    nullable = float(attr.get('nullable', 0))

    type_id = type_mapping.get(node_type, len(type_mapping))
    name_id = name_mapping[node_name]
    type_name_id = type_name_mapping[actual_type]

    return [float(type_id), float(name_id), float(type_name_id), nullable]

def preprocess_tdg(tdg):
    features = []
    labels = []
    node_ids = []
    for node_id, attr in tdg.graph.nodes(data='attr'):
        if attr and attr.get('type') in ['method', 'field', 'parameter']:
            feature_vector = extract_features(attr)
            label = float(attr.get('nullable', 0))
            features.append(feature_vector)
            labels.append(label)
            node_ids.append(node_id)
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.float32), np.array(node_ids, dtype=np.str)

def load_tdg_data(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        tdg = JavaTDG()
        tdg.graph = nx.node_link_graph(data)
        return preprocess_tdg(tdg)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {json_path}: {e}")
        return np.array([]), np.array([]), np.array([])  # Return empty arrays if there's an error

def balance_dataset(features, labels):
    pos_indices = [i for i, label in enumerate(labels) if label == 1]
    neg_indices = [i for i, label in enumerate(labels) if label == 0]
    
    random.shuffle(neg_indices)
    selected_neg_indices = neg_indices[:len(pos_indices)]
    
    selected_indices = pos_indices + selected_neg_indices
    random.shuffle(selected_indices)
    
    balanced_features = np.array([features[i] for i in selected_indices])
    balanced_labels = np.array([labels[i] for i in selected_indices])
    
    return balanced_features, balanced_labels

def data_generator(file_list):
    tdg = JavaTDG()
    for file_path in file_list:
        process_file(file_path, tdg)
    features, labels, node_ids = preprocess_tdg(tdg)
    features, labels = balance_dataset(features, labels)
    for feature, label, node_id in zip(features, labels, node_ids):
        yield feature, label, node_id

def create_tf_dataset(file_list, batch_size):
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(file_list),
        output_signature=(
            tf.TensorSpec(shape=(4,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.string),  # Use string for node_id to keep full identifier
        )
    )
    dataset = dataset.shuffle(buffer_size=10000).batch(batch_size)
    return dataset

def process_file(file_path, tdg):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        if not content.strip():
            logging.warning(f"File {file_path} is empty, skipping.")
            return

        tree = javalang.parse.parse(content)
        file_name = os.path.basename(file_path)
        logging.info(f"Processing file {file_path}")

        file_id = file_name
        tdg.add_node(file_id, "file", file_name)

        for path, node in tree:
            if isinstance(node, javalang.tree.ClassDeclaration):
                class_id = f"{file_name}.{node.name}"
                tdg.add_node(class_id, "class", node.name)
                tdg.add_classname(node.name)
                tdg.add_edge(file_id, class_id, "contains")
                for method in node.methods:
                    method_id = f"{class_id}.{method.name}()"
                    tdg.add_node(method_id, "method", method.name)
                    tdg.add_edge(class_id, method_id, "contains")
                    for param in method.parameters:
                        param_id = f"{method_id}.{param.name}"
                        actual_type = get_actual_type(param)
                        tdg.add_node(param_id, "parameter", param.name, actual_type=actual_type)
                        tdg.add_edge(method_id, param_id, "has_parameter")
                for field in node.fields:
                    for decl in field.declarators:
                        field_id = f"{class_id}.{decl.name}"
                        actual_type = get_actual_type(decl)
                        tdg.add_node(field_id, "field", decl.name, actual_type=actual_type)
                        tdg.add_edge(class_id, field_id, "has_field")
            elif isinstance(node, javalang.tree.MethodDeclaration):
                method_id = f"{file_name}.{node.name}()"
                tdg.add_node(method_id, "method", node.name)
                for param in node.parameters:
                    param_id = f"{method_id}.{param.name}"
                    actual_type = get_actual_type(param)
                    tdg.add_node(param_id, "parameter", param.name, actual_type=actual_type)
                    tdg.add_edge(method_id, param_id, "has_parameter")
            elif isinstance(node, javalang.tree.FieldDeclaration):
                for decl in node.declarators:
                    field_id = f"{file_name}.{decl.name}"
                    actual_type = get_actual_type(decl)
                    tdg.add_node(field_id, "field", decl.name, actual_type=actual_type)
                    tdg.add_edge(file_name, field_id, "has_field")
            elif isinstance(node, javalang.tree.VariableDeclarator):
                var_id = f"{file_name}.{node.name}"
                actual_type = get_actual_type(node)
                tdg.add_node(var_id, "variable", node.name, actual_type=actual_type)
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
