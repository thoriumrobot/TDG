import json
import networkx as nx
from collections import defaultdict
from tensorflow.keras import backend as K
import tensorflow as tf
import random

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
    for node_id, attr in tdg.graph.nodes(data='attr'):
        if attr and attr.get('type') in ['method', 'field', 'parameter']:
            feature_vector = extract_features(attr)
            label = float(attr.get('nullable', 0))
            features.append(feature_vector)
            labels.append(label)
    return np.array(features), np.array(labels)

def load_tdg_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    tdg = JavaTDG()
    tdg.graph = nx.node_link_graph(data)
    return preprocess_tdg(tdg)

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
    for file_path in file_list:
        features, labels = load_tdg_data(file_path)
        features, labels = balance_dataset(features, labels)
        for feature, label in zip(features, labels):
            yield feature, label

def create_tf_dataset(file_list, batch_size):
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(file_list),
        output_signature=(
            tf.TensorSpec(shape=(4,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        )
    )
    dataset = dataset.shuffle(buffer_size=10000).batch(batch_size)
    return dataset
