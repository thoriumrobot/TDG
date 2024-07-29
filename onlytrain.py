import os
import sys
import json
import javalang
import networkx as nx
from collections import defaultdict
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint
import logging
import traceback

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

    def to_json(self, file_path):
        data = {
            "nodes": [{"id": n, "attr": self.graph.nodes[n]} for n in self.graph.nodes()],
            "edges": [{"from": u, "to": v, "attr": self.graph.edges[u, v]} for u, v in self.graph.edges()],
            "classnames": list(self.classnames),
            "aliasgraph": [{"from": u, "to": v, "attr": self.aliasgraph.edges[u, v]} for u, v in self.aliasgraph.edges()],
            "callgraph": dict(self.callgraph)
        }
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

def alias_analysis(tdg, project_dir):
    logging.info("[1st Pass: Alias Analysis] Started...")
    for root, _, files in os.walk(project_dir):
        for file in files:
            if file.endswith('.java'):
                try:
                    with open(os.path.join(root, file), 'r') as f:
                        content = f.read()
                    tree = javalang.parse.parse(content)
                    file_name = os.path.basename(file)

                    for path, node in tree:
                        if isinstance(node, javalang.tree.VariableDeclarator):
                            var_name = node.name
                            if isinstance(node.initializer, javalang.tree.MemberReference):
                                alias_name = node.initializer.member
                                var_id = f"{file_name}.{var_name}"
                                alias_id = f"{file_name}.{alias_name}"
                                tdg.add_alias(alias_id, var_id)
                            elif isinstance(node.initializer, javalang.tree.Literal):
                                pass
                            elif isinstance(node.initializer, javalang.tree.MethodInvocation):
                                pass
                            else:
                                pass
                except Exception as e:
                    logging.error(f"Error processing file {file}: {e}")
                    logging.error(traceback.format_exc())
    logging.info("[1st Pass: Alias Analysis] Finished...")

def call_analysis(tdg, project_dir):
    logging.info("[2nd Pass: Call Analysis] Started...")
    for root, _, files in os.walk(project_dir):
        for file in files:
            if file.endswith(".java"):
                try:
                    with open(os.path.join(root, file), 'r') as f:
                        content = f.read()
                    tree = javalang.parse.parse(content)
                    for path, node in tree:
                        if isinstance(node, javalang.tree.MethodInvocation):
                            caller = ".".join(str(p) for p in path if isinstance(p, javalang.tree.MethodDeclaration))
                            callee = node.member
                            tdg.callgraph[caller].append(callee)
                except Exception as e:
                    logging.error(f"Error processing file {file}: {e}")
                    logging.error(traceback.format_exc())
    logging.info(f"[2nd Pass: Call Analysis] Captured {sum(len(v) for v in tdg.callgraph.values())} call relationships.")
    logging.info("[2nd Pass: Call Analysis] Finished...")

def load_tdg_data(json_dir):
    data = []
    for file_name in os.listdir(json_dir):
        if file_name.endswith('.json'):
            with open(os.path.join(json_dir, file_name), 'r') as f:
                data.append(json.load(f))
    return data

def preprocess_data(tdg_data):
    features = []
    labels = []
    for tdg in tdg_data:
        for node in tdg['nodes']:
            feature_vector = extract_features(node)
            label = get_label(node)
            features.append(feature_vector)
            labels.append(label)
    return features, labels

def extract_features(node):
    return [node['attr'].get('type', 0), node['attr'].get('name', 0)]

def get_label(node):
    return node['attr'].get('nullable', 0)

def build_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[checkpoint])
    return history

def main(class_dirs, json_output_dir, model_output_path):
    tdg_data = load_tdg_data(json_output_dir)
    features, labels = preprocess_data(tdg_data)
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
    input_dim = len(features[0])
    model = build_model(input_dim)
    train_model(model, X_train, y_train, X_val, y_val)
    best_model = load_model('best_model.keras')
    best_model.save(model_output_path)
    logging.info(f"Model training complete and saved as {model_output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python onlytrain.py <JsonOutputDir> <ModelOutputPath> <ClassDirs...>")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)

    json_output_dir = sys.argv[1]
    model_output_path = sys.argv[2]
    class_dirs = sys.argv[3:]

    main(class_dirs, json_output_dir, model_output_path)

