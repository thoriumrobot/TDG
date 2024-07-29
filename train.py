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
import numpy as np

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

def process_file(file_path, tdg):
    try:
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

def save_tdg_to_json(tdg, output_dir, class_name):
    os.makedirs(output_dir, exist_ok=True)
    tdg_file_path = os.path.join(output_dir, f"{class_name}_tdg.json")
    tdg.to_json(tdg_file_path)
    logging.info(f"TDG saved to {tdg_file_path}")
    return tdg_file_path

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
    return np.array(features), np.array(labels)

def extract_features(node):
    return [float(node['attr'].get('type', 0)), float(node['attr'].get('name', 0))]

def get_label(node):
    return float(node['attr'].get('nullable', 0))

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
    # Ensure the inputs are numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    
    # Print debug information
    print(f"X_train type: {type(X_train)}, shape: {X_train.shape}")
    print(f"y_train type: {type(y_train)}, shape: {y_train.shape}")
    print(f"X_val type: {type(X_val)}, shape: {X_val.shape}")
    print(f"y_val type: {type(y_val)}, shape: {y_val.shape}")

    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[checkpoint])
    return history

def main(class_dirs, json_output_dir, model_output_path):
    for class_dir in class_dirs:
        for root, _, files in os.walk(class_dir):
            for file in files:
                if file.endswith('.java'):
                    class_name = os.path.splitext(file)[0]
                    file_path = os.path.join(root, file)
                    tdg = JavaTDG()
                    process_file(file_path, tdg)
                    save_tdg_to_json(tdg, json_output_dir, class_name)

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
        print("Usage: python train.py <JsonOutputDir> <ModelOutputPath> <ClassDirs...>")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)

    json_output_dir = sys.argv[1]
    model_output_path = sys.argv[2]
    class_dirs = sys.argv[3:]

    main(class_dirs, json_output_dir, model_output_path)

