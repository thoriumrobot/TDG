import json
import networkx as nx
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import random
import logging
import os
import javalang
import traceback

class JavaTDG:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.classnames = set()

    def add_node(self, node_id, node_type, name, line_number=None, nullable=False, actual_type=None):
        self.graph.add_node(node_id, attr={'type': node_type, 'name': name, 'line_number': line_number, 'nullable': nullable, 'actual_type': actual_type})

    def add_edge(self, from_node, to_node, edge_type):
        self.graph.add_edge(from_node, to_node, type=edge_type)
        self.graph.add_edge(to_node, from_node, type=f"reverse_{edge_type}")

    def add_classname(self, classname):
        self.classnames.add(classname)

def f1_score(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + K.epsilon())

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        return true_positives / (predicted_positives + K.epsilon())

    precision_val = precision(y_true, y_pred)
    recall_val = recall(y_true, y_pred)
    return 2 * ((precision_val * recall_val) / (precision_val + recall_val + K.epsilon()))

class NodeIDMapper:
    def __init__(self):
        self.id_to_int = {}
        self.int_to_id = {}
        self.counter = 0

    def get_int(self, node_id):
        if node_id not in self.id_to_int:
            self.id_to_int[node_id] = self.counter
            self.int_to_id[self.counter] = node_id
            self.counter += 1
        return self.id_to_int[node_id]

    def get_id(self, node_int):
        return self.int_to_id.get(node_int, None)

node_id_mapper = NodeIDMapper()

# Define mappings globally
type_mapping = {'class': 0, 'method': 1, 'field': 2, 'parameter': 3, 'variable': 4, 'literal': 5}

def extract_features(attr):
    if attr is None:
        logging.warning("Encountered NoneType for attr. Using default values.")
        return [0.0, 0.0]  # Default feature vector with type_id and nullable

    node_type = attr.get('type', '')
    nullable = float(attr.get('nullable', 0))

    type_id = type_mapping.get(node_type, len(type_mapping))

    return [float(type_id), nullable]

def load_tdg_data(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        tdg = JavaTDG()
        tdg.graph = nx.node_link_graph(data)
        return preprocess_tdg(tdg)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {json_path}: {e}")
        return ([], [], [], [], [])  # Return empty placeholders if there's an error
    except Exception as e:
        logging.error(f"Error processing {json_path}: {e}")
        return ([], [], [], [], [])  # Handle other errors similarly

def balance_dataset(features, labels, node_ids, adjacency_matrix):
    pos_indices = [i for i, label in enumerate(labels) if label == 1]
    neg_indices = [i for i, label in enumerate(labels) if label == 0]

    if len(pos_indices) == 0 or len(neg_indices) == 0:
        logging.warning("Cannot balance dataset with no positive or negative examples.")
        return features, labels, node_ids, adjacency_matrix

    random.shuffle(neg_indices)
    selected_neg_indices = neg_indices[:len(pos_indices)]

    selected_indices = pos_indices + selected_neg_indices
    random.shuffle(selected_indices)

    selected_features = features[selected_indices]
    selected_labels = labels[selected_indices]
    selected_node_ids = node_ids[selected_indices]
    selected_adjacency_matrix = adjacency_matrix[selected_indices, :][:, selected_indices]

    return selected_features, selected_labels, selected_node_ids, selected_adjacency_matrix

def preprocess_tdg(tdg):
    features = []
    labels = []
    node_ids = []
    prediction_node_ids = []
    all_node_ids = list(tdg.graph.nodes)

    if len(all_node_ids) == 0:
        return np.zeros((1, 2)), np.zeros((1,)), np.zeros((1,)), np.zeros((1, 1)), []

    node_id_map = {}  # Map node IDs to indices
    for idx, node_id in enumerate(all_node_ids):
        node_id_map[node_id] = idx

    num_nodes = len(all_node_ids)
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for node_id in all_node_ids:
        attr = tdg.graph.nodes[node_id].get('attr', {})
        feature_vector = extract_features(attr)
        features.append(feature_vector)
        node_index = node_id_map[node_id]
        node_ids.append(node_index)

        # Map node IDs to indices for consistent mapping
        node_id_mapper.get_int(node_id)  # Ensure the node ID is mapped

        if attr.get('type') in ['method', 'field', 'parameter']:
            labels.append(float(attr.get('nullable', 0)))
            prediction_node_ids.append(node_index)

    for from_node, to_node in tdg.graph.edges():
        from_idx = node_id_map.get(from_node)
        to_idx = node_id_map.get(to_node)
        if from_idx is not None and to_idx is not None:
            adjacency_matrix[from_idx, to_idx] = 1.0

    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    node_ids = np.array(node_ids, dtype=np.int32)

    if features.size == 0 or adjacency_matrix.size == 0:
        logging.warning("Skipping empty or invalid graph.")
        return np.zeros((1, 2)), np.zeros((1,)), np.zeros((1,)), np.zeros((1, 1)), []

    return features, labels, node_ids, adjacency_matrix, prediction_node_ids

def data_generator(file_list, balance=False, is_tdg=True):
    if is_tdg:
        # Training: Process pre-extracted graphs
        for file_path in file_list:
            try:
                result = load_tdg_data(file_path)
                if len(result) != 5:
                    logging.error(f"Graph from {file_path} returned {len(result)} values. Expected 5. Skipping this graph.")
                    continue

                features, labels, node_ids, adjacency_matrix, prediction_node_ids = result

                features = np.array(features, dtype=np.float32)
                adjacency_matrix = np.array(adjacency_matrix, dtype=np.float32)

                if features.size == 0 or adjacency_matrix.size == 0:
                    logging.warning(f"Skipping empty or invalid graph in file: {file_path}")
                    continue

                if balance:
                    features, labels, prediction_node_ids, adjacency_matrix = balance_dataset(features, labels, prediction_node_ids, adjacency_matrix)

                yield (features, adjacency_matrix, prediction_node_ids), labels
            except Exception as e:
                logging.error(f"Error processing graph in file {file_path}: {e}")
                continue
    else:
        # Prediction logic (if needed)
        pass

def create_tf_dataset(file_list, batch_size, balance=False, is_tdg=True):
    def generator():
        for (features, adjacency_matrix, prediction_node_ids), labels in data_generator(file_list, balance, is_tdg):
            features = np.array(features, dtype=np.float32)
            adjacency_matrix = np.array(adjacency_matrix, dtype=np.float32)
            labels = np.array(labels, dtype=np.float32)

            prediction_mask = np.zeros(features.shape[0], dtype=bool)
            prediction_mask[prediction_node_ids] = True

            # Extract labels corresponding to the prediction_mask
            all_labels = np.zeros((features.shape[0], 1), dtype=np.float32)
            all_labels[prediction_node_ids] = labels.reshape(-1, 1)
            masked_labels = all_labels[prediction_mask]

            yield (features, adjacency_matrix, prediction_mask), masked_labels

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            (tf.TensorSpec(shape=(None, 2), dtype=tf.float32),  # Node features
             tf.TensorSpec(shape=(None, None), dtype=tf.float32),  # Adjacency matrix
             tf.TensorSpec(shape=(None,), dtype=tf.bool)),  # Prediction mask
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32)  # Labels
        )
    )

    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=(
            (tf.TensorShape([None, 2]),  # Node features
             tf.TensorShape([None, None]),  # Adjacency matrix
             tf.TensorShape([None])),  # Prediction mask
            tf.TensorShape([None, 1])  # Labels
        ),
        padding_values=(
            (tf.constant(0.0),  # Padding value for features
             tf.constant(0.0),  # Padding value for adjacency matrix
             tf.constant(False)),  # Padding value for prediction mask
            tf.constant(0.0)  # Padding value for labels
        )
    )
    return dataset

def has_nullable_annotation(annotations):
    return any(annotation.name == 'Nullable' for annotation in annotations)

def get_actual_type(node):
    if hasattr(node, 'type') and hasattr(node.type, 'name'):
        return node.type.name
    return None

def get_superclass_name(node):
    """
    Extracts the superclass name from a class declaration node, if present.
    """
    if node.extends:
        return node.extends.name
    return None

def process_field_declaration(field, class_id, tdg):
    """
    Processes field declarations and connects them to the TDG.
    """
    for decl in field.declarators:
        field_id = f"{class_id}.{decl.name}"
        line_number = field.position.line if field.position else None
        actual_type = get_actual_type(decl)
        nullable = has_nullable_annotation(field.annotations)
        
        # Add the field to the TDG
        tdg.add_node(field_id, "field", decl.name, line_number=line_number, actual_type=actual_type, nullable=nullable)
        tdg.add_edge(class_id, field_id, "has_field")

        # Handle assignment to field via method call (e.g., field initialization)
        if isinstance(decl.initializer, javalang.tree.MethodInvocation):
            method_call_id = f"{class_id}.{decl.initializer.member}()"
            tdg.add_edge(field_id, method_call_id, "assigned_from_method")

def process_method_invocation(method_id, class_id, method_invocation, tdg):
    """
    Handles method invocations, ensuring they are correctly linked to the TDG.
    """
    called_method_id = f"{class_id}.{method_invocation.member}()"
    tdg.add_edge(method_id, called_method_id, "calls")
    return called_method_id

def process_expression(expression, method_id, class_id, tdg):
    """
    Recursively processes expressions to extract method invocations and variable references.
    """
    # If the expression is a method invocation
    if isinstance(expression, javalang.tree.MethodInvocation):
        method_call_id = process_method_invocation(method_id, class_id, expression, tdg)
        return method_call_id

    # If the expression is a member reference (i.e., a variable)
    if isinstance(expression, javalang.tree.MemberReference):
        referenced_var_id = f"{method_id}.{expression.member}"
        return referenced_var_id

    # Recursively process binary operations (e.g., x + y)
    if isinstance(expression, javalang.tree.BinaryOperation):
        left_result = process_expression(expression.operandl, method_id, class_id, tdg)
        right_result = process_expression(expression.operandr, method_id, class_id, tdg)
        return left_result, right_result

    return None

def process_assignment(statement, method_id, class_id, tdg):
    """
    Processes assignments to variables, handling complex expressions involving method calls or other variables.
    """
    if isinstance(statement, javalang.tree.Assignment):
        assigned_var_id = f"{method_id}.{statement.left.name}"

        # Process the right-hand expression of the assignment (may contain method calls or variables)
        results = process_expression(statement.expression, method_id, class_id, tdg)
        
        # Handle results from expressions (could be method calls or variable references)
        if isinstance(results, tuple):  # If both left and right are processed (binary operations)
            for result in results:
                if result:
                    tdg.add_edge(assigned_var_id, result, "assigned_from_expression")
        elif results:
            tdg.add_edge(assigned_var_id, results, "assigned_from_expression")

        return assigned_var_id
    return None

def process_java_file(file_path, tdg):
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
                line_number = node.position.line if node.position else None
                tdg.add_node(class_id, "class", node.name, line_number=line_number)
                tdg.add_classname(node.name)
                tdg.add_edge(file_id, class_id, "contains")

                # Process each method in the class
                for method in node.methods:
                    method_id = f"{class_id}.{method.name}()"
                    line_number = method.position.line if method.position else None
                    nullable = has_nullable_annotation(method.annotations)
                    tdg.add_node(method_id, "method", method.name, line_number=line_number, nullable=nullable)
                    tdg.add_edge(class_id, method_id, "contains")

                    # Check for overridden methods (inheritance)
                    if any(annotation.name == "Override" for annotation in method.annotations):
                        superclass_name = get_superclass_name(node)
                        if superclass_name:
                            superclass_method_id = f"{superclass_name}.{method.name}()"
                            tdg.add_edge(method_id, superclass_method_id, "overrides")

                    # Add method return value as a node
                    return_id = f"{method_id}.return"
                    tdg.add_node(return_id, "return", "return_value", line_number=line_number)
                    tdg.add_edge(method_id, return_id, "has_return")

                    # Add method parameters and variables
                    for param in method.parameters:
                        param_id = f"{method_id}.{param.name}"
                        line_number = param.position.line if param.position else None
                        actual_type = get_actual_type(param)
                        nullable = has_nullable_annotation(param.annotations)
                        tdg.add_node(param_id, "parameter", param.name, line_number=line_number, actual_type=actual_type, nullable=nullable)
                        tdg.add_edge(method_id, param_id, "has_parameter")

                    # Process method body statements (assignments and standalone method calls)
                    for statement in method.body:
                        # Handle standalone method calls
                        if isinstance(statement, javalang.tree.MethodInvocation):
                            process_method_invocation(method_id, class_id, statement, tdg)

                        # Handle variable assignments (with method calls or variables)
                        process_assignment(statement, method_id, class_id, tdg)

                    # Add variables used in the method as nodes and connect them
                    for local_var in method.body:
                        if isinstance(local_var, javalang.tree.VariableDeclarator):
                            var_id = f"{method_id}.{local_var.name}"
                            line_number = local_var.position.line if local_var.position else None
                            actual_type = get_actual_type(local_var)
                            tdg.add_node(var_id, "variable", local_var.name, line_number=line_number, actual_type=actual_type)
                            tdg.add_edge(method_id, var_id, "has_variable")

                # Process field declarations
                for field in node.fields:
                    process_field_declaration(field, class_id, tdg)

            # Handle top-level method declarations
            elif isinstance(node, javalang.tree.MethodDeclaration):
                method_id = f"{file_name}.{node.name}()"
                line_number = node.position.line if node.position else None
                tdg.add_node(method_id, "method", node.name, line_number=line_number)
                for param in node.parameters:
                    param_id = f"{method_id}.{param.name}"
                    line_number = param.position.line if param.position else None
                    actual_type = get_actual_type(param)
                    nullable = has_nullable_annotation(param.annotations)
                    tdg.add_node(param_id, "parameter", param.name, line_number=line_number, actual_type=actual_type, nullable=nullable)
                    tdg.add_edge(method_id, param_id, "has_parameter")

            # Handle field declarations at the top level
            elif isinstance(node, javalang.tree.FieldDeclaration):
                process_field_declaration(node, file_name, tdg)

            # Handle variables and null literals
            elif isinstance(node, javalang.tree.VariableDeclarator):
                var_id = f"{file_name}.{node.name}"
                line_number = node.position.line if node.position else None
                actual_type = get_actual_type(node)
                tdg.add_node(var_id, "variable", node.name, line_number=line_number, actual_type=actual_type)

            elif isinstance(node, javalang.tree.Literal) and node.value == "null":
                if node.position:
                    null_id = f"{file_name}.null_{node.position.line}_{node.position.column}"
                    tdg.add_node(null_id, "literal", "null", line_number=node.position.line)
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
