import json
import networkx as nx
from collections import defaultdict
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import random
import logging
import os
import javalang
import traceback
import pdb

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

def extract_features(attr):
    if attr is None:
        # Handle the None case by providing default values
        logging.warning("Encountered NoneType for attr. Using default values.")
        return [0.0, 0.0, 0.0, 0.0]  # Default feature vector

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

    random.shuffle(neg_indices)
    selected_neg_indices = neg_indices[:len(pos_indices)]

    selected_indices = pos_indices + selected_neg_indices
    random.shuffle(selected_indices)

    # Create subgraph with selected indices
    selected_features = features[selected_indices]
    selected_labels = labels[selected_indices]
    #selected_node_ids = node_ids[selected_indices]
    selected_adjacency_matrix = adjacency_matrix[selected_indices, :][:, selected_indices]

    # Renumber the nodes in the selected subgraph
    selected_features, selected_labels, selected_node_ids, selected_adjacency_matrix = renumber_and_prune(selected_features, selected_labels, selected_adjacency_matrix)

    return selected_features, selected_labels, selected_node_ids, selected_adjacency_matrix

def renumber_and_prune(features, labels, adjacency_matrix):
    """
    Renumber nodes in a connected subgraph and prune any disconnected nodes.
    """
    # Convert adjacency matrix to a graph object
    graph = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
    
    # Check if the graph is empty or has no connected components
    if len(graph) == 0 or not list(nx.weakly_connected_components(graph)):
        # Return empty arrays if the graph has no connected components
        return np.array([]), np.array([]), np.array([]), np.array([[]])

    # Find the largest connected component
    largest_cc = max(nx.weakly_connected_components(graph), key=len)

    # Create a mapping from old to new node IDs
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(largest_cc)}

    # Apply mapping to the features, labels, and adjacency matrix
    pruned_features = np.array([features[old_to_new[i]] for i in sorted(largest_cc)])
    pruned_labels = np.array([labels[old_to_new[i]] for i in sorted(largest_cc)])
    pruned_adjacency_matrix = np.zeros((len(largest_cc), len(largest_cc)))

    for i, old_id in enumerate(sorted(largest_cc)):
        for j, old_id2 in enumerate(sorted(largest_cc)):
            pruned_adjacency_matrix[i, j] = adjacency_matrix[old_id, old_id2]

    return pruned_features, pruned_labels, np.array(list(range(len(largest_cc)))), pruned_adjacency_matrix

def preprocess_tdg(tdg):
    features = []
    labels = []
    node_ids = []
    prediction_node_ids = []  # Track node IDs for prediction
    all_node_ids = set(tdg.graph.nodes)  # Track all nodes

    node_id_map = {node: idx for idx, node in enumerate(all_node_ids)}  # Map all nodes

    num_valid_nodes = len(all_node_ids)
    if num_valid_nodes == 0:
        return np.zeros((1, 4)), np.zeros((1,)), np.zeros((1,)), np.zeros((1, 1)), []

    adjacency_matrix = np.zeros((num_valid_nodes, num_valid_nodes), dtype=np.float32)

    for node_id, attr in tdg.graph.nodes(data='attr'):
        # Check if attr is None and handle it
        if attr is None:
            logging.warning(f"Node {node_id} has no attributes. Skipping.")
            continue

        feature_vector = extract_features(attr)
        node_index = node_id_map[node_id]
        features.append(feature_vector)
        node_ids.append(node_index)

        if attr.get('type') in ['method', 'field', 'parameter']:
            prediction_node_ids.append(node_index)
            labels.append(float(attr.get('nullable', 0)))

        # Map all nodes regardless of their type
        node_id_mapper.get_int(node_id)  # Ensure the node ID is mapped

    for from_node, to_node in tdg.graph.edges():
        if from_node in all_node_ids and to_node in all_node_ids:
            from_id = node_id_map[from_node]
            to_id = node_id_map[to_node]
            adjacency_matrix[from_id, to_id] = 1.0
    
    features = np.array(features, dtype=np.float32)
    adjacency_matrix = np.array(adjacency_matrix, dtype=np.float32)

    if features.size == 0 or adjacency_matrix.size == 0:
        logging.warning("Skipping empty or invalid graph.")
        return np.zeros((1, 4)), np.zeros((1,)), np.zeros((1,)), np.zeros((1, 1)), []

    return features, np.array(labels, dtype=np.float32), np.array(node_ids, dtype=np.int32), adjacency_matrix, prediction_node_ids

def data_generator(file_list, balance=False, is_tdg=True, max_nodes=8000):
    graphs = []
    
    if is_tdg:
        # Training: Process pre-extracted graphs
        for file_path in file_list:
            try:
                result = load_tdg_data(file_path)
                if len(result) != 5:  # Adjust for the new return value
                    logging.error(f"Graph from {file_path} returned {len(result)} values. Expected 5. Skipping this graph.")
                    continue

                features, labels, node_ids, adjacency_matrix, prediction_node_ids = result

                features = np.array(features, dtype=np.float32)
                adjacency_matrix = np.array(adjacency_matrix, dtype=np.float32)
                
                # Skip if the graph is empty or invalid
                if features.size == 0 or adjacency_matrix.size == 0:
                    logging.warning(f"Skipping empty or invalid graph in file: {file_path}")
                    continue
                
                if balance:
                    features, labels, prediction_node_ids, adjacency_matrix = balance_dataset(features, labels, prediction_node_ids, adjacency_matrix)

                graphs.append((features, labels, node_ids, adjacency_matrix, prediction_node_ids))
            except Exception as e:
                logging.error(f"Error processing graph in file {file_path}: {e}")
                continue

    else:
        # Prediction: Combine all Java source code into a single graph
        tdg = JavaTDG()
        for file_path in file_list:
            process_java_file(file_path, tdg)

        # Preprocess the combined graph
        try:
            result = preprocess_tdg(tdg)
            if len(result) != 5:
                logging.error(f"Combined graph returned {len(result)} values. Expected 5. Skipping this graph.")
                return
            
            features, labels, node_ids, adjacency_matrix, prediction_node_ids = map(np.array, result)
            
            if features.size == 0 or adjacency_matrix.size == 0:
                logging.warning(f"Skipping empty or invalid graph in file: {file_path}")
                return
            graphs.append((features, labels, node_ids, adjacency_matrix, prediction_node_ids))
        except Exception as e:
            logging.error(f"Error processing combined graph: {e}")
            return

    # Accumulate and split graphs into batches of max_nodes
    for padded_features, padded_labels, padded_node_ids, padded_adj_matrix, prediction_node_ids in accumulate_and_split_graphs(graphs, max_nodes):
        yield (padded_features, padded_adj_matrix), (padded_labels, prediction_node_ids)

def accumulate_and_split_graphs(graphs, max_nodes=8000):
    accumulated_features = []
    accumulated_labels = []
    accumulated_node_ids = []
    accumulated_adj_matrix = []
    accumulated_prediction_node_ids = []

    current_node_count = 0

    for features, labels, node_ids, adjacency_matrix, prediction_node_ids in graphs:
        num_nodes = features.shape[0]

        # Skip empty or invalid graphs
        if num_nodes == 0:
            continue

        # If adding this graph would exceed max_nodes, pad the current batch and start a new one
        if current_node_count + num_nodes > max_nodes:
            yield pad_batch(accumulated_features, accumulated_labels, accumulated_node_ids, accumulated_adj_matrix, accumulated_prediction_node_ids, max_nodes)

            # Reset accumulation
            accumulated_features = []
            accumulated_labels = []
            accumulated_node_ids = []
            accumulated_adj_matrix = []
            accumulated_prediction_node_ids = []
            current_node_count = 0

        # Add the current graph to the accumulation
        accumulated_features.append(features)
        accumulated_labels.append(labels)
        accumulated_node_ids.append(node_ids)
        accumulated_adj_matrix.append(adjacency_matrix)
        accumulated_prediction_node_ids.append(prediction_node_ids)
        current_node_count += num_nodes

    # Yield any remaining accumulated graphs as the final batch
    if accumulated_features:
        yield pad_batch(accumulated_features, accumulated_labels, accumulated_node_ids, accumulated_adj_matrix, accumulated_prediction_node_ids, max_nodes)

def pad_batch(features, labels, node_ids, adjacency_matrix, prediction_node_ids, max_nodes):
    feature_dim = 4

    # Initialize padded arrays
    padded_features = np.zeros((max_nodes, feature_dim), dtype=np.float32)
    padded_labels = np.zeros((max_nodes,), dtype=np.float32)
    padded_node_ids = np.zeros((max_nodes,), dtype=np.int32)
    padded_adj_matrix = np.zeros((max_nodes, max_nodes), dtype=np.float32)

    # Combine features, labels, node_ids, and adjacency matrices correctly
    combined_features = np.concatenate(features, axis=0)
    combined_labels = np.concatenate(labels, axis=0)
    combined_node_ids = np.concatenate(node_ids, axis=0)

    # Ensure combined features are within max_nodes
    num_nodes = min(combined_features.shape[0], max_nodes)

    # Adjust adjacency matrix
    combined_adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    offset = 0
    for adj in adjacency_matrix:
        size = adj.shape[0]
        if offset + size > max_nodes:
            break  # Stop if adding this adjacency matrix would exceed max_nodes
        combined_adj_matrix[offset:offset+size, offset:offset+size] = adj
        offset += size

    # Apply the padding to the final batch
    padded_features[:num_nodes, :] = combined_features[:num_nodes]
    padded_labels[:num_nodes] = combined_labels[:num_nodes]
    padded_node_ids[:num_nodes] = combined_node_ids[:num_nodes]
    padded_adj_matrix[:num_nodes, :num_nodes] = combined_adj_matrix

    return padded_features, padded_labels, padded_node_ids, padded_adj_matrix, prediction_node_ids

def create_tf_dataset(file_list, batch_size, balance=False, is_tdg=True):
    def generator():
        for (features, adjacency_matrix), (labels, prediction_node_ids) in data_generator(file_list, balance, is_tdg):
            features = np.array(features, dtype=np.float32)
            adjacency_matrix = np.array(adjacency_matrix, dtype=np.float32)
            labels = np.array(labels, dtype=np.float32)

            # Ensure labels have the correct shape
            if len(labels.shape) == 1:
                labels = np.expand_dims(labels, -1)

            # Ensure prediction_node_ids is a fixed-size 2D array, pad if necessary
            max_prediction_nodes = 100  # Define a fixed length for prediction nodes
            padded_prediction_node_ids = []
            
            for node_ids in prediction_node_ids:
                if len(node_ids) > max_prediction_nodes:
                    node_ids = node_ids[:max_prediction_nodes]  # Truncate
                else:
                    padding_length = max_prediction_nodes - len(node_ids)
                    node_ids = np.pad(node_ids, (0, padding_length), 'constant', constant_values=-1)  # Pad with -1
                padded_prediction_node_ids.append(node_ids)
            
            prediction_node_ids = np.array(padded_prediction_node_ids, dtype=np.int32)

            # Generate the prediction mask
            prediction_mask = np.zeros(features.shape[0], dtype=bool)
            prediction_mask[prediction_node_ids] = True
            
            if features.shape[-1] != 4 or len(adjacency_matrix.shape) != 2:
                raise ValueError(f"Unexpected shapes: features shape {features.shape}, adjacency_matrix shape {adjacency_matrix.shape}, labels shape {labels.shape}")
            
            yield (features, adjacency_matrix, prediction_mask), labels

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            (tf.TensorSpec(shape=(None, 4), dtype=tf.float32),  # Node features
             tf.TensorSpec(shape=(None, None), dtype=tf.float32),  # Adjacency matrix
             tf.TensorSpec(shape=(None,), dtype=tf.bool)),  # Prediction mask
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32)  # Labels
        )
    )

    dataset = dataset.shuffle(buffer_size=10000).padded_batch(
        batch_size, 
        padded_shapes=(
            (tf.TensorShape([None, 4]),  # Node features
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
