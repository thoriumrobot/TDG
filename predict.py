import os
import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import logging
import tensorflow as tf
from tdg_utils import f1_score, preprocess_tdg, process_java_file, NodeIDMapper, node_id_mapper, JavaTDG
import networkx as nx

@tf.keras.utils.register_keras_serializable()
class BooleanMaskLayer(Layer):
    def call(self, inputs):
        output, mask = inputs
        return tf.boolean_mask(output, mask)

    def compute_output_shape(self, input_shape):
        output_shape, mask_shape = input_shape
        return (None, output_shape[-1])

def annotate_file(file_path, annotations, output_file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    for annotation in annotations:
        file_name, node_name, line_num = annotation
        
        if line_num is None:
            logging.warning(f"Line number is None for node {node_name} in file {file_path}")
            continue
        
        line = lines[line_num - 1]
        
        if 0 <= line_num - 1 < len(lines) and "@Nullable {node_name}" not in line:
            lines[line_num - 1] = line.replace(node_name, f"@Nullable {node_name}")
        else:
            logging.warning(f"Line number {line_num} is out of range in file {file_path}")
    
    with open(output_file_path, 'w') as file:
        file.writelines(lines)

def create_combined_tdg(file_list):
    combined_tdg = JavaTDG()
    file_mappings = {}  # Map node IDs to file paths
    line_number_mapping = {}  # Track line numbers for nodes across files
    
    for file_path in file_list:
        class_tdg = JavaTDG()
        process_java_file(file_path, class_tdg)
        
        for node_id, node_data in class_tdg.graph.nodes(data=True):
            #if node_data['attr'].get('line_number') is None:
                #continue
            
            # Normalize node_id by stripping the file-specific prefix
            combined_node_id = normalize_node_id(node_id)
            
            if 'attr' not in node_data:
                logging.warning(f"Node {node_id} is missing 'attr'. Skipping.")
                continue
            
            combined_tdg.add_node(
                combined_node_id, 
                node_data['attr']['type'], 
                node_data['attr']['name'], 
                line_number=node_data['attr'].get('line_number'), 
                nullable=node_data['attr'].get('nullable'), 
                actual_type=node_data['attr'].get('actual_type')
            )
            
            # Maintain the original file mapping and line numbers
            file_mappings[combined_node_id] = file_path
            line_number_mapping[combined_node_id] = node_data['attr'].get('line_number')
        
        for from_node, to_node, edge_data in class_tdg.graph.edges(data=True):
            combined_tdg.add_edge(normalize_node_id(from_node), normalize_node_id(to_node), edge_data['type'])
    
    return combined_tdg, file_mappings, line_number_mapping

def normalize_node_id(node_id):
    # Remove the file-specific part of the node_id to create a unified identifier
    parts = node_id.split('.')
    return '.'.join(parts[2:])  # This assumes the first part is the file-specific prefix

def process_project(project_dir, output_dir, model, batch_size):
    file_list = [os.path.join(root, file)
                 for root, _, files in os.walk(project_dir)
                 for file in files if file.endswith('.java')]

    combined_tdg, file_mappings, line_number_mapping = create_combined_tdg(file_list)
    features, _, node_ids, adjacency_matrix, prediction_node_ids = preprocess_tdg(combined_tdg)

    if features.size == 0 or adjacency_matrix.size == 0:
        logging.warning(f"No valid TDG created for project {project_dir}. Skipping.")
        return
    
    num_nodes = features.shape[0]
    max_nodes = 8000
    feature_dim = features.shape[1] if features.ndim > 1 else 4
    
    if num_nodes > max_nodes:
        logging.warning(f"Number of nodes ({num_nodes}) exceeds max_nodes ({max_nodes}). Truncating the graph.")
        num_nodes = max_nodes
        adjacency_matrix = adjacency_matrix[:num_nodes, :num_nodes]
        features = features[:num_nodes, :]

    if num_nodes < max_nodes:
        padded_features = np.zeros((max_nodes, feature_dim), dtype=np.float32)
        padded_adjacency_matrix = np.zeros((max_nodes, max_nodes), dtype=np.float32)
        padded_features[:num_nodes, :] = features[:num_nodes, :]
        padded_adjacency_matrix[:num_nodes, :num_nodes] = adjacency_matrix[:num_nodes, :num_nodes]
        features = padded_features
        adjacency_matrix = padded_adjacency_matrix
    
    prediction_mask = np.zeros((max_nodes,), dtype=bool)
    valid_prediction_node_ids = [idx for idx in prediction_node_ids if idx < num_nodes]
    prediction_mask[valid_prediction_node_ids] = True
    
    features = np.expand_dims(features, axis=0)
    adjacency_matrix = np.expand_dims(adjacency_matrix, axis=0)
    prediction_mask = np.expand_dims(prediction_mask, axis=0)
    
    batch_predictions = model.predict([features, adjacency_matrix, prediction_mask])

    if batch_predictions.shape[0] != len(valid_prediction_node_ids):
        logging.error(f"Model output shape {batch_predictions.shape} does not match the expected prediction indices.")
        return

    annotations = []
    counter=0
    
    for node_index in valid_prediction_node_ids:
        prediction = batch_predictions[counter, 0]
        counter+=1
        if prediction > 0.2:
            mapped_node_id = node_id_mapper.get_id(node_index)
            if mapped_node_id is None:
                logging.warning(f"Node index {node_index} not found in NodeIDMapper. Skipping.")
                continue

            file_name = file_mappings.get(mapped_node_id)
            if not file_name:
                logging.warning(f"No file mapping found for node_id {mapped_node_id}. Skipping annotation.")
                continue

            line_num = line_number_mapping.get(mapped_node_id, 0)
            if line_num == 0:
                logging.warning(f"Line number for node_id {mapped_node_id} not found. Skipping annotation.")
                continue
            
            node_name = combined_tdg.graph.nodes[mapped_node_id]['attr']['name']
            
            annotations.append((file_name, node_name, line_num))
    
    file_names = set([ann[0] for ann in annotations])
    
    for file_name in file_names:
        base_file_name = os.path.basename(file_name)
        relative_subdir = os.path.relpath(file_name, project_dir)
        output_file_path = os.path.join(output_dir, relative_subdir)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        file_annotations = [ann for ann in annotations if ann[0] == file_name]
        annotate_file(file_name, file_annotations, output_file_path)
    
    logging.info(f"Annotation complete for project {project_dir}")

def main(project_dir, model_path, output_dir, batch_size):
    model = load_model(model_path, custom_objects={'f1_score': f1_score, 'BooleanMaskLayer': BooleanMaskLayer})
    
    for subdir in os.listdir(project_dir):
        subdir_path = os.path.join(project_dir, subdir)
        if os.path.isdir(subdir_path):
            output_subdir = os.path.join(output_dir, subdir)
            os.makedirs(output_subdir, exist_ok=True)
            process_project(subdir_path, output_subdir, model, batch_size)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python predict.py <ProjectDir> <ModelPath> <OutputDir>")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO)

    project_dir = sys.argv[1]
    model_path = sys.argv[2]
    output_dir = sys.argv[3]
    batch_size = 1

    main(project_dir, model_path, output_dir, batch_size)
