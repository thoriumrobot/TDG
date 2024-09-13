import os
import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import logging
import tensorflow as tf
from tdg_utils import f1_score, preprocess_tdg, process_java_file, node_id_mapper, JavaTDG
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
    
    # Group annotations by line number
    annotations_by_line = {}
    for annotation in annotations:
        _, _, line_num = annotation
        annotations_by_line.setdefault(line_num, []).append(annotation)
    
    # Process lines in reverse order to avoid affecting line numbers
    for line_num in sorted(annotations_by_line.keys(), reverse=True):
        line = lines[line_num - 1]
        if "@Nullable" in line:
            continue  # Skip if already annotated
        
        # Insert @Nullable before the type
        tokens = line.strip().split()
        if len(tokens) >= 2:
            # Find the index of the type (assuming it's the first token after any modifiers)
            type_index = 0
            for i, token in enumerate(tokens):
                if token not in ['public', 'private', 'protected', 'static', 'final', 'abstract', 'synchronized', 'volatile', 'transient']:
                    type_index = i
                    break
            tokens.insert(type_index, '@Nullable')
            new_line = ' '.join(tokens) + '\n'
            lines[line_num - 1] = new_line
        else:
            logging.warning(f"Unable to annotate line {line_num} in file {file_path}: insufficient tokens.")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
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
            attr = node_data.get('attr', {})
            if 'attr' not in node_data or not attr:
                logging.warning(f"Node {node_id} is missing 'attr'. Skipping.")
                continue

            # Add node to combined_tdg
            combined_tdg.add_node(node_id, attr.get('type'), attr.get('name'),
                                  line_number=attr.get('line_number'),
                                  nullable=attr.get('nullable'),
                                  actual_type=attr.get('actual_type'))

            # Map node_id to file_path and line_number
            file_mappings[node_id] = file_path
            line_number_mapping[node_id] = attr.get('line_number')

        for from_node, to_node, edge_data in class_tdg.graph.edges(data=True):
            combined_tdg.add_edge(from_node, to_node, edge_data['type'])

    return combined_tdg, file_mappings, line_number_mapping

def process_project(project_dir, output_dir, model, batch_size):
    file_list = [os.path.join(root, file)
                 for root, _, files in os.walk(project_dir)
                 for file in files if file.endswith('.java')]

    combined_tdg, file_mappings, line_number_mapping = create_combined_tdg(file_list)
    features, _, node_ids, adjacency_matrix, prediction_node_ids = preprocess_tdg(combined_tdg)

    if features.size == 0 or adjacency_matrix.size == 0:
        logging.warning(f"No valid TDG created for project {project_dir}. Skipping.")
        return

    # Update node_id_mapper with node IDs
    for idx, node_id in enumerate(node_ids):
        node_id_mapper.id_to_int[node_id] = idx
        node_id_mapper.int_to_id[idx] = node_id

    prediction_mask = np.zeros(features.shape[0], dtype=bool)
    prediction_mask[prediction_node_ids] = True

    features = np.expand_dims(features, axis=0)
    adjacency_matrix = np.expand_dims(adjacency_matrix, axis=0)
    prediction_mask = np.expand_dims(prediction_mask, axis=0)

    batch_predictions = model.predict([features, adjacency_matrix, prediction_mask])

    annotations = []
    prediction_indices = np.where(prediction_mask[0])[0]
    counter = 0

    for idx in prediction_indices:
        prediction = batch_predictions[counter]
        counter += 1
        if prediction > 0.5:  # Use a threshold, e.g., 0.5
            node_id = node_id_mapper.get_id(idx)
            if node_id is None:
                logging.warning(f"Node index {idx} not found in NodeIDMapper. Skipping.")
                continue

            file_name = file_mappings.get(node_id)
            if not file_name:
                logging.warning(f"No file mapping found for node_id {node_id}. Skipping annotation.")
                continue

            line_num = line_number_mapping.get(node_id, 0)
            if line_num == 0:
                logging.warning(f"Line number for node_id {node_id} not found. Skipping annotation.")
                continue

            node_name = combined_tdg.graph.nodes[node_id]['attr']['name']

            annotations.append((file_name, node_name, line_num))

    file_names = set([ann[0] for ann in annotations])

    for file_name in file_names:
        relative_path = os.path.relpath(file_name, project_dir)
        output_file_path = os.path.join(output_dir, relative_path)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        file_annotations = [ann for ann in annotations if ann[0] == file_name]
        annotate_file(file_name, file_annotations, output_file_path)

    logging.info(f"Annotation complete for project {project_dir}")

def main(project_dir, model_path, output_dir, batch_size):
    model = load_model(model_path, custom_objects={'f1_score': f1_score, 'BooleanMaskLayer': BooleanMaskLayer})

    process_project(project_dir, output_dir, model, batch_size)

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
