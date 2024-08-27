import os
import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import logging
import tensorflow as tf
from tdg_utils import f1_score, preprocess_tdg, process_java_file, NodeIDMapper, node_id_mapper, JavaTDG

@tf.keras.utils.register_keras_serializable()
class BooleanMaskLayer(Layer):
    def call(self, inputs):
        output, mask = inputs
        return tf.boolean_mask(output, mask)

    def compute_output_shape(self, input_shape):
        output_shape, mask_shape = input_shape
        # Since we are masking, the output shape is unknown until runtime
        return (None, output_shape[-1])  # Returns the correct shape after masking

def annotate_file(file_path, annotations, output_file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    for annotation in annotations:
        node_id, line_num, col_num = annotation
        if 0 <= line_num - 1 < len(lines):
            lines[line_num - 1] = lines[line_num - 1][:col_num] + "@Nullable " + lines[line_num - 1][col_num:]
        else:
            logging.warning(f"Line number {line_num} is out of range in file {file_path}")
    
    with open(output_file_path, 'w') as file:
        file.writelines(lines)

def create_combined_tdg(file_list):
    combined_tdg = JavaTDG()
    for file_path in file_list:
        process_java_file(file_path, combined_tdg)
    return combined_tdg

def process_project(project_dir, output_dir, model, batch_size):
    file_list = [os.path.join(root, file)
                 for root, _, files in os.walk(project_dir)
                 for file in files if file.endswith('.java')]

    combined_tdg = create_combined_tdg(file_list)
    features, _, node_ids, adjacency_matrix, prediction_node_ids = preprocess_tdg(combined_tdg)

    if features.size == 0 or adjacency_matrix.size == 0:
        logging.warning(f"No valid TDG created for project {project_dir}. Skipping.")
        return
    
    # Pad features, adjacency matrix, and create the prediction mask
    num_nodes = features.shape[0]
    max_nodes = 8000
    feature_dim = features.shape[1] if features.ndim > 1 else 4
    
    if num_nodes < max_nodes:
        padded_features = np.zeros((max_nodes, feature_dim), dtype=np.float32)
        padded_adjacency_matrix = np.zeros((max_nodes, max_nodes), dtype=np.float32)
        padded_features[:num_nodes, :] = features
        padded_adjacency_matrix[:adjacency_matrix.shape[0], :adjacency_matrix.shape[1]] = adjacency_matrix
        features = padded_features
        adjacency_matrix = padded_adjacency_matrix
    
    # Create prediction mask
    prediction_mask = np.zeros((max_nodes,), dtype=bool)
    prediction_mask[prediction_node_ids] = True
    
    # Expand dimensions to match model input expectations
    features = np.expand_dims(features, axis=0)
    adjacency_matrix = np.expand_dims(adjacency_matrix, axis=0)
    prediction_mask = np.expand_dims(prediction_mask, axis=0)
    
    # Run the model prediction
    batch_predictions = model.predict([features, adjacency_matrix, prediction_mask])

    annotations = []

    for node_index in prediction_node_ids:  # Iterate only over prediction nodes
        prediction = batch_predictions[node_index, 0]  # Use node index as the first index, output index as second
        if prediction > 0:
            mapped_node_id = node_id_mapper.get_id(node_index)
            if mapped_node_id is None:
                logging.warning(f"Node index {node_index} not found in NodeIDMapper. Skipping.")
                continue

            node_info = mapped_node_id.split('.')
            file_name = node_info[0]
            try:
                line_num = int(node_info[1])
            except ValueError:
                logging.warning(f"Invalid line number in node_id {mapped_node_id} for project {project_dir}. Skipping annotation.")
                continue
            col_num = 0

            annotations.append((file_name, line_num, col_num))
    
    # Annotate files with predictions
    for file_name in set([ann[0] for ann in annotations]):
        input_file_path = os.path.join(project_dir, file_name)
        output_file_path = os.path.join(output_dir, file_name)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        file_annotations = [ann for ann in annotations if ann[0] == file_name]
        annotate_file(input_file_path, file_annotations, output_file_path)
    
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
