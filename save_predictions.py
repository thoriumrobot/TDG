import os
import sys
import csv
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

def save_predictions_to_csv(predictions, output_dir):
    methods = []
    fields = []
    parameters = []

    for node_id, prediction in predictions:
        node_id = node_id_mapper.get_id(node_id)
        if node_id:
            node_info = node_id.split('.')
            node_type = node_info[-1]
            if '()' in node_type:
                methods.append((node_id, prediction))
            elif 'field' in node_type:
                fields.append((node_id, prediction))
            elif 'param' in node_type:
                parameters.append((node_id, prediction))

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'methods.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Node ID', 'Prediction'])
        for row in methods:
            writer.writerow(row)

    with open(os.path.join(output_dir, 'fields.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Node ID', 'Prediction'])
        for row in fields:
            writer.writerow(row)

    with open(os.path.join(output_dir, 'parameters.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Node ID', 'Prediction'])
        for row in parameters:
            writer.writerow(row)

def create_combined_tdg(file_list):
    combined_tdg = JavaTDG()
    for file_path in file_list:
        process_java_file(file_path, combined_tdg)
    return combined_tdg

def process_project(project_dir, model, batch_size):
    file_list = [os.path.join(root, file)
                 for root, _, files in os.walk(project_dir)
                 for file in files if file.endswith('.java')]

    combined_tdg = create_combined_tdg(file_list)
    features, _, node_ids, adjacency_matrix, prediction_node_ids = preprocess_tdg(combined_tdg)

    if features.size == 0 or adjacency_matrix.size == 0:
        logging.warning(f"No valid TDG created for project {project_dir}. Skipping.")
        return []
    
    # Use the actual number of nodes from features
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
    
    # Create prediction mask
    prediction_mask = np.zeros((max_nodes,), dtype=bool)
    valid_prediction_node_ids = [idx for idx in prediction_node_ids if idx < num_nodes]
    prediction_mask[valid_prediction_node_ids] = True
    
    # Expand dimensions to match model input expectations
    features = np.expand_dims(features, axis=0)
    adjacency_matrix = np.expand_dims(adjacency_matrix, axis=0)
    prediction_mask = np.expand_dims(prediction_mask, axis=0)
    
    # Run the model prediction
    batch_predictions = model.predict([features, adjacency_matrix, prediction_mask])

    if batch_predictions.shape[0] != len(valid_prediction_node_ids):
        logging.error(f"Model output shape {batch_predictions.shape} does not match the expected prediction indices.")
        return []

    predictions = []

    for node_index in valid_prediction_node_ids:  # Iterate only over valid prediction nodes
        if node_index >= batch_predictions.shape[0]:
            logging.error(f"Node index {node_index} is out of bounds for predictions of size {batch_predictions.shape[0]}. Skipping.")
            continue
        
        prediction = batch_predictions[node_index, 0]  # Use node index as the first index, output index as second
        predictions.append((node_index, prediction))

    return predictions

def main(project_dir, model_path, output_dir, batch_size):
    model = load_model(model_path, custom_objects={'f1_score': f1_score, 'BooleanMaskLayer': BooleanMaskLayer})

    all_predictions = []

    for subdir in os.listdir(project_dir):
        subdir_path = os.path.join(project_dir, subdir)
        if os.path.isdir(subdir_path):
            subdir_predictions = process_project(subdir_path, model, batch_size)
            all_predictions.extend(subdir_predictions)

    save_predictions_to_csv(all_predictions, output_dir)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python save_predictions.py <ProjectDir> <ModelPath> <OutputDir>")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO)

    project_dir = sys.argv[1]
    model_path = sys.argv[2]
    output_dir = sys.argv[3]
    batch_size = 1

    main(project_dir, model_path, output_dir, batch_size)
