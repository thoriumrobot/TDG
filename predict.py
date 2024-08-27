import os
import sys
import numpy as np
from tensorflow.keras.models import load_model
import logging
import traceback
import tensorflow as tf
from tdg_utils import f1_score, create_tf_dataset, process_java_file, NodeIDMapper, node_id_mapper

#node_id_mapper = NodeIDMapper()  # Initialize the node ID mapper

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
    features, _, node_ids, adjacency_matrix = preprocess_tdg(combined_tdg)  # Omit labels here

    if features.size == 0 or adjacency_matrix.size == 0:
        logging.warning(f"No valid TDG created for project {project_dir}. Skipping.")
        return
    
    dataset = tf.data.Dataset.from_tensor_slices((features, adjacency_matrix)).batch(batch_size)  # Remove labels
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    annotations = []

    for batch in dataset:
        features, adjacency_matrix = batch  # Adjust to match new structure
        batch_predictions = model.predict([features, adjacency_matrix])
        for node_id, prediction in zip(node_ids.numpy(), batch_predictions):
            if prediction > 0:  # Assuming a threshold of 0 for @Nullable annotation
                node_id = node_id_mapper.get_id(node_id)
                node_info = node_id.split('.')
                file_name = node_info[0]
                try:
                    line_num = int(node_info[1])
                except ValueError:
                    logging.warning(f"Invalid line number in node_id {node_id} for project {project_dir}. Skipping annotation.")
                    continue
                col_num = 0

                annotations.append((file_name, line_num, col_num))
    
    for file_name in set([ann[0] for ann in annotations]):
        input_file_path = os.path.join(project_dir, file_name)
        output_file_path = os.path.join(output_dir, file_name)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        file_annotations = [ann for ann in annotations if ann[0] == file_name]
        annotate_file(input_file_path, file_annotations, output_file_path)
    
    logging.info(f"Annotation complete for project {project_dir}")

def main(project_dir, model_path, output_dir, batch_size):
    model = load_model(model_path, custom_objects={'f1_score': f1_score})
    
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
    batch_size = 32

    main(project_dir, model_path, output_dir, batch_size)
