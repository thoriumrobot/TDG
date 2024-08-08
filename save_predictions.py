import os
import sys
import csv
import numpy as np
from tensorflow.keras.models import load_model
import logging
import tensorflow as tf
from tdg_utils import f1_score, create_tf_dataset, process_java_file, NodeIDMapper, node_id_mapper

#node_id_mapper = NodeIDMapper()  # Initialize the node ID mapper

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

def process_project(project_dir, model, batch_size):
    file_list = [os.path.join(root, file)
                 for root, _, files in os.walk(project_dir)
                 for file in files if file.endswith('.java')]
    
    dataset = create_tf_dataset(file_list, batch_size, balance=False, is_tdg=False)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    predictions = []
    iterator = iter(dataset)
    
    try:
        while True:
            batch = next(iterator)
            features, labels, node_ids = batch
            batch_predictions = model.predict(features)
            for node_id, prediction in zip(node_ids.numpy(), batch_predictions):
                predictions.append((node_id, prediction[0]))  # Save the prediction and node_id
    except StopIteration:
        pass

    return predictions

def main(project_dir, model_path, output_dir, batch_size):
    model = load_model(model_path, custom_objects={'f1_score': f1_score})

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
    batch_size = 32

    main(project_dir, model_path, output_dir, batch_size)
